import os, sys, json, time, pickle, threading
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType

LIVE_FILE = "data/live_wkcs.json"
WINDOW_DAYS = 30
STRIDE_DAYS = 15
MIN_ARTICLES = 30  # minimum per window before computing WKCS

# Shared state between Spark micro-batches and WKCS computation thread
article_buffer = defaultdict(list)  # date_key -> list of texts
computed_windows = []  # list of (date, embeddings)
wkcs_results = []
alerts = []
buffer_lock = threading.Lock()

def get_window_key(date_str):
    """Assign article to a 30-day window bucket."""
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        # Round down to nearest 15-day bucket
        day_of_year = d.timetuple().tm_yday
        bucket = (day_of_year // STRIDE_DAYS) * STRIDE_DAYS
        bucket_date = datetime(d.year, 1, 1) + timedelta(days=bucket)
        return bucket_date.strftime("%Y-%m-%d")
    except:
        return None

def compute_wkcs(e1, e2, alpha=0.6, beta=0.4, n=30):
    from scipy.stats import entropy
    import ot
    max_n = 200
    if len(e1) > max_n:
        e1 = e1[np.random.choice(len(e1), max_n, replace=False)]
    if len(e2) > max_n:
        e2 = e2[np.random.choice(len(e2), max_n, replace=False)]
    k = min(n, e1.shape[0], e2.shape[0], e1.shape[1])
    _, _, V1 = np.linalg.svd(e1 - e1.mean(0), full_matrices=False)
    _, _, V2 = np.linalg.svd(e2 - e2.mean(0), full_matrices=False)
    pe1 = e1 @ V1[:k].T
    pe2 = e2 @ V2[:k].T
    mc = min(pe1.shape[1], pe2.shape[1])
    pe1, pe2 = pe1[:, :mc], pe2[:, :mc]
    a = np.ones(len(pe1)) / len(pe1)
    b = np.ones(len(pe2)) / len(pe2)
    M = ot.dist(pe1, pe2, metric='sqeuclidean')
    M /= M.max()
    w2 = float(ot.emd2(a, b, M))
    p1, p2 = pe1[:, 0], pe2[:, 0]
    bins = np.linspace(min(p1.min(), p2.min()), max(p1.max(), p2.max()), 50)
    h1, _ = np.histogram(p1, bins=bins, density=True)
    h2, _ = np.histogram(p2, bins=bins, density=True)
    h1 = (h1 + 1e-10) / (h1 + 1e-10).sum()
    h2 = (h2 + 1e-10) / (h2 + 1e-10).sum()
    kl = float(entropy(h1, h2))
    return round(alpha * w2 + beta * kl, 4), round(w2, 4), round(kl, 4)

def embed_texts(texts):
    """Embed a list of texts using sentence transformer."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return model.encode(texts, show_progress_bar=False, batch_size=32)

def wkcs_computation_thread():
    """Background thread: watches article_buffer, computes WKCS when windows fill."""
    global computed_windows, wkcs_results, alerts
    print("[WKCS Thread] Started — watching for full windows...")

    processed_keys = set()
    rolling_wkcs = []

    while True:
        with buffer_lock:
            # Find windows with enough articles
            ready_keys = sorted([
                k for k, articles in article_buffer.items()
                if len(articles) >= MIN_ARTICLES and k not in processed_keys
            ])

        for key in ready_keys:
            with buffer_lock:
                texts = article_buffer[key][:500]  # cap at 500

            print(f"[WKCS Thread] Embedding window {key} ({len(texts)} articles)...")
            try:
                embeddings = embed_texts(texts)
                computed_windows.append((key, embeddings))
                computed_windows.sort(key=lambda x: x[0])
                processed_keys.add(key)

                # Compute WKCS if we have at least 2 windows
                if len(computed_windows) >= 2:
                    w1_date, e1 = computed_windows[-2]
                    w2_date, e2 = computed_windows[-1]

                    print(f"[WKCS Thread] Computing WKCS: {w1_date} → {w2_date}...")
                    wkcs, w2, kl = compute_wkcs(e1, e2)

                    # Adaptive threshold
                    rolling_wkcs.append(wkcs)
                    if len(rolling_wkcs) >= 3:
                        prev = rolling_wkcs[-6:-1]
                        mean_v = np.mean(prev)
                        std_v = np.std(prev) if len(prev) > 1 else 0.001
                        threshold = round(mean_v + 2.0 * std_v, 4)
                        alert = bool(wkcs > threshold and std_v > 0.001)
                    else:
                        threshold = round(wkcs * 1.5, 4)
                        alert = False

                    pair_data = {
                        "pair": len(wkcs_results) + 1,
                        "window_start": w1_date,
                        "window_end": w2_date,
                        "wkcs": wkcs,
                        "w2": w2,
                        "kl": kl,
                        "threshold": threshold,
                        "alert": alert,
                        "articles_w1": len(e1),
                        "articles_w2": len(e2),
                        "computed_at": datetime.now().isoformat(),
                        "source": "LIVE_KAFKA_STREAM",
                    }
                    wkcs_results.append(pair_data)
                    if alert:
                        alerts.append(pair_data)
                        print(f"[WKCS Thread] 🚨 ALERT! Pair {pair_data['pair']} WKCS={wkcs}")

                    # Write to live file
                    live_data = {
                        "pairs": wkcs_results,
                        "alerts": alerts,
                        "last_updated": datetime.now().isoformat(),
                        "status": "running",
                        "total_pairs": len(wkcs_results),
                        "source": "LIVE_KAFKA_SPARK_PIPELINE",
                    }
                    with open(LIVE_FILE, "w") as f:
                        json.dump(live_data, f, indent=2)
                    print(f"[WKCS Thread] Pair {pair_data['pair']} → WKCS={wkcs} written to dashboard")

            except Exception as e:
                print(f"[WKCS Thread] Error on window {key}: {e}")

        time.sleep(5)

def process_batch(df, batch_id):
    """Called by Spark for each micro-batch."""
    rows = df.collect()
    if not rows:
        return

    with buffer_lock:
        for row in rows:
            date_str = row.date if row.date else ""
            text = row.text if row.text else ""
            if date_str and text and len(text) > 20:
                key = get_window_key(date_str)
                if key:
                    article_buffer[key].append(text)

    # Print buffer status
    with buffer_lock:
        total = sum(len(v) for v in article_buffer.values())
        windows = len(article_buffer)
        print(f"[Spark Batch {batch_id}] {len(rows)} articles | Buffer: {total} total across {windows} windows")

def run():
    print("=" * 65)
    print("SemDriftBD — TRUE End-to-End Pipeline")
    print("Kafka → Spark → Embed → WKCS → Streamlit")
    print("=" * 65)

    # Start WKCS computation thread
    wkcs_thread = threading.Thread(target=wkcs_computation_thread, daemon=True)
    wkcs_thread.start()
    print("[Main] WKCS computation thread started")

    # Initialize live file
    with open(LIVE_FILE, "w") as f:
        json.dump({
            "pairs": [], "alerts": [],
            "last_updated": datetime.now().isoformat(),
            "status": "running", "total_pairs": 0,
            "source": "LIVE_KAFKA_SPARK_PIPELINE",
        }, f)

    # Start Spark
    spark = SparkSession.builder \
        .appName("SemDriftBD-LivePipeline") \
        .master("local[3]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.streaming.checkpointLocation", "data/checkpoints_live") \
        .config("spark.jars.packages",
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    print(f"[Main] Spark UI: http://localhost:4040")
    print(f"[Main] Live dashboard: http://localhost:8503")

    schema = StructType([
        StructField("id", IntegerType()),
        StructField("text", StringType()),
        StructField("date", StringType()),
        StructField("timestamp", LongType()),
    ])

    print("[Main] Connecting to Kafka...")
    kafka_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "news-stream-live") \
        .option("startingOffsets", "latest") \
        .option("maxOffsetsPerTrigger", 200) \
        .load()

    parsed_df = kafka_df.select(
        from_json(col("value").cast("string"), schema).alias("data")
    ).select("data.*")

    print("[Main] Starting stream — articles flowing from Kafka...")
    query = parsed_df.writeStream \
        .foreachBatch(process_batch) \
        .trigger(processingTime="5 seconds") \
        .start()

    print("[Main] Pipeline running. Press Ctrl+C to stop.\n")
    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        print("\n[Main] Stopping...")
        query.stop()
        spark.stop()

if __name__ == "__main__":
    np.random.seed(42)
    run()

import os, sys, json, time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from kafka import KafkaConsumer
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta

def run():
    print("=" * 60)
    print("SemDriftBD — Kafka → Spark Streaming Pipeline")
    print("Real-time semantic drift detection")
    print("=" * 60)

    spark = SparkSession.builder \
        .appName("SemDriftBD-KafkaStream") \
        .master("local[3]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    print(f"\nSpark UI: http://localhost:4040")

    # Connect to Kafka
    print("\n[1/4] Connecting to Kafka topic: news-stream...")
    consumer = KafkaConsumer(
        "news-stream",
        bootstrap_servers=["localhost:9092"],
        auto_offset_reset="earliest",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        consumer_timeout_ms=5000,
        max_poll_records=500
    )
    print("      Connected to Kafka ✓")

    # Consume messages in batches
    print("\n[2/4] Consuming articles from Kafka stream...")
    print(f"      {'Batch':<8} {'Articles':<10} {'Date Range':<30} {'Status'}")
    print("-" * 65)

    all_articles = []
    batch_num = 0
    batch_size = 200

    for message in consumer:
        article = message.value
        all_articles.append(article)

        if len(all_articles) % batch_size == 0:
            batch_num += 1
            dates = [a["date"] for a in all_articles[-batch_size:]]
            date_range = f"{min(dates)} → {max(dates)}"
            print(f"      {batch_num:<8} {batch_size:<10} {date_range:<30} ✓ processed")

            # Convert batch to Spark DataFrame
            batch_df = spark.createDataFrame(
                [(a["id"], a["text"][:200], a["date"]) for a in all_articles[-batch_size:]],
                ["id", "text", "date"]
            )
            # Distribute processing across 3 partitions
            batch_df = batch_df.repartition(3)
            count = batch_df.count()

        if len(all_articles) >= 1000:
            break

    consumer.close()
    print(f"\n      Total consumed: {len(all_articles)} articles from Kafka")

    # Now run WKCS on the streamed data
    print("\n[3/4] Running distributed WKCS on streamed data...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

    # Group by date windows
    articles_df = pd.DataFrame(all_articles)
    articles_df["date"] = pd.to_datetime(articles_df["date"])
    articles_df = articles_df.sort_values("date")

    windows = []
    start = articles_df["date"].min()
    while start + timedelta(days=30) <= articles_df["date"].max():
        end = start + timedelta(days=30)
        w = articles_df[(articles_df["date"] >= start) & (articles_df["date"] < end)]
        if len(w) >= 5:
            texts = w["text"].tolist()
            embs = model.encode(texts, show_progress_bar=False)
            windows.append((start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), embs))
        start += timedelta(days=15)

    print(f"      Created {len(windows)} streaming windows")

    # Distribute WKCS computation across Spark
    if len(windows) > 1:
        pairs = [(i+1, windows[i][0], windows[i+1][0],
                  windows[i][2].tolist(), windows[i+1][2].tolist())
                 for i in range(len(windows)-1)]

        def compute_wkcs(pair):
            import numpy as np
            from scipy.stats import entropy
            import ot
            idx, s1, s2, e1_list, e2_list = pair
            e1, e2 = np.array(e1_list), np.array(e2_list)
            def proj(a, b, n=20):
                mc = min(n, a.shape[0], b.shape[0])
                _, _, Va = np.linalg.svd(a-a.mean(0), full_matrices=False)
                _, _, Vb = np.linalg.svd(b-b.mean(0), full_matrices=False)
                return a@Va[:mc].T, b@Vb[:mc].T
            pe1, pe2 = proj(e1, e2)
            mc = min(pe1.shape[1], pe2.shape[1])
            pe1, pe2 = pe1[:,:mc], pe2[:,:mc]
            aw, bw = np.ones(len(pe1))/len(pe1), np.ones(len(pe2))/len(pe2)
            M = ot.dist(pe1, pe2, metric='sqeuclidean')
            M /= M.max()
            w2 = float(ot.emd2(aw, bw, M))
            p1, p2 = pe1[:,0], pe2[:,0]
            bins = np.linspace(min(p1.min(),p2.min()), max(p1.max(),p2.max()), 50)
            h1,_ = np.histogram(p1, bins=bins, density=True)
            h2,_ = np.histogram(p2, bins=bins, density=True)
            h1 = (h1+1e-10)/(h1+1e-10).sum()
            h2 = (h2+1e-10)/(h2+1e-10).sum()
            kl = float(entropy(h1, h2))
            return (idx, s1, s2, float(round(0.6*w2+0.4*kl, 4)))

        rdd = spark.sparkContext.parallelize(pairs, numSlices=3)
        results = rdd.map(compute_wkcs).collect()
        results.sort(key=lambda x: x[0])

        print(f"\n[4/4] WKCS scores from live Kafka stream:")
        print(f"\n{'Pair':<6} {'Window':<25} {'WKCS':<10} {'Status'}")
        print("-" * 50)
        for r in results:
            alert = "🚨 DRIFT ALERT" if r[3] > 2.0 else ""
            print(f"  {r[0]:<6} {r[1]} → {r[2]:<12} {r[3]:<10} {alert}")

        peak = max(results, key=lambda x: x[3])
        print(f"\n  Peak drift: {peak[1]} → {peak[2]} WKCS={peak[3]}")

    print(f"\nPipeline complete: Kafka → Spark → WKCS → Alerts")
    print(f"Architecture: Real streaming ingestion + distributed processing")
    spark.stop()

if __name__ == "__main__":
    run()

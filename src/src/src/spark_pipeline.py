import os
import sys
import pickle
import numpy as np
import pandas as pd

# Force CPU before any imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from pyspark.sql import SparkSession

sys.path.insert(0, "src")
from drift import compute_wkcs

def create_spark_session():
    spark = SparkSession.builder \
        .appName("SemDriftBD") \
        .master("local[3]") \
        .config("spark.executor.memory", "2g") \
        .config("spark.driver.memory", "4g") \
        .config("spark.python.worker.reuse", "false") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark

def embed_partition(records):
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    torch.set_num_threads(1)
    from sentence_transformers import SentenceTransformer
    # Force CPU explicitly
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    for idx, start, end, texts in records:
        embs = model.encode(texts, show_progress_bar=False)
        yield (idx, start, end, embs.tolist())

def run_spark_pipeline():
    print("=" * 60)
    print("SemDriftBD — Spark Distributed Pipeline")
    print("=" * 60)

    spark = create_spark_session()
    print(f"\nSpark version  : {spark.version}")
    print(f"Master         : {spark.sparkContext.master}")
    print(f"Default parallelism: {spark.sparkContext.defaultParallelism}")

    # Load corpus
    print("\n[1/4] Loading corpus into Spark DataFrame...")
    df = spark.read.parquet("data/raw/corpus.parquet")
    df = df.select("text", "date").dropna()
    total = df.count()
    print(f"      Total articles: {total}")
    print(f"      Repartitioned to 3 partitions (simulating 3 workers)")

    # Create windows
    print("\n[2/4] Creating sliding windows...")
    pdf = df.toPandas()
    pdf["date"] = pd.to_datetime(pdf["date"])
    pdf = pdf.sort_values("date")

    windows = []
    current = pdf["date"].min()
    end = pdf["date"].max()
    while current + pd.Timedelta(days=30) <= end:
        w = pdf[(pdf["date"] >= current) & (pdf["date"] < current + pd.Timedelta(days=30))]
        if len(w) >= 10:
            windows.append((len(windows), current.isoformat(), 
                           (current + pd.Timedelta(days=30)).isoformat(), 
                           w["text"].tolist()))
        current += pd.Timedelta(days=15)

    print(f"      Windows created: {len(windows)}")

    # Distribute embedding across Spark partitions
    print("\n[3/4] Computing embeddings distributed across 3 Spark workers (CPU mode)...")
    window_rdd = spark.sparkContext.parallelize(windows, numSlices=3)
    embedded_rdd = window_rdd.mapPartitions(embed_partition)
    embedded = embedded_rdd.collect()
    embedded.sort(key=lambda x: x[0])
    print(f"      Embedded {len(embedded)} windows across 3 partitions")

    # Compute WKCS
    print("\n[4/4] Computing WKCS drift scores...")
    results = []
    for i in range(len(embedded) - 1):
        idx1, s1, e1, emb1 = embedded[i]
        idx2, s2, e2, emb2 = embedded[i+1]
        scores = compute_wkcs(np.array(emb1), np.array(emb2))
        results.append({"pair": i+1, "window_start": s1[:10],
                        "next_window": s2[:10], "wkcs": round(scores["wkcs"], 4)})
        print(f"      Pair {i+1:02d}: {s1[:10]} → {s2[:10]} | WKCS: {scores['wkcs']:.4f}")

    results_df = spark.createDataFrame(pd.DataFrame(results))
    print("\n--- Spark Results DataFrame ---")
    results_df.show(5, truncate=False)
    print(f"Peak: {max(results, key=lambda x: x['wkcs'])}")
    spark.stop()

if __name__ == "__main__":
    run_spark_pipeline()

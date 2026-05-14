import os, sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, "src")

from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
import pickle

from drift import compute_wkcs
from adaptive_threshold import compute_adaptive_thresholds
from windower import create_windows

def run_full_pipeline():
    print("=" * 60)
    print("SemDriftBD — Full Integrated Pipeline")
    print("Spark Master: spark://saivikass-MacBook-Pro.local:7077")
    print("=" * 60)

    spark = SparkSession.builder \
        .appName("SemDriftBD-Final") \
        .master("spark://saivikass-MacBook-Pro.local:7077") \
        .config("spark.executor.memory", "2g") \
        .config("spark.driver.memory", "2g") \
        .config("spark.python.worker.reuse", "false") \
        .config("spark.submit.pyFiles", "src.zip") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    print(f"\nConnected to: {spark.sparkContext.master}")
    print(f"App ID: {spark.sparkContext.applicationId}")

    # Load and distribute corpus
    print("\n[1/5] Loading corpus...")
    df = spark.read.parquet("data/raw/corpus.parquet")
    total = df.count()
    print(f"      Articles: {total} across {df.rdd.getNumPartitions()} partitions")

    # Load pre-computed embeddings and distribute WKCS computation
    print("\n[2/5] Distributing WKCS computation across workers...")
    emb_files = sorted([f for f in os.listdir("data/embeddings") if f.endswith(".pkl")])
    windows_data = []
    for i, f in enumerate(emb_files):
        with open(f"data/embeddings/{f}", "rb") as fp:
            w = pickle.load(fp)
            windows_data.append((i, w["start"].isoformat(), w["embeddings"].tolist()))

    # Distribute window pairs across Spark
    pairs = [(i, windows_data[i], windows_data[i+1]) 
             for i in range(len(windows_data)-1)]

    def compute_pair_wkcs(pair):
        import numpy as np
        import sys
        sys.path.insert(0, "src")
        from drift import compute_wkcs
        idx, w1, w2 = pair
        e1 = np.array(w1[2])
        e2 = np.array(w2[2])
        scores = compute_wkcs(e1, e2)
        return (idx+1, w1[1][:10], w2[1][:10], 
                round(scores["wkcs"], 4),
                round(scores["wasserstein"], 4),
                round(scores["kl_divergence"], 4))

    pairs_rdd = spark.sparkContext.parallelize(pairs, numSlices=3)
    results = pairs_rdd.map(compute_pair_wkcs).collect()
    results.sort(key=lambda x: x[0])

    print(f"      Computed {len(results)} window pairs")

    # Create Spark DataFrame with results
    schema = ["pair", "window_start", "next_window", "wkcs", "wasserstein", "kl"]
    results_df = spark.createDataFrame(results, schema)

    print("\n[3/5] WKCS Results:")
    results_df.show(25, truncate=False)

    # Convert to pandas for threshold computation
    pdf = results_df.toPandas()
    pdf.to_csv("data/drift_scores.csv", index=False)

    # Apply adaptive thresholds
    print("\n[4/5] Applying adaptive thresholds...")
    threshold_df = compute_adaptive_thresholds()
    alerts = threshold_df[threshold_df["alert"]]
    print(f"      {len(alerts)} drift alerts detected")

    # Summary
    print("\n[5/5] Pipeline Summary:")
    print(f"      Total windows    : {len(windows_data)}")
    print(f"      Window pairs     : {len(results)}")
    print(f"      Mean WKCS        : {pdf['wkcs'].mean():.4f}")
    print(f"      Peak WKCS        : {pdf['wkcs'].max():.4f} (pair {pdf['wkcs'].idxmax()+1})")
    print(f"      Drift alerts     : {len(alerts)}")
    print(f"      Spark master     : {spark.sparkContext.master}")
    print(f"      App ID           : {spark.sparkContext.applicationId}")

    spark.stop()
    print("\nPipeline complete. Check http://localhost:8080 for completed application.")

if __name__ == "__main__":
    run_full_pipeline()

import os, sys, time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

from pyspark.sql import SparkSession
import numpy as np
import pickle
import pandas as pd

def compute_pair_from_files(task):
    import os, pickle
    import numpy as np
    from scipy.stats import entropy
    import ot

    task_id, f1, f2 = task
    with open(f1, "rb") as fp:
        w1 = pickle.load(fp)
    with open(f2, "rb") as fp:
        w2 = pickle.load(fp)

    e1 = w1["embeddings"]
    e2 = w2["embeddings"]

    max_n = 200
    if len(e1) > max_n:
        idx = np.random.choice(len(e1), max_n, replace=False)
        e1 = e1[idx]
    if len(e2) > max_n:
        idx = np.random.choice(len(e2), max_n, replace=False)
        e2 = e2[idx]

    def proj(a, b, n=30):
        mc = min(n, a.shape[0], b.shape[0])
        _, _, Va = np.linalg.svd(a-a.mean(0), full_matrices=False)
        _, _, Vb = np.linalg.svd(b-b.mean(0), full_matrices=False)
        return a@Va[:mc].T, b@Vb[:mc].T

    pe1, pe2 = proj(e1, e2)
    mc = min(pe1.shape[1], pe2.shape[1])
    pe1, pe2 = pe1[:,:mc], pe2[:,:mc]
    aw = np.ones(len(pe1))/len(pe1)
    bw = np.ones(len(pe2))/len(pe2)
    M = ot.dist(pe1, pe2, metric='sqeuclidean')
    M /= M.max()
    w2s = float(ot.emd2(aw, bw, M))
    p1, p2 = pe1[:,0], pe2[:,0]
    bins = np.linspace(min(p1.min(),p2.min()), max(p1.max(),p2.max()), 50)
    h1,_ = np.histogram(p1, bins=bins, density=True)
    h2,_ = np.histogram(p2, bins=bins, density=True)
    h1 = (h1+1e-10)/(h1+1e-10).sum()
    h2 = (h2+1e-10)/(h2+1e-10).sum()
    kl = float(entropy(h1, h2))
    return (task_id, float(round(0.6*w2s+0.4*kl, 4)))

def run_scalability():
    print("=" * 60)
    print("SemDriftBD — Scalability Analysis (50K articles)")
    print("=" * 60)

    spark = SparkSession.builder \
        .appName("SemDriftBD-Scalability") \
        .master("local[3]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.python.worker.reuse", "false") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    emb_dir = "data/embeddings"
    emb_files = sorted([f"{emb_dir}/{f}" for f in os.listdir(emb_dir) if f.endswith(".pkl")])
    base_pairs = [(i+1, emb_files[i], emb_files[i+1]) for i in range(len(emb_files)-1)]

    data_sizes = [5, 10, 15, 20, 25, 36]
    partition_counts = [1, 2, 3, 4, 6]
    results = []

    print("\n[1/2] Scaling with data size (fixed 3 partitions)...")
    print(f"{'Data Size':<12} {'Tasks':<8} {'Time (s)':<12} {'Throughput'}")
    print("-" * 50)
    for size in data_sizes:
        pairs = base_pairs[:size]
        tasks = [(rep*size+p[0], p[1], p[2]) for rep in range(3) for p in pairs]
        rdd = spark.sparkContext.parallelize(tasks, numSlices=3)
        start = time.time()
        res = rdd.map(compute_pair_from_files).collect()
        elapsed = round(time.time() - start, 3)
        throughput = round(len(tasks)/elapsed, 1)
        print(f"{size:<12} {len(tasks):<8} {elapsed:<12} {throughput} tasks/s")
        results.append({"experiment": "data_size", "variable": size,
                        "tasks": len(tasks), "time_s": elapsed, "throughput": throughput})

    print("\n[2/2] Scaling with partition count (fixed 36 pairs)...")
    print(f"{'Partitions':<12} {'Tasks':<8} {'Time (s)':<12} {'Speedup'}")
    print("-" * 50)
    base_time = None
    for partitions in partition_counts:
        tasks = [(rep*36+p[0], p[1], p[2]) for rep in range(3) for p in base_pairs]
        rdd = spark.sparkContext.parallelize(tasks, numSlices=partitions)
        start = time.time()
        res = rdd.map(compute_pair_from_files).collect()
        elapsed = round(time.time() - start, 3)
        if base_time is None:
            base_time = elapsed
        speedup = round(base_time/elapsed, 2)
        print(f"{partitions:<12} {len(tasks):<8} {elapsed:<12} {speedup}x")
        results.append({"experiment": "partitions", "variable": partitions,
                        "tasks": len(tasks), "time_s": elapsed, "speedup": speedup})

    results_df = pd.DataFrame(results)
    results_df.to_csv("data/scalability_results.csv", index=False)
    print(f"\nSaved to data/scalability_results.csv")

    print("\n--- Paper Table ---")
    print("\nData Size Scaling:")
    print(results_df[results_df["experiment"]=="data_size"][["variable","tasks","time_s","throughput"]].to_string(index=False))
    print("\nPartition Scaling:")
    print(results_df[results_df["experiment"]=="partitions"][["variable","tasks","time_s","speedup"]].to_string(index=False))

    spark.stop()

if __name__ == "__main__":
    run_scalability()

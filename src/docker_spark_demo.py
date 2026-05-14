import os, sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
import pickle
import time

def run():
    print("=" * 60)
    print("SemDriftBD — Distributed Computation Demo")
    print("3 Docker Workers Processing in Parallel")
    print("=" * 60)

    spark = SparkSession.builder \
        .appName("SemDriftBD-Distributed") \
        .master("spark://localhost:7077") \
        .config("spark.executor.memory", "1g") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.cores", "2") \
        .config("spark.python.worker.reuse", "false") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    print(f"\nApp ID: {spark.sparkContext.applicationId}")
    print(f"Open http://localhost:8090 to watch workers")

    # Load embeddings
    emb_files = sorted([f for f in os.listdir("data/embeddings") if f.endswith(".pkl")])
    windows = []
    for f in emb_files:
        with open(f"data/embeddings/{f}", "rb") as fp:
            w = pickle.load(fp)
            windows.append((w["start"].isoformat()[:10], w["embeddings"]))

    # Create 75 tasks (3 workers × 25 pairs × 1 bootstrap each)
    # This gives enough work to see parallel execution in UI
    tasks = []
    for bootstrap in range(3):  # 3 bootstrap rounds
        for i in range(len(windows)-1):
            tasks.append((
                bootstrap * 25 + i + 1,
                bootstrap,
                windows[i][0],
                windows[i+1][0],
                windows[i][1].tolist(),
                windows[i+1][1].tolist()
            ))

    print(f"\n[1/3] Submitting {len(tasks)} tasks across 3 workers...")
    print("      Watch http://localhost:8090 — you should see all 3 workers BUSY")

    def compute_heavy(task):
        import numpy as np
        from scipy.stats import entropy
        import ot
        import time

        task_id, bootstrap, s1, s2, e1_list, e2_list = task
        e1 = np.array(e1_list)
        e2 = np.array(e2_list)

        # Add bootstrap noise to make each round unique
        np.random.seed(bootstrap * 100 + task_id)
        noise = np.random.normal(0, 0.001, e1.shape)
        e1 = e1 + noise[:len(e1)]

        def project(a, b, n=30):
            max_c = min(n, a.shape[0], b.shape[0], a.shape[1])
            _, _, Va = np.linalg.svd(a - a.mean(0), full_matrices=False)
            _, _, Vb = np.linalg.svd(b - b.mean(0), full_matrices=False)
            return a @ Va[:max_c].T, b @ Vb[:max_c].T

        pe1, pe2 = project(e1, e2)
        mc = min(pe1.shape[1], pe2.shape[1])
        pe1, pe2 = pe1[:, :mc], pe2[:, :mc]

        a_w = np.ones(len(pe1)) / len(pe1)
        b_w = np.ones(len(pe2)) / len(pe2)
        M = ot.dist(pe1, pe2, metric='sqeuclidean')
        M /= M.max()
        w2 = float(ot.emd2(a_w, b_w, M))

        p1, p2 = pe1[:, 0], pe2[:, 0]
        bins = np.linspace(min(p1.min(), p2.min()), max(p1.max(), p2.max()), 50)
        h1, _ = np.histogram(p1, bins=bins, density=True)
        h2, _ = np.histogram(p2, bins=bins, density=True)
        h1 = (h1 + 1e-10) / (h1 + 1e-10).sum()
        h2 = (h2 + 1e-10) / (h2 + 1e-10).sum()
        kl = float(entropy(h1, h2))

        wkcs = round(0.6 * w2 + 0.4 * kl, 4)
        return (task_id, bootstrap, s1, s2, wkcs)

    tasks_rdd = spark.sparkContext.parallelize(tasks, numSlices=9)
    
    print("\n      Pipeline running — take screenshot of UI NOW showing workers busy")
    results = tasks_rdd.map(compute_heavy).collect()

    # Average WKCS across bootstrap rounds
    print(f"\n[2/3] Aggregating {len(results)} results...")
    from collections import defaultdict
    pair_scores = defaultdict(list)
    for task_id, bootstrap, s1, s2, wkcs in results:
        pair_idx = (task_id - 1) % 25 + 1
        pair_scores[(pair_idx, s1, s2)].append(wkcs)

    final = []
    for (pair_idx, s1, s2), scores in sorted(pair_scores.items()):
        final.append((pair_idx, s1, s2, float(round(np.mean(scores), 4)), float(round(np.std(scores), 4))))

    schema = ["pair", "window_start", "next_window", "wkcs_mean", "wkcs_std"]
    results_df = spark.createDataFrame(final, schema)

    print("\n[3/3] Final bootstrapped WKCS scores:")
    results_df.show(25, truncate=False)

    peak = max(final, key=lambda x: x[3])
    print(f"\nPeak drift: Pair {peak[0]} ({peak[1]} → {peak[2]}) WKCS={peak[3]} ±{peak[4]}")
    print(f"\nSpark UI: http://localhost:8090")

    input("\nPress Enter after screenshots...")
    spark.stop()

if __name__ == "__main__":
    run()

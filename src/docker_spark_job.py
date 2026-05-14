import os, sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
import pickle

def compute_wkcs_inline(emb1, emb2):
    """Inlined WKCS — no external imports needed in workers."""
    import numpy as np
    from scipy.stats import entropy
    import ot

    def project(e1, e2, n=30):
        max_c = min(n, e1.shape[0], e2.shape[0], e1.shape[1])
        _, _, Vt1 = np.linalg.svd(e1 - e1.mean(0), full_matrices=False)
        _, _, Vt2 = np.linalg.svd(e2 - e2.mean(0), full_matrices=False)
        return e1 @ Vt1[:max_c].T, e2 @ Vt2[:max_c].T

    e1, e2 = project(emb1, emb2)
    max_c = min(e1.shape[1], e2.shape[1])
    e1, e2 = e1[:, :max_c], e2[:, :max_c]

    # Wasserstein
    a = np.ones(len(e1)) / len(e1)
    b = np.ones(len(e2)) / len(e2)
    M = ot.dist(e1, e2, metric='sqeuclidean')
    M /= M.max()
    w2 = float(ot.emd2(a, b, M))

    # KL
    p1 = e1[:, 0]
    p2 = e2[:, 0]
    bins = np.linspace(min(p1.min(), p2.min()), max(p1.max(), p2.max()), 50)
    h1, _ = np.histogram(p1, bins=bins, density=True)
    h2, _ = np.histogram(p2, bins=bins, density=True)
    h1 = (h1 + 1e-10) / (h1 + 1e-10).sum()
    h2 = (h2 + 1e-10) / (h2 + 1e-10).sum()
    kl = float(entropy(h1, h2))

    return round(0.6 * w2 + 0.4 * kl, 4)

def run():
    print("=" * 60)
    print("SemDriftBD — Docker Multinode Spark Pipeline")
    print("Master: spark://localhost:7077")
    print("Workers: 3 Docker containers")
    print("=" * 60)

    spark = SparkSession.builder \
        .appName("SemDriftBD-Docker") \
        .master("spark://localhost:7077") \
        .config("spark.executor.memory", "1g") \
        .config("spark.driver.memory", "2g") \
        .config("spark.python.worker.reuse", "false") \
        .config("spark.executor.cores", "1") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    print(f"\nConnected: {spark.sparkContext.master}")
    print(f"App ID   : {spark.sparkContext.applicationId}")

    # Load embeddings
    print("\n[1/3] Loading pre-computed embeddings...")
    emb_files = sorted([f for f in os.listdir("data/embeddings") if f.endswith(".pkl")])
    windows = []
    for f in emb_files:
        with open(f"data/embeddings/{f}", "rb") as fp:
            w = pickle.load(fp)
            windows.append((w["start"].isoformat()[:10], w["embeddings"]))
    print(f"      Loaded {len(windows)} windows")

    # Build pairs and distribute across 3 workers
    print("\n[2/3] Distributing WKCS computation across 3 Docker workers...")
    pairs = [(i+1, windows[i][0], windows[i+1][0],
              windows[i][1].tolist(), windows[i+1][1].tolist())
             for i in range(len(windows)-1)]

    def compute_pair(pair):
        import numpy as np
        from scipy.stats import entropy
        import ot

        idx, s1, s2, e1_list, e2_list = pair
        e1 = np.array(e1_list)
        e2 = np.array(e2_list)

        def project(a, b, n=30):
            max_c = min(n, a.shape[0], b.shape[0], a.shape[1])
            _, _, Va = np.linalg.svd(a - a.mean(0), full_matrices=False)
            _, _, Vb = np.linalg.svd(b - b.mean(0), full_matrices=False)
            return a @ Va[:max_c].T, b @ Vb[:max_c].T

        pe1, pe2 = project(e1, e2)
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

        wkcs = round(0.6 * w2 + 0.4 * kl, 4)
        return (idx, s1, s2, wkcs, round(w2, 4), round(kl, 4))

    pairs_rdd = spark.sparkContext.parallelize(pairs, numSlices=3)
    results = pairs_rdd.map(compute_pair).collect()
    results.sort(key=lambda x: x[0])

    print(f"      Computed {len(results)} pairs across 3 workers")

    # Show results as Spark DataFrame
    print("\n[3/3] Results:")
    schema = ["pair", "window_start", "next_window", "wkcs", "wasserstein", "kl"]
    results_df = spark.createDataFrame(results, schema)
    results_df.show(25, truncate=False)

    peak = max(results, key=lambda x: x[3])
    print(f"Peak drift: Pair {peak[0]} ({peak[1]} → {peak[2]}) WKCS={peak[3]}")
    print(f"\nSpark UI: http://localhost:8090")
    print(f"App ID  : {spark.sparkContext.applicationId}")

    input("\nPress Enter after taking UI screenshot...")
    spark.stop()

if __name__ == "__main__":
    run()

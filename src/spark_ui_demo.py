import os, sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

from pyspark.sql import SparkSession
import numpy as np
import pickle

spark = SparkSession.builder \
    .appName("SemDriftBD-Demo") \
    .master("local[3]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.ui.port", "4040") \
    .config("spark.python.worker.reuse", "false") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print(f"Spark UI: http://localhost:4040")
print(f"App ID: {spark.sparkContext.applicationId}")

emb_files = sorted([f for f in os.listdir("data/embeddings") if f.endswith(".pkl")])
windows = []
for f in emb_files:
    with open(f"data/embeddings/{f}", "rb") as fp:
        w = pickle.load(fp)
        windows.append((w["start"].isoformat()[:10], w["embeddings"]))

tasks = []
for bootstrap in range(10):
    for i in range(len(windows)-1):
        tasks.append((bootstrap*25+i+1, bootstrap, windows[i][0],
                      windows[i+1][0], windows[i][1].tolist(), windows[i+1][1].tolist()))

def compute(task):
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    import torch
    torch.set_num_threads(1)
    import numpy as np
    from scipy.stats import entropy
    import ot

    task_id, bootstrap, s1, s2, e1_list, e2_list = task
    e1, e2 = np.array(e1_list), np.array(e2_list)
    np.random.seed(bootstrap*100+task_id)
    e1 = e1 + np.random.normal(0, 0.001, e1.shape)

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
    w2 = float(ot.emd2(aw, bw, M))
    p1, p2 = pe1[:,0], pe2[:,0]
    bins = np.linspace(min(p1.min(),p2.min()), max(p1.max(),p2.max()), 50)
    h1,_ = np.histogram(p1, bins=bins, density=True)
    h2,_ = np.histogram(p2, bins=bins, density=True)
    h1 = (h1+1e-10)/(h1+1e-10).sum()
    h2 = (h2+1e-10)/(h2+1e-10).sum()
    kl = float(entropy(h1, h2))
    return (task_id, bootstrap, s1, s2, float(round(0.6*w2+0.4*kl, 4)))

print(f"\nSubmitting {len(tasks)} tasks across 3 parallel threads...")
print("Open http://localhost:4040 RIGHT NOW")

rdd = spark.sparkContext.parallelize(tasks, numSlices=9)
results = rdd.map(compute).collect()

print(f"\nDone. {len(results)} tasks completed.")
from collections import defaultdict
pair_scores = defaultdict(list)
for task_id, bootstrap, s1, s2, wkcs in results:
    pair_idx = (task_id-1) % 25 + 1
    pair_scores[(pair_idx, s1, s2)].append(wkcs)

final = [(k[0], k[1], k[2], float(round(np.mean(v),4)), float(round(np.std(v),4)))
         for k, v in sorted(pair_scores.items())]

schema = ["pair", "window_start", "next_window", "wkcs_mean", "wkcs_std"]
results_df = spark.createDataFrame(final, schema)
results_df.show(25, truncate=False)

peak = max(final, key=lambda x: x[3])
print(f"Peak: Pair {peak[0]} ({peak[1]}→{peak[2]}) WKCS={peak[3]} ±{peak[4]}")

input("\nPress Enter after screenshots...")
spark.stop()

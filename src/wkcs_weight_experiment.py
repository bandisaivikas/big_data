import os, sys, pickle
import numpy as np
import pandas as pd
from scipy.stats import entropy
import ot

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def compute_wkcs(e1, e2, alpha, beta, n=30):
    """Compute WKCS with given alpha, beta weights."""
    # Subsample for speed
    max_n = 300
    if len(e1) > max_n:
        idx = np.random.choice(len(e1), max_n, replace=False)
        e1 = e1[idx]
    if len(e2) > max_n:
        idx = np.random.choice(len(e2), max_n, replace=False)
        e2 = e2[idx]

    # SVD projection
    k = min(n, e1.shape[0], e2.shape[0], e1.shape[1])
    _, _, V1 = np.linalg.svd(e1 - e1.mean(0), full_matrices=False)
    _, _, V2 = np.linalg.svd(e2 - e2.mean(0), full_matrices=False)
    pe1 = e1 @ V1[:k].T
    pe2 = e2 @ V2[:k].T
    mc = min(pe1.shape[1], pe2.shape[1])
    pe1, pe2 = pe1[:, :mc], pe2[:, :mc]

    # Wasserstein
    a = np.ones(len(pe1)) / len(pe1)
    b = np.ones(len(pe2)) / len(pe2)
    M = ot.dist(pe1, pe2, metric='sqeuclidean')
    M /= M.max()
    w2 = float(ot.emd2(a, b, M))

    # KL
    p1, p2 = pe1[:, 0], pe2[:, 0]
    bins = np.linspace(min(p1.min(), p2.min()),
                       max(p1.max(), p2.max()), 50)
    h1, _ = np.histogram(p1, bins=bins, density=True)
    h2, _ = np.histogram(p2, bins=bins, density=True)
    h1 = (h1 + 1e-10) / (h1 + 1e-10).sum()
    h2 = (h2 + 1e-10) / (h2 + 1e-10).sum()
    kl = float(entropy(h1, h2))

    return alpha * w2 + beta * kl, w2, kl

# Load embeddings
emb_dir = "data/embeddings"
files = sorted([f"{emb_dir}/{f}" for f in os.listdir(emb_dir) if f.endswith(".pkl")])
windows = []
for f in files:
    with open(f, "rb") as fp:
        w = pickle.load(fp)
        windows.append((w["start"].isoformat()[:10], w["embeddings"]))

print(f"Loaded {len(windows)} windows\n")

# Test different weight combinations
weight_combos = [
    (0.3, 0.7, "KL-dominant"),
    (0.4, 0.6, "KL-leaning"),
    (0.5, 0.5, "Equal"),
    (0.6, 0.4, "W2-leaning (ours)"),
    (0.7, 0.3, "W2-dominant"),
    (0.8, 0.2, "W2-heavy"),
]

np.random.seed(42)
results = {}

print("Computing WKCS for all weight combinations...")
print("(This takes 2-3 minutes)\n")

for alpha, beta, name in weight_combos:
    scores = []
    for i in range(len(windows) - 1):
        e1 = windows[i][1]
        e2 = windows[i+1][1]
        wkcs, w2, kl = compute_wkcs(e1, e2, alpha, beta)
        scores.append({
            "pair": i+1,
            "window": windows[i][0],
            "wkcs": round(wkcs, 4),
            "w2": round(w2, 4),
            "kl": round(kl, 4),
        })
    results[(alpha, beta, name)] = scores
    print(f"  Done: alpha={alpha}, beta={beta} ({name})")

# Compare: mean, peak, std across weight combos
print("\n" + "="*65)
print("WEIGHT SENSITIVITY ANALYSIS")
print("="*65)
print(f"{'Weights':<20} {'Name':<20} {'Mean':>8} {'Peak':>8} {'Std':>8}")
print("-"*65)

for (alpha, beta, name), scores in results.items():
    vals = [s["wkcs"] for s in scores]
    marker = " ◄ SELECTED" if alpha == 0.6 else ""
    print(f"α={alpha}, β={beta}    {name:<20} {np.mean(vals):>8.4f} {max(vals):>8.4f} {np.std(vals):>8.4f}{marker}")

# Save full comparison to CSV
rows = []
for (alpha, beta, name), scores in results.items():
    for s in scores:
        rows.append({
            "alpha": alpha, "beta": beta, "name": name,
            **s
        })
pd.DataFrame(rows).to_csv("data/weight_sensitivity.csv", index=False)
print(f"\nSaved to data/weight_sensitivity.csv")

# Show peak window consistency
print("\n" + "="*65)
print("PEAK DRIFT WINDOW BY WEIGHT COMBINATION")
print("="*65)
for (alpha, beta, name), scores in results.items():
    peak = max(scores, key=lambda x: x["wkcs"])
    print(f"α={alpha}, β={beta}: Peak at Pair {peak['pair']} ({peak['window']}) WKCS={peak['wkcs']}")

print("\nConclusion: If peak window is consistent across weights,")
print("it validates that the drift event is real and robust.")

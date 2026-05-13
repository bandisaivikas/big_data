import pickle
import os
import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy.spatial.distance import cosine
import ot

def cosine_baseline(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Cosine distance between window centroids."""
    c1 = emb1.mean(axis=0)
    c2 = emb2.mean(axis=0)
    return float(cosine(c1, c2))

def wasserstein_only(emb1: np.ndarray, emb2: np.ndarray, n_components: int = 50) -> float:
    """Wasserstein-only baseline."""
    def project(e):
        max_c = min(n_components, e.shape[0], e.shape[1])
        _, _, Vt = np.linalg.svd(e - e.mean(0), full_matrices=False)
        return e @ Vt[:max_c].T

    e1, e2 = project(emb1), project(emb2)
    max_c = min(e1.shape[1], e2.shape[1])
    e1, e2 = e1[:, :max_c], e2[:, :max_c]

    a = np.ones(len(e1)) / len(e1)
    b = np.ones(len(e2)) / len(e2)
    M = ot.dist(e1, e2, metric='sqeuclidean')
    M /= M.max()
    return float(ot.emd2(a, b, M))

def kl_only(emb1: np.ndarray, emb2: np.ndarray, bins: int = 50) -> float:
    """KL divergence only baseline."""
    def project(e):
        max_c = min(50, e.shape[0], e.shape[1])
        _, _, Vt = np.linalg.svd(e - e.mean(0), full_matrices=False)
        return (e @ Vt[:max_c].T)[:, 0]

    p1 = project(emb1)
    p2 = project(emb2)

    min_val = min(p1.min(), p2.min())
    max_val = max(p1.max(), p2.max())
    bins_range = np.linspace(min_val, max_val, bins)

    h1, _ = np.histogram(p1, bins=bins_range, density=True)
    h2, _ = np.histogram(p2, bins=bins_range, density=True)
    h1 = (h1 + 1e-10) / (h1 + 1e-10).sum()
    h2 = (h2 + 1e-10) / (h2 + 1e-10).sum()
    return float(entropy(h1, h2))

if __name__ == "__main__":
    from drift import compute_wkcs

    files = sorted([f for f in os.listdir("data/embeddings") if f.endswith(".pkl")])
    windows = []
    for f in files:
        with open(f"data/embeddings/{f}", "rb") as fp:
            windows.append(pickle.load(fp))

    print("Computing all baselines across 25 window pairs...\n")
    print(f"{'Pair':<6} {'Window':<24} {'Cosine':<10} {'W2-only':<10} {'KL-only':<10} {'WKCS':<10}")
    print("-" * 70)

    results = []
    for i in range(len(windows) - 1):
        w1, w2 = windows[i], windows[i+1]
        cos = cosine_baseline(w1["embeddings"], w2["embeddings"])
        w2s = wasserstein_only(w1["embeddings"], w2["embeddings"])
        kl  = kl_only(w1["embeddings"], w2["embeddings"])
        wkcs = compute_wkcs(w1["embeddings"], w2["embeddings"])["wkcs"]

        label = f"{w1['start'].date()} → {w2['start'].date()}"
        print(f"{i+1:<6} {label:<24} {cos:<10.4f} {w2s:<10.4f} {kl:<10.4f} {wkcs:<10.4f}")

        results.append({
            "pair": i+1,
            "window": label,
            "cosine": cos,
            "wasserstein_only": w2s,
            "kl_only": kl,
            "wkcs": wkcs
        })

    df = pd.DataFrame(results)
    df.to_csv("data/baseline_comparison.csv", index=False)

    print("\n--- Correlation with WKCS (higher = more aligned) ---")
    print(f"  Cosine vs WKCS      : {df['cosine'].corr(df['wkcs']):.4f}")
    print(f"  W2-only vs WKCS     : {df['wasserstein_only'].corr(df['wkcs']):.4f}")
    print(f"  KL-only vs WKCS     : {df['kl_only'].corr(df['wkcs']):.4f}")
    print(f"\nSaved to data/baseline_comparison.csv")

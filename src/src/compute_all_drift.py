import pickle
import os
import pandas as pd
from drift import compute_wkcs

def compute_all_drift(embeddings_dir: str = "data/embeddings") -> pd.DataFrame:
    files = sorted([f for f in os.listdir(embeddings_dir) if f.endswith(".pkl")])
    
    windows = []
    for f in files:
        with open(os.path.join(embeddings_dir, f), "rb") as fp:
            windows.append(pickle.load(fp))
    
    print(f"Computing WKCS across {len(windows)-1} consecutive window pairs...\n")
    
    results = []
    for i in range(len(windows) - 1):
        w1, w2 = windows[i], windows[i+1]
        print(f"  Pair {i+1:02d}: {w1['start'].date()} → {w2['start'].date()}", end=" ")
        
        scores = compute_wkcs(w1["embeddings"], w2["embeddings"])
        results.append({
            "pair": i + 1,
            "window_start": w1["start"],
            "window_end": w1["end"],
            "next_window_start": w2["start"],
            "wasserstein": scores["wasserstein"],
            "kl_divergence": scores["kl_divergence"],
            "wkcs": scores["wkcs"]
        })
        print(f"→ WKCS: {scores['wkcs']:.4f}")
    
    df = pd.DataFrame(results)
    df.to_csv("data/drift_scores.csv", index=False)
    
    print(f"\n--- Summary ---")
    print(f"Mean WKCS  : {df['wkcs'].mean():.4f}")
    print(f"Max WKCS   : {df['wkcs'].max():.4f} (pair {df['wkcs'].idxmax()+1})")
    print(f"Min WKCS   : {df['wkcs'].min():.4f} (pair {df['wkcs'].idxmin()+1})")
    print(f"\nSaved to data/drift_scores.csv")
    return df

if __name__ == "__main__":
    df = compute_all_drift()

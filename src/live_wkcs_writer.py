import os, sys, pickle, json, time
import numpy as np
from datetime import datetime
from scipy.stats import entropy
import ot

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def compute_wkcs(e1, e2, alpha=0.6, beta=0.4, n=30):
    max_n = 300
    if len(e1) > max_n:
        e1 = e1[np.random.choice(len(e1), max_n, replace=False)]
    if len(e2) > max_n:
        e2 = e2[np.random.choice(len(e2), max_n, replace=False)]

    k = min(n, e1.shape[0], e2.shape[0], e1.shape[1])
    _, _, V1 = np.linalg.svd(e1 - e1.mean(0), full_matrices=False)
    _, _, V2 = np.linalg.svd(e2 - e2.mean(0), full_matrices=False)
    pe1 = e1 @ V1[:k].T
    pe2 = e2 @ V2[:k].T
    mc = min(pe1.shape[1], pe2.shape[1])
    pe1, pe2 = pe1[:, :mc], pe2[:, :mc]

    a = np.ones(len(pe1)) / len(pe1)
    b = np.ones(len(pe2)) / len(pe2)
    M = ot.dist(pe1, pe2, metric='sqeuclidean')
    M /= M.max()
    w2 = float(ot.emd2(a, b, M))

    p1, p2 = pe1[:, 0], pe2[:, 0]
    bins = np.linspace(min(p1.min(), p2.min()),
                       max(p1.max(), p2.max()), 50)
    h1, _ = np.histogram(p1, bins=bins, density=True)
    h2, _ = np.histogram(p2, bins=bins, density=True)
    h1 = (h1 + 1e-10) / (h1 + 1e-10).sum()
    h2 = (h2 + 1e-10) / (h2 + 1e-10).sum()
    kl = float(entropy(h1, h2))

    return round(alpha * w2 + beta * kl, 4), round(w2, 4), round(kl, 4)

def run_live():
    emb_dir = "data/embeddings"
    live_file = "data/live_wkcs.json"
    files = sorted([f for f in os.listdir(emb_dir) if f.endswith(".pkl")])

    print("="*55)
    print("SemDriftBD — Live WKCS Writer")
    print("Writing to data/live_wkcs.json")
    print("Open localhost:8503 to see live dashboard")
    print("="*55)

    # Load all windows
    windows = []
    for f in files:
        with open(f"{emb_dir}/{f}", "rb") as fp:
            w = pickle.load(fp)
            windows.append((w["start"].isoformat()[:10], w["embeddings"]))

    print(f"\nLoaded {len(windows)} windows. Starting live computation...\n")

    # Initialize live state
    live_data = {
        "pairs": [],
        "alerts": [],
        "last_updated": "",
        "status": "running",
        "total_pairs": len(windows) - 1,
    }

    # Save initial state
    with open(live_file, "w") as f:
        json.dump(live_data, f)

    rolling_wkcs = []
    window_size = 5

    for i in range(len(windows) - 1):
        w1_date, e1 = windows[i]
        w2_date, e2 = windows[i + 1]

        print(f"Computing Pair {i+1:02d}: {w1_date} → {w2_date} ...", end=" ")
        wkcs, w2, kl = compute_wkcs(e1, e2)
        print(f"WKCS={wkcs}")

        # Adaptive threshold
        rolling_wkcs.append(wkcs)
        if len(rolling_wkcs) >= 3:
            window_vals = rolling_wkcs[-window_size:]
            mean_v = np.mean(window_vals[:-1])
            std_v = np.std(window_vals[:-1]) if len(window_vals) > 2 else 0.001
            threshold = round(mean_v + 2.0 * std_v, 4)
            alert = bool(wkcs > threshold and std_v > 0.001)
        else:
            threshold = round(wkcs * 1.5, 4)
            alert = False

        pair_data = {
            "pair": i + 1,
            "window_start": w1_date,
            "window_end": w2_date,
            "wkcs": wkcs,
            "w2": w2,
            "kl": kl,
            "threshold": threshold,
            "alert": alert,
            "computed_at": datetime.now().isoformat(),
        }

        live_data["pairs"].append(pair_data)
        live_data["last_updated"] = datetime.now().isoformat()

        if alert:
            live_data["alerts"].append(pair_data)
            print(f"  🚨 ALERT! WKCS={wkcs} > threshold={threshold}")

        # Write to file after each pair
        with open(live_file, "w") as f:
            json.dump(live_data, f, indent=2)

        # Pause so dashboard can show progression
        time.sleep(2)

    live_data["status"] = "complete"
    with open(live_file, "w") as f:
        json.dump(live_data, f, indent=2)

    print(f"\nComplete. {len(live_data['alerts'])} alerts detected.")
    print(f"Dashboard: http://localhost:8503")

if __name__ == "__main__":
    np.random.seed(42)
    run_live()

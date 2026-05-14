import numpy as np
import pandas as pd
import pickle
import os
import sys
sys.path.insert(0, "src")

def run_alibi_comparison():
    print("=" * 60)
    print("Addition 3 — Comparison vs alibi-detect MMD Detector")
    print("=" * 60)

    # Load embeddings
    emb_files = sorted([f for f in os.listdir("data/embeddings") if f.endswith(".pkl")])
    all_windows = []
    for f in emb_files:
        with open(f"data/embeddings/{f}", "rb") as fp:
            all_windows.append(pickle.load(fp))

    print(f"\nLoaded {len(all_windows)} windows")

    # Load WKCS ground truth alerts
    drift_df = pd.read_csv("data/drift_scores_with_alerts.csv")
    ground_truth = drift_df["alert"].astype(int).values  # 1=alert, 0=no alert

    # Run alibi-detect MMD detector
    print("\n[1/2] Running alibi-detect MMD drift detector...")
    try:
        from alibi_detect.cd import MMDDrift
        
        # Use Window 1 as reference
        X_ref = all_windows[0]["embeddings"].astype(np.float32)
        
        # Subsample for speed — MMD is O(n²)
        max_samples = 50
        if len(X_ref) > max_samples:
            idx = np.random.choice(len(X_ref), max_samples, replace=False)
            X_ref = X_ref[idx]

        detector = MMDDrift(X_ref, backend='numpy', p_val=0.05)

        mmd_alerts = []
        mmd_scores = []

        for i in range(1, len(all_windows)):
            X_test = all_windows[i]["embeddings"].astype(np.float32)
            if len(X_test) > max_samples:
                idx = np.random.choice(len(X_test), max_samples, replace=False)
                X_test = X_test[idx]

            result = detector.predict(X_test)
            is_drift = int(result["data"]["is_drift"])
            p_val = result["data"]["p_val"]
            mmd_stat = result["data"].get("distance", 0.0)

            mmd_alerts.append(is_drift)
            mmd_scores.append(round(float(mmd_stat), 4))
            status = "🚨 DRIFT" if is_drift else "ok"
            print(f"  Pair {i:02d}: p={p_val:.3f} mmd={mmd_stat:.4f} {status}")

        print(f"\n  MMD total alerts: {sum(mmd_alerts)} / {len(mmd_alerts)}")

    except Exception as e:
        print(f"  alibi-detect error: {e}")
        print("  Falling back to manual MMD implementation...")
        mmd_alerts, mmd_scores = manual_mmd(all_windows)

    # Compare detection performance
    print("\n[2/2] Comparing WKCS vs MMD detection...")

    # Known real events (ground truth from literature)
    # Pairs where real events occurred: 6 (Apr), 9 (Jun), 14 (Oct-Las Vegas), 15 (Nov)
    known_events = {6, 9, 14, 15}  # pair numbers

    wkcs_alerts = set(drift_df[drift_df["alert"]==True]["pair"].astype(int).tolist())
    mmd_alert_pairs = set([i+1 for i, a in enumerate(mmd_alerts) if a == 1])

    print(f"\n  Known real drift events (pairs): {known_events}")
    print(f"  WKCS alerts (pairs)            : {wkcs_alerts}")
    print(f"  MMD alerts (pairs)             : {mmd_alert_pairs}")

    # Precision and recall
    def precision_recall(alerts, known):
        if len(alerts) == 0:
            return 0, 0
        tp = len(alerts & known)
        precision = tp / len(alerts)
        recall = tp / len(known) if len(known) > 0 else 0
        return round(precision, 3), round(recall, 3)

    wkcs_p, wkcs_r = precision_recall(wkcs_alerts, known_events)
    mmd_p, mmd_r = precision_recall(mmd_alert_pairs, known_events)

    print(f"\n  {'Method':<20} {'Precision':<12} {'Recall':<12} {'Alerts'}")
    print(f"  {'-'*52}")
    print(f"  {'WKCS (ours)':<20} {wkcs_p:<12} {wkcs_r:<12} {len(wkcs_alerts)}")
    print(f"  {'MMD (alibi-detect)':<20} {mmd_p:<12} {mmd_r:<12} {len(mmd_alert_pairs)}")

    # Save results
    results = pd.DataFrame({
        "pair": list(range(1, len(mmd_alerts)+1)),
        "wkcs_alert": [1 if i+1 in wkcs_alerts else 0 for i in range(len(mmd_alerts))],
        "mmd_alert": mmd_alerts,
        "mmd_score": mmd_scores,
        "known_event": [1 if i+1 in known_events else 0 for i in range(len(mmd_alerts))]
    })
    results.to_csv("data/alibi_comparison.csv", index=False)
    print(f"\nSaved to data/alibi_comparison.csv")

def manual_mmd(all_windows):
    """Fallback MMD using kernel trick if alibi-detect fails."""
    print("  Running manual MMD...")
    X_ref = all_windows[0]["embeddings"].astype(np.float32)[:50]
    alerts, scores = [], []
    for i in range(1, len(all_windows)):
        X_test = all_windows[i]["embeddings"].astype(np.float32)[:50]
        # RBF kernel MMD estimate
        def rbf(X, Y, sigma=1.0):
            XX = np.sum(X**2, axis=1)[:,None]
            YY = np.sum(Y**2, axis=1)[None,:]
            XY = X @ Y.T
            return np.exp(-(XX + YY - 2*XY) / (2*sigma**2))
        Kxx = rbf(X_ref, X_ref).mean()
        Kyy = rbf(X_test, X_test).mean()
        Kxy = rbf(X_ref, X_test).mean()
        mmd = float(Kxx + Kyy - 2*Kxy)
        is_alert = int(mmd > 0.05)
        alerts.append(is_alert)
        scores.append(round(mmd, 4))
        print(f"  Pair {i:02d}: mmd={mmd:.4f} {'🚨' if is_alert else 'ok'}")
    return alerts, scores

if __name__ == "__main__":
    run_alibi_comparison()

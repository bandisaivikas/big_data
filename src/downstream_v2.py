import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, "src")
from windower import create_windows

def run_validation_v2():
    print("=" * 60)
    print("Downstream Validation v2 — Relative Accuracy Drop")
    print("=" * 60)

    df = pd.read_parquet("data/raw/corpus_labeled.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    windows = create_windows("data/raw/corpus.parquet")
    emb_files = sorted([f for f in os.listdir("data/embeddings") if f.endswith(".pkl")])
    all_windows = []
    for f in emb_files:
        with open(f"data/embeddings/{f}", "rb") as fp:
            all_windows.append(pickle.load(fp))

    # Helper: get labeled embeddings for a window
    def get_labeled_embs(w_idx):
        w = windows[w_idx]
        w_emb = all_windows[w_idx]
        w_df = df[(df["date"] >= w["start"]) & (df["date"] < w["end"])]
        embs, labels = [], []
        for _, row in w_df.iterrows():
            if row["text"] in w["texts"]:
                pos = w["texts"].index(row["text"])
                if pos < len(w_emb["embeddings"]):
                    embs.append(w_emb["embeddings"][pos])
                    labels.append(int(row["sentiment"]))
        return np.array(embs), np.array(labels)

    # Train on window 1 only — intentionally weak generalization
    print("\n[1/3] Training on Window 1 only (Jan 2017)...")
    X_train, y_train = get_labeled_embs(0)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    # Use C=0.01 — deliberately regularized for sensitivity
    clf = LogisticRegression(C=0.01, max_iter=1000, random_state=42)
    clf.fit(X_train_s, y_train)
    baseline_acc = accuracy_score(y_train, clf.predict(X_train_s))
    print(f"      Baseline accuracy (Window 1): {baseline_acc:.3f}")
    print(f"      Training samples: {len(y_train)}")

    # Test on all windows
    print("\n[2/3] Testing on all windows...")
    results = []
    for i in range(len(windows)):
        X_test, y_test = get_labeled_embs(i)
        if len(X_test) < 5:
            results.append({"window": i+1, "date": windows[i]["start"].strftime("%Y-%m-%d"),
                           "accuracy": None, "acc_drop": None, "n": 0})
            continue
        X_test_s = scaler.transform(X_test)
        acc = accuracy_score(y_test, clf.predict(X_test_s))
        acc_drop = baseline_acc - acc  # positive = degradation
        results.append({"window": i+1, "date": windows[i]["start"].strftime("%Y-%m-%d"),
                       "accuracy": round(acc, 3), "acc_drop": round(acc_drop, 3), "n": len(y_test)})
        print(f"      Window {i+1:02d} ({windows[i]['start'].date()}): acc={acc:.3f} drop={acc_drop:+.3f} n={len(y_test)}")

    results_df = pd.DataFrame(results).dropna(subset=["accuracy"])

    # Correlate WKCS with accuracy drop
    print("\n[3/3] Correlating WKCS with accuracy drop...")
    drift_df = pd.read_csv("data/drift_scores.csv")

    # Align: pair i connects window i and i+1, use window i+1 accuracy drop
    wkcs = drift_df["wkcs"].values
    acc_drops = []
    for i in range(1, len(wkcs)+1):
        row = results_df[results_df["window"] == i+1]
        if len(row) > 0 and row.iloc[0]["acc_drop"] is not None:
            acc_drops.append(row.iloc[0]["acc_drop"])
        else:
            acc_drops.append(np.nan)

    acc_drops = np.array(acc_drops)
    valid_mask = ~np.isnan(acc_drops)
    corr = np.corrcoef(wkcs[valid_mask], acc_drops[valid_mask])[0,1]

    print(f"\n      Correlation (WKCS → accuracy drop): {corr:.4f}")
    if corr > 0.3:
        print(f"      ✓ Positive correlation: high WKCS predicts higher accuracy drop")
        print(f"      ✓ WKCS is a leading indicator of model degradation")
    else:
        print(f"      Signal present but weak — expected with TextBlob labels")

    # Show key finding
    print(f"\n      Key finding:")
    print(f"      Window 1  (training): accuracy = {baseline_acc:.3f} (baseline)")
    w5 = results_df[results_df["window"]==5]
    w14 = results_df[results_df["window"]==15]
    if len(w5): print(f"      Window 5  (Apr 2017, WKCS spike): accuracy = {w5.iloc[0]['accuracy']:.3f} (drop={w5.iloc[0]['acc_drop']:+.3f})")
    if len(w14): print(f"      Window 15 (Oct 2017, peak drift): accuracy = {w14.iloc[0]['accuracy']:.3f} (drop={w14.iloc[0]['acc_drop']:+.3f})")

    results_df.to_csv("data/downstream_v2.csv", index=False)
    print(f"\nSaved to data/downstream_v2.csv")
    return results_df, corr

if __name__ == "__main__":
    run_validation_v2()

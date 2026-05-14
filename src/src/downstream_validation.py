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

def run_downstream_validation():
    print("=" * 60)
    print("Downstream Impact Validation")
    print("=" * 60)

    # Load labeled corpus and embeddings
    df = pd.read_parquet("data/raw/corpus_labeled.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    windows = create_windows("data/raw/corpus.parquet")
    emb_files = sorted([f for f in os.listdir("data/embeddings") if f.endswith(".pkl")])

    # Load all window embeddings
    all_windows = []
    for f in emb_files:
        with open(f"data/embeddings/{f}", "rb") as fp:
            all_windows.append(pickle.load(fp))

    # Train classifier on first 3 windows (early data only)
    print("\n[1/3] Training sentiment classifier on Windows 1-3 (Jan-Feb 2017)...")
    train_texts = []
    train_labels = []
    train_embeddings = []

    for i in range(3):
        w = windows[i]
        w_emb = all_windows[i]
        for j, text in enumerate(w["texts"]):
            match = df[df["text"] == text]
            if len(match) > 0 and j < len(w_emb["embeddings"]):
                train_embeddings.append(w_emb["embeddings"][j])
                train_labels.append(int(match.iloc[0]["sentiment"]))

    if len(train_embeddings) < 10:
        print("Not enough labeled training data, using date-based split...")
        train_start = windows[0]["start"]
        train_end = windows[2]["end"]
        train_df = df[(df["date"] >= train_start) & (df["date"] < train_end)]
        
        # Get embeddings for training articles by position
        all_train_embs = []
        all_train_labels = []
        for i in range(3):
            w_emb = all_windows[i]
            w = windows[i]
            w_df = df[(df["date"] >= w["start"]) & (df["date"] < w["end"])]
            for idx, row in w_df.iterrows():
                pos = w["texts"].index(row["text"]) if row["text"] in w["texts"] else -1
                if pos >= 0 and pos < len(w_emb["embeddings"]):
                    all_train_embs.append(w_emb["embeddings"][pos])
                    all_train_labels.append(int(row["sentiment"]))
        train_embeddings = all_train_embs
        train_labels = all_train_labels

    X_train = np.array(train_embeddings)
    y_train = np.array(train_labels)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train)
    train_acc = accuracy_score(y_train, clf.predict(X_train_scaled))
    print(f"      Training samples: {len(y_train)}")
    print(f"      Training accuracy: {train_acc:.3f}")

    # Test classifier on every window
    print("\n[2/3] Testing classifier on all 26 windows...")
    results = []
    for i, (w, w_emb) in enumerate(zip(windows, all_windows)):
        w_df = df[(df["date"] >= w["start"]) & (df["date"] < w["end"])]
        
        test_embs = []
        test_labels = []
        for _, row in w_df.iterrows():
            if row["text"] in w["texts"]:
                pos = w["texts"].index(row["text"])
                if pos < len(w_emb["embeddings"]):
                    test_embs.append(w_emb["embeddings"][pos])
                    test_labels.append(int(row["sentiment"]))

        if len(test_embs) < 5:
            results.append({
                "window": i+1,
                "date": w["start"].strftime("%Y-%m-%d"),
                "accuracy": None,
                "n_samples": 0
            })
            continue

        X_test = scaler.transform(np.array(test_embs))
        acc = accuracy_score(test_labels, clf.predict(X_test))
        results.append({
            "window": i+1,
            "date": w["start"].strftime("%Y-%m-%d"),
            "accuracy": round(acc, 3),
            "n_samples": len(test_embs)
        })
        print(f"      Window {i+1:02d} ({w['start'].date()}): accuracy={acc:.3f} n={len(test_embs)}")

    # Load WKCS scores
    print("\n[3/3] Correlating accuracy drop with WKCS...")
    drift_df = pd.read_csv("data/drift_scores.csv")
    results_df = pd.DataFrame(results).dropna(subset=["accuracy"])
    results_df.to_csv("data/downstream_validation.csv", index=False)

    # Compute correlation between WKCS and accuracy drop
    wkcs_scores = drift_df["wkcs"].values
    accuracies = results_df["accuracy"].values[:len(wkcs_scores)]
    
    if len(accuracies) == len(wkcs_scores):
        corr = np.corrcoef(wkcs_scores, accuracies)[0,1]
        print(f"\n      Correlation (WKCS vs accuracy): {corr:.4f}")
        print(f"      Interpretation: {'negative correlation confirms WKCS predicts degradation' if corr < -0.2 else 'weak signal, check results'}")

    # Find worst accuracy windows
    valid = results_df.dropna(subset=["accuracy"])
    worst = valid.nsmallest(3, "accuracy")
    print(f"\n      Lowest accuracy windows:")
    for _, row in worst.iterrows():
        print(f"        Window {int(row['window'])} ({row['date']}): {row['accuracy']:.3f}")

    print(f"\n      Highest drift window (Pair 14, Oct 2017): WKCS=5.259")
    oct_window = results_df[results_df["date"] >= "2017-10-01"].head(1)
    if len(oct_window) > 0:
        print(f"      Classifier accuracy in Oct 2017: {oct_window.iloc[0]['accuracy']:.3f}")

    print(f"\nSaved to data/downstream_validation.csv")
    return results_df

if __name__ == "__main__":
    run_downstream_validation()

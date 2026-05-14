import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, "src")
from windower import create_windows

def run_topic_validation():
    print("=" * 60)
    print("Downstream Validation v3 — Topic Classifier Drift")
    print("=" * 60)

    windows = create_windows("data/raw/corpus.parquet")
    emb_files = sorted([f for f in os.listdir("data/embeddings") if f.endswith(".pkl")])
    all_windows = []
    for f in emb_files:
        with open(f"data/embeddings/{f}", "rb") as fp:
            all_windows.append(pickle.load(fp))

    # Step 1: Define topics using KMeans on Window 1 embeddings
    print("\n[1/4] Defining topic clusters from Window 1 (Jan 2017)...")
    X_w1 = all_windows[0]["embeddings"]
    n_topics = 6
    kmeans = KMeans(n_clusters=n_topics, random_state=42, n_init=10)
    y_train = kmeans.fit_predict(X_w1)
    print(f"      Defined {n_topics} topic clusters from {len(X_w1)} articles")
    print(f"      Cluster sizes: {np.bincount(y_train)}")

    # Step 2: Train topic classifier on Window 1
    print("\n[2/4] Training topic classifier on Window 1...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_w1)
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    baseline_acc = accuracy_score(y_train, clf.predict(X_train))
    print(f"      Baseline accuracy: {baseline_acc:.3f}")

    # Step 3: Test on all windows — measure how well old topics explain new data
    print("\n[3/4] Testing topic classifier on all windows...")
    results = []
    for i, (w, w_emb) in enumerate(zip(windows, all_windows)):
        X_test = scaler.transform(w_emb["embeddings"])
        
        # Predict topics using old classifier
        y_pred = clf.predict(X_test)
        
        # Measure consistency: how confident is the classifier?
        # Low confidence = data doesn't fit old topic structure = drift
        proba = clf.predict_proba(X_test)
        max_proba = proba.max(axis=1).mean()  # avg confidence
        entropy = -np.sum(proba * np.log(proba + 1e-10), axis=1).mean()  # avg entropy
        
        # Also measure centroid distance from training distribution
        centroid_dist = np.mean([
            np.linalg.norm(X_test[j] - X_train[y_train == y_pred[j]].mean(axis=0))
            for j in range(min(50, len(X_test)))
        ])

        results.append({
            "window": i+1,
            "date": w["start"].strftime("%Y-%m-%d"),
            "avg_confidence": round(max_proba, 4),
            "avg_entropy": round(entropy, 4),
            "centroid_dist": round(centroid_dist, 4),
            "n": len(w_emb["embeddings"])
        })
        print(f"      Window {i+1:02d} ({w['start'].date()}): confidence={max_proba:.3f} entropy={entropy:.3f} n={len(w_emb['embeddings'])}")

    results_df = pd.DataFrame(results)

    # Step 4: Correlate with WKCS
    print("\n[4/4] Correlating with WKCS...")
    drift_df = pd.read_csv("data/drift_scores.csv")
    wkcs = drift_df["wkcs"].values

    # Use window i+1 metrics vs pair i WKCS
    confidences = results_df["avg_confidence"].values[1:len(wkcs)+1]
    entropies = results_df["avg_entropy"].values[1:len(wkcs)+1]
    dists = results_df["centroid_dist"].values[1:len(wkcs)+1]

    corr_conf = np.corrcoef(wkcs, confidences)[0,1]
    corr_entr = np.corrcoef(wkcs, entropies)[0,1]
    corr_dist = np.corrcoef(wkcs, dists)[0,1]

    print(f"\n      WKCS vs classifier confidence : {corr_conf:.4f}")
    print(f"      WKCS vs classifier entropy    : {corr_entr:.4f}")
    print(f"      WKCS vs centroid distance     : {corr_dist:.4f}")

    # Key finding
    print(f"\n      Key findings:")
    print(f"      Window 1  (training, Jan 2017): confidence={results_df.iloc[0]['avg_confidence']:.3f}")
    high_drift = results_df[results_df["date"] >= "2017-10-01"].head(1)
    low_drift  = results_df[results_df["date"] >= "2017-09-13"].head(1)
    if len(high_drift):
        print(f"      Window 15 (Oct 2017, PEAK drift): confidence={high_drift.iloc[0]['avg_confidence']:.3f}, entropy={high_drift.iloc[0]['avg_entropy']:.3f}")
    if len(low_drift):
        print(f"      Window 13 (Sep 2017, LOW drift) : confidence={low_drift.iloc[0]['avg_confidence']:.3f}, entropy={low_drift.iloc[0]['avg_entropy']:.3f}")

    results_df.to_csv("data/downstream_v3.csv", index=False)
    print(f"\nSaved to data/downstream_v3.csv")
    return results_df

if __name__ == "__main__":
    run_topic_validation()

import pandas as pd
import numpy as np

def compute_adaptive_thresholds(drift_csv: str = "data/drift_scores.csv", 
                                 window_size: int = 5, 
                                 z_score: float = 2.0) -> pd.DataFrame:
    print("=" * 60)
    print("Adaptive Drift Threshold Alerting")
    print("=" * 60)

    df = pd.read_csv(drift_csv)
    wkcs = df["wkcs"].values

    thresholds = []
    alerts = []
    baselines = []
    sigmas = []

    for i in range(len(wkcs)):
        # Use preceding N windows as baseline (not current)
        if i < window_size:
            baseline_vals = wkcs[:max(1, i)]
        else:
            baseline_vals = wkcs[i-window_size:i]

        if len(baseline_vals) == 0:
            mu, sigma, threshold = wkcs[0], 0.1, wkcs[0] + 0.2
        else:
            mu = np.mean(baseline_vals)
            sigma = np.std(baseline_vals) + 1e-6
            threshold = mu + z_score * sigma

        is_alert = wkcs[i] > threshold
        thresholds.append(round(threshold, 4))
        alerts.append(is_alert)
        baselines.append(round(mu, 4))
        sigmas.append(round(sigma, 4))

    df["baseline_mean"] = baselines
    df["baseline_std"] = sigmas
    df["adaptive_threshold"] = thresholds
    df["alert"] = alerts
    df["z_score"] = [(wkcs[i] - baselines[i]) / (sigmas[i] + 1e-6) 
                     for i in range(len(wkcs))]
    df["z_score"] = df["z_score"].round(3)

    print(f"\nResults across {len(df)} window pairs:\n")
    print(f"{'Pair':<6} {'Date':<12} {'WKCS':<8} {'Threshold':<12} {'Z-score':<10} {'ALERT'}")
    print("-" * 58)
    for _, row in df.iterrows():
        alert_str = "🚨 ALERT" if row["alert"] else ""
        print(f"{int(row['pair']):<6} {row['window_start'][:10]:<12} {row['wkcs']:<8.4f} "
              f"{row['adaptive_threshold']:<12.4f} {row['z_score']:<10.3f} {alert_str}")

    alert_df = df[df["alert"]]
    print(f"\nTotal alerts: {len(alert_df)} / {len(df)} windows")
    print(f"\nAlert windows:")
    for _, row in alert_df.iterrows():
        print(f"  Pair {int(row['pair'])}: {row['window_start'][:10]} — "
              f"WKCS={row['wkcs']:.4f}, threshold={row['adaptive_threshold']:.4f}, "
              f"z={row['z_score']:.2f}σ")

    df.to_csv("data/drift_scores_with_alerts.csv", index=False)
    print(f"\nSaved to data/drift_scores_with_alerts.csv")
    return df

if __name__ == "__main__":
    compute_adaptive_thresholds()

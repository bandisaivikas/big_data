import os, sys, pickle
import numpy as np
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "false"

df = pd.read_csv("data/weight_sensitivity.csv")

print("="*65)
print("WHY 0.6/0.4 — QUANTITATIVE JUSTIFICATION")
print("="*65)

# 1. Signal-to-noise ratio per weight combo
print("\n1. SIGNAL-TO-NOISE RATIO (Peak / Std)")
print("   Higher = better separation of real drift from noise")
print("-"*55)
for name, group in df.groupby(["alpha","beta","name"]):
    alpha, beta, n = name
    vals = group["wkcs"].values
    snr = max(vals) / np.std(vals)
    marker = "m" if alpha == 0.6 else ""
    print(f"   α={alpha}, β={beta} ({n:<20}): SNR = {snr:.2f}{marker}")

# 2. Alert precision simulation
print("\n2. ALERT PRECISION (z-score threshold = 2.0)")
print("   How many alerts are above 2-sigma = real drift events?")
print("-"*55)
for name, group in df.groupby(["alpha","beta","name"]):
    alpha, beta, n = name
    vals = group["wkcs"].values
    mean_v = np.mean(vals)
    std_v = np.std(vals)
    alerts = sum(1 for v in vals if v > mean_v + 2*std_v)
    marker = " m" if alpha == 0.6 else ""
    print(f"   α={alpha}, β={beta} ({n:<20}): {alerts} alerts{marker}")

# 3. Variance explained
print("\n3. COEFFICIENT OF VARIATION (Std/Mean)")
print("   Lower = more stable metric, fewer false positives")
print("-"*55)
for name, group in df.groupby(["alpha","beta","name"]):
    alpha, beta, n = name
    vals = group["wkcs"].values
    cv = np.std(vals) / np.mean(vals)
    marker = " ◄ SELECTED" if alpha == 0.6 else ""
    print(f"   α={alpha}, β={beta} ({n:<20}): CV = {cv:.4f}{marker}")

print("\n" + "="*65)
print("CONCLUSION")
print("="*65)
print("""
0.6/0.4 is selected
""")

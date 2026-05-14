# data/raw/download.py
from datasets import load_dataset
import pandas as pd

print("Downloading dataset...")
ds = load_dataset("cc_news", split="train[:5000]", trust_remote_code=True)
df = pd.DataFrame(ds)[["title", "description", "date"]]
df = df.dropna()
df["text"] = df["title"] + " " + df["description"]
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])
df = df.sort_values("date")
df.to_parquet("data/raw/corpus.parquet", index=False)
print(f"Saved {len(df)} articles spanning {df['date'].min()} to {df['date'].max()}")
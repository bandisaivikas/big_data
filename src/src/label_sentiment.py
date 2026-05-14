import pandas as pd
from textblob import TextBlob

print("Loading corpus...")
df = pd.read_parquet("data/raw/corpus.parquet")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

print("Labeling sentiment...")
def get_sentiment(text):
    try:
        pol = TextBlob(str(text)).sentiment.polarity
        if pol > 0.05: return 1    # positive
        elif pol < -0.05: return 0  # negative
        else: return None           # neutral — skip
    except:
        return None

df["sentiment"] = df["text"].apply(get_sentiment)
df = df.dropna(subset=["sentiment"])
df["sentiment"] = df["sentiment"].astype(int)

print(f"Labeled {len(df)} articles")
print(f"Positive: {df['sentiment'].sum()}, Negative: {len(df) - df['sentiment'].sum()}")
df.to_parquet("data/raw/corpus_labeled.parquet", index=False)
print("Saved to data/raw/corpus_labeled.parquet")

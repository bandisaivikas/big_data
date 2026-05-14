import pandas as pd
import json
import time
import random
from kafka import KafkaProducer

def stream_random(parquet_path="data/raw/corpus.parquet",
                  topic="news-stream-live",
                  delay=0.05):

    print("=" * 60)
    print("SemDriftBD — Live Random Producer")
    print("Sending articles in random date order")
    print("This fills multiple windows simultaneously")
    print("=" * 60)

    producer = KafkaProducer(
        bootstrap_servers=["localhost:9092"],
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )

    df = pd.read_parquet(parquet_path)
    df["date"] = pd.to_datetime(df["date"])

    # Shuffle so multiple windows fill at same time
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\nStreaming {len(df):,} articles in random order...")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"This will fill multiple 30-day windows simultaneously\n")

    total_sent = 0
    while True:
        for i, row in df.iterrows():
            message = {
                "id": int(i),
                "text": str(row["text"])[:500],
                "date": str(row["date"].date()),
                "timestamp": int(row["date"].timestamp() * 1000),
            }
            producer.send(topic, value=message)
            total_sent += 1

            if total_sent % 200 == 0:
                producer.flush()
                print(f"  Sent {total_sent:,} | latest date: {row['date'].date()}")

            time.sleep(delay)

        print(f"\nFull corpus sent. Restarting...")
        df = df.sample(frac=1).reset_index(drop=True)

if __name__ == "__main__":
    stream_random()

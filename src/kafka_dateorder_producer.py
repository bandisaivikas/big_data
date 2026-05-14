import pandas as pd
import json
import time
from kafka import KafkaProducer

def stream_dateorder(parquet_path="data/raw/corpus.parquet",
                     topic="news-stream-live",
                     delay=0.02):

    print("=" * 60)
    print("SemDriftBD — Date-Order Producer")
    print("Sending articles in chronological order")
    print("Real temporal drift will be detected")
    print("=" * 60)

    producer = KafkaProducer(
        bootstrap_servers=["localhost:9092"],
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )

    df = pd.read_parquet(parquet_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    print(f"\nStreaming {len(df):,} articles in date order...")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}\n")

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

            if total_sent % 500 == 0:
                producer.flush()
                print(f"  Sent {total_sent:,} | date: {row['date'].date()}")

            time.sleep(delay)

        print(f"\nFull corpus sent. Restarting...")

if __name__ == "__main__":
    stream_dateorder()

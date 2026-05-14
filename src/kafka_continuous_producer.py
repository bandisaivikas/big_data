import pandas as pd
import json
import time
from kafka import KafkaProducer

def stream_continuously(parquet_path="data/raw/corpus.parquet",
                        topic="news-stream-live",
                        delay=0.1):

    print("=" * 60)
    print("SemDriftBD — Continuous Kafka Stream Producer")
    print(f"Streaming one article every {delay}s — runs until stopped")
    print("=" * 60)

    producer = KafkaProducer(
        bootstrap_servers=["localhost:9092"],
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )

    df = pd.read_parquet(parquet_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    print(f"\nStreaming {len(df):,} articles continuously...")
    print("Press Ctrl+C to stop\n")

    total_sent = 0
    while True:
        for i, row in df.iterrows():
            message = {
                "id": int(i),
                "text": str(row["text"])[:500],
                "date": str(row["date"].date()),
                "timestamp": int(row["date"].timestamp() * 1000)
            }
            producer.send(topic, value=message)
            total_sent += 1

            if total_sent % 100 == 0:
                producer.flush()
                print(f"  Sent {total_sent:,} articles | current: {row['date'].date()}")

            time.sleep(delay)

        print(f"\nFull corpus streamed. Restarting...")
        time.sleep(1)

if __name__ == "__main__":
    stream_continuously()

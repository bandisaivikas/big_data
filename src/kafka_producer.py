import pandas as pd
import json
import time
from kafka import KafkaProducer

def stream_articles(parquet_path: str = "data/raw/corpus.parquet",
                    topic: str = "news-stream",
                    delay: float = 0.05):

    print("=" * 60)
    print("SemDriftBD — Kafka News Stream Producer")
    print(f"Topic: {topic}")
    print("=" * 60)

    producer = KafkaProducer(
        bootstrap_servers=["localhost:9092"],
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8")
    )

    df = pd.read_parquet(parquet_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    print(f"\nStreaming {len(df)} articles to Kafka topic '{topic}'...")
    print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Delay between messages: {delay}s\n")

    for i, row in df.iterrows():
        message = {
            "id": i,
            "text": str(row["text"])[:500],
            "date": str(row["date"].date()),
            "timestamp": int(row["date"].timestamp())
        }
        producer.send(
            topic,
            key=str(row["date"].date()),
            value=message
        )
        if i % 100 == 0:
            print(f"  Sent {i}/{len(df)} articles — current date: {row['date'].date()}")

    producer.flush()
    print(f"\nDone. {len(df)} articles streamed to Kafka.")
    producer.close()

if __name__ == "__main__":
    stream_articles()

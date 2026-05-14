import os, sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, window, collect_list, count
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType
import time

def run_streaming_consumer():
    print("=" * 60)
    print("SemDriftBD — Spark Structured Streaming Consumer")
    print("Reading live from Kafka topic: news-stream")
    print("=" * 60)

    spark = SparkSession.builder \
        .appName("SemDriftBD-Streaming") \
        .master("local[3]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.streaming.checkpointLocation", "data/checkpoints") \
        .config("spark.jars.packages",
                "org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.0") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    print(f"\nSpark UI: http://localhost:4040")

    # Define schema for incoming messages
    schema = StructType([
        StructField("id", IntegerType()),
        StructField("text", StringType()),
        StructField("date", StringType()),
        StructField("timestamp", LongType())
    ])

    # Read from Kafka as a stream
    print("\n[1/3] Connecting to Kafka stream...")
    kafka_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "news-stream") \
        .option("startingOffsets", "earliest") \
        .option("maxOffsetsPerTrigger", 200) \
        .load()

    # Parse JSON messages
    parsed_df = kafka_df.select(
        from_json(col("value").cast("string"), schema).alias("data"),
        col("timestamp").alias("kafka_timestamp")
    ).select("data.*", "kafka_timestamp")

    print("[2/3] Setting up streaming window aggregation...")

    # Count articles per date window — shows streaming aggregation
    windowed_df = parsed_df \
        .withWatermark("kafka_timestamp", "1 minute") \
        .groupBy(
            window(col("kafka_timestamp").cast("timestamp"), "30 days", "15 days"),
            col("date")
        ) \
        .agg(count("*").alias("article_count"))

    # Write stream to console and Delta Lake
    print("[3/3] Starting stream — articles flowing from Kafka...")
    print("      Each micro-batch = one sliding window processed\n")

    os.makedirs("data/streaming_output", exist_ok=True)

    query = parsed_df.writeStream \
        .format("console") \
        .option("truncate", False) \
        .option("numRows", 5) \
        .trigger(processingTime="5 seconds") \
        .start()

    # Let it run for 30 seconds to show streaming
    print("Stream running for 30 seconds — watch micro-batches below:")
    print("(Open http://localhost:4040 to see streaming jobs)\n")
    query.awaitTermination(30)
    query.stop()

    print("\nStreaming complete.")
    print("This demonstrates real-time Kafka → Spark Structured Streaming pipeline.")
    spark.stop()

if __name__ == "__main__":
    run_streaming_consumer()

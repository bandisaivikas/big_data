import os, sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["SPARK_HOME"] = os.path.dirname(__import__("pyspark").__file__)

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, window, count, to_timestamp
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType

def run():
    print("=" * 60)
    print("SemDriftBD — TRUE Spark Structured Streaming")
    print("Kafka → spark-sql-kafka connector → micro-batches")
    print("Industry standard architecture — Spark 3.5.3")
    print("=" * 60)

    spark = SparkSession.builder \
        .appName("SemDriftBD-StructuredStreaming") \
        .master("local[3]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.streaming.checkpointLocation", "data/checkpoints") \
        .config("spark.jars.packages",
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    print(f"\nSpark version: {spark.version}")
    print(f"Spark UI: http://localhost:4040")
    print(f"Structured Streaming: http://localhost:4040/StreamingQuery\n")

    schema = StructType([
        StructField("id", IntegerType()),
        StructField("text", StringType()),
        StructField("date", StringType()),
        StructField("timestamp", LongType())
    ])

    print("[1/3] Connecting to Kafka via spark-sql-kafka connector...")
    kafka_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "news-stream-live") \
        .option("startingOffsets", "latest") \
        .option("maxOffsetsPerTrigger", 100) \
        .load()

    print("[2/3] Setting up sliding window aggregation...")
    parsed_df = kafka_df.select(
        from_json(col("value").cast("string"), schema).alias("data")
    ).select("data.*")

    windowed = parsed_df \
        .withColumn("event_time",
                    (col("timestamp") / 1000).cast("timestamp")) \
        .withWatermark("event_time", "60 minutes") \
        .groupBy(
            window(col("event_time"), "30 days", "15 days"),
            col("date")
        ) \
        .agg(count("*").alias("article_count"))

    print("[3/3] Stream running — micro-batches every 3 seconds")
    print("      Start kafka_continuous_producer.py in another terminal\n")

    query = windowed.writeStream \
        .outputMode("update") \
        .format("console") \
        .option("truncate", False) \
        .option("numRows", 5) \
        .trigger(processingTime="3 seconds") \
        .start()

    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        query.stop()
        spark.stop()
        print("\nStream stopped.")

if __name__ == "__main__":
    run()

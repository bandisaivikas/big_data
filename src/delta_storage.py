import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
import pandas as pd

def save_to_delta():
    print("Setting up Delta Lake storage...")

    builder = SparkSession.builder \
        .appName("SemDriftBD-Delta") \
        .master("local[3]") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.driver.memory", "4g")

    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # Save drift scores to Delta Lake
    df = pd.read_csv("data/drift_scores_with_alerts.csv")
    spark_df = spark.createDataFrame(df)

    delta_path = "data/delta/drift_scores"
    spark_df.write.format("delta").mode("overwrite").save(delta_path)
    print(f"Saved drift scores to Delta Lake: {delta_path}")

    # Read back and show — proves ACID transactions work
    delta_df = spark.read.format("delta").load(delta_path)
    print(f"\nDelta Lake table — {delta_df.count()} records:")
    delta_df.select("pair", "window_start", "wkcs", "alert").show(5)

    # Show Delta history — proves versioning
    from delta.tables import DeltaTable
    dt = DeltaTable.forPath(spark, delta_path)
    print("\nDelta Lake transaction log:")
    dt.history().select("version", "timestamp", "operation").show()

    # Save scalability results to Delta too
    scale_df = pd.read_csv("data/scalability_results.csv")
    spark.createDataFrame(scale_df) \
        .write.format("delta").mode("overwrite") \
        .save("data/delta/scalability")
    print("Saved scalability results to Delta Lake")

    spark.stop()
    print("\nDelta Lake setup complete.")

if __name__ == "__main__":
    save_to_delta()

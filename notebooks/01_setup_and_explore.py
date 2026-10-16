# Databricks notebook source
# Confirm the data file is accessible
volume_path = "/Volumes/dbw_pred_maint_kkm/default/raw_data/"
files = dbutils.fs.ls(volume_path)
for f in files:
    print(f.name, f.size)

# COMMAND ----------

raw_text = spark.read.text(f"{volume_path}train_FD001.txt")
print(f"Total rows: {raw_text.count()}")
raw_text.show(5, truncate=False)

# COMMAND ----------

from pyspark.sql.functions import split, col

column_names = (
    ["engine_id", "cycle"] +
    [f"op_setting_{i}" for i in range(1, 4)] +
    [f"sensor_{i}" for i in range(1, 22)]
)

parsed = raw_text.select(
    split(col("value"), "\\s+").alias("parts")
).select(
    *[col("parts")[i].cast("double").alias(column_names[i]) for i in range(len(column_names))]
)

parsed = parsed.withColumn("engine_id", col("engine_id").cast("int")) \
               .withColumn("cycle", col("cycle").cast("int"))

print(f"Parsed: {parsed.count()} rows, {len(parsed.columns)} columns")
parsed.show(5)


# COMMAND ----------

num_engines = parsed.select("engine_id").distinct().count()
print(f"Number of engines: {num_engines}")

# COMMAND ----------

from pyspark.sql.functions import min, max, count

parsed.groupBy("engine_id") \
    .agg(
        max("cycle").alias("max_cycle"),
        count("*").alias("num_readings")
    ) \
    .orderBy("engine_id") \
    .show(10)


# COMMAND ----------

sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
parsed.select(sensor_cols).describe().show()
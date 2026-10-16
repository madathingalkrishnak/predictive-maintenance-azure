# Databricks notebook source
from pyspark.sql.functions import *
from pyspark.sql.types import *

# Define the schema — we know the structure from our earlier exploration
columns = ['engine_id', 'cycle'] + \
          [f'op_setting_{i}' for i in range(1, 4)] + \
          [f'sensor_{i}' for i in range(1, 22)]

# Read as text and split by whitespace
raw_df = spark.read.text("/Volumes/dbw_pred_maint_kkm/default/raw_data/train_FD001.txt")

# Split each line into columns and cast to proper types
split_col = split(raw_df.value, r'\s+')

df = raw_df.select([
    split_col.getItem(i).cast('double').alias(columns[i])
    for i in range(len(columns))
])

# Cast engine_id and cycle to integer
df = df.withColumn("engine_id", col("engine_id").cast("integer")) \
       .withColumn("cycle", col("cycle").cast("integer"))

df.printSchema()
df.show(5)
print(f"Total rows: {df.count()}")
print(f"Number of engines: {df.select('engine_id').distinct().count()}")

# COMMAND ----------

# RUL = max_cycle_for_this_engine - current_cycle
# This is the core label for our predictive maintenance model.
#
# Example: Engine 1 runs for 192 cycles.
#   At cycle 1:   RUL = 192 - 1   = 191 (far from failure)
#   At cycle 190:  RUL = 192 - 190 = 2   (about to fail!)
#   At cycle 192:  RUL = 192 - 192 = 0   (failed)

from pyspark.sql.window import Window

# Get max cycle per engine (= the failure point)
w = Window.partitionBy("engine_id")
df = df.withColumn("max_cycle", max("cycle").over(w))
df = df.withColumn("RUL", col("max_cycle") - col("cycle"))
df = df.drop("max_cycle")

# Verify: engine 1 should have RUL going from 191 down to 0
df.filter(col("engine_id") == 1).select("engine_id", "cycle", "RUL") \
  .orderBy("cycle").show(10)

# Also check the last few cycles
df.filter(col("engine_id") == 1).select("engine_id", "cycle", "RUL") \
  .orderBy(col("cycle").desc()).show(5)

# COMMAND ----------

# Some sensors have zero or near-zero variance — they're constant
# and won't help the model distinguish healthy from failing engines.
# Let's find them using standard deviation.

from pyspark.sql.functions import stddev

sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
op_cols = [f'op_setting_{i}' for i in range(1, 4)]

# Compute stddev for each sensor
stddevs = df.select([stddev(c).alias(c) for c in sensor_cols + op_cols]).collect()[0]

low_variance_cols = []
print("Column Variance Check:")
print("-" * 40)
for c in sensor_cols + op_cols:
    sd = stddevs[c]
    status = "DROP" if (sd is None or sd < 0.01) else "KEEP"
    if status == "DROP":
        low_variance_cols.append(c)
    print(f"  {c:15s}  stddev={sd:10.4f}  → {status}" if sd else f"  {c:15s}  stddev=None        → {status}")

print(f"\nDropping {len(low_variance_cols)} low-variance columns: {low_variance_cols}")

# Drop the useless columns
df = df.drop(*low_variance_cols)
print(f"\nRemaining columns: {df.columns}")

# COMMAND ----------

# Raw sensor values are noisy. Rolling averages and standard deviations
# smooth out the noise and capture TRENDS — which is what predicts failure.
#
# Think of it this way: a single high temperature reading might be noise,
# but a steadily INCREASING average temperature over 5 cycles = real degradation.

from pyspark.sql.window import Window

# Window: for each engine, look at the last 5 cycles
w5 = Window.partitionBy("engine_id").orderBy("cycle").rowsBetween(-4, 0)

# Get the sensor columns that survived the variance filter
remaining_sensors = [c for c in df.columns if c.startswith("sensor_")]

# Create rolling mean and rolling stddev for each sensor
for sensor in remaining_sensors:
    df = df.withColumn(f"{sensor}_rolling_mean_5", avg(col(sensor)).over(w5))
    df = df.withColumn(f"{sensor}_rolling_std_5", stddev(col(sensor)).over(w5))

# Fill nulls from the first few rows (where window isn't full yet)
df = df.fillna(0)

# Show what we've built
print(f"Total features: {len(df.columns)}")
print(f"Columns: {df.columns}")
df.filter(col("engine_id") == 1).select(
    "cycle", "RUL", "sensor_2", "sensor_2_rolling_mean_5", "sensor_2_rolling_std_5"
).show(10)

# COMMAND ----------

# In practice, predicting RUL=191 vs RUL=185 doesn't matter — the engine
# is healthy either way. What matters is predicting the LAST ~125 cycles
# when degradation actually shows up in the sensors.
#
# Capping RUL at 125 is a standard technique in the literature.
# It turns this from a pure regression into a "piecewise" problem:
#   - RUL > 125: engine is healthy (RUL capped at 125)
#   - RUL < 125: engine is degrading (actual RUL value)

RUL_CAP = 125
df = df.withColumn("RUL", when(col("RUL") > RUL_CAP, RUL_CAP).otherwise(col("RUL")))

# Verify the cap
print("RUL distribution after capping:")
df.select("RUL").describe().show()

# Save to a volume as parquet for model training
output_path = "/Volumes/dbw_pred_maint_kkm/default/raw_data/features_train.parquet"
df.write.mode("overwrite").parquet(output_path)

print(f"Saved {df.count()} rows to {output_path}")
print(f"Features: {len(df.columns)} columns")

# Quick preview of final dataset
df.show(5)

# COMMAND ----------


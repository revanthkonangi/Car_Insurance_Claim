# Databricks notebook source
# MAGIC %pip install sparkmeasure

# COMMAND ----------

from sparkmeasure import StageMetrics
stagemetrics = StageMetrics(spark)
stagemetrics.begin()

# COMMAND ----------

try :
    env = dbutils.widgets.get("env")
except :
    env = "dev"
print(f"Input environment : {env}")

# COMMAND ----------

# MAGIC %md <b> User Inputs

# COMMAND ----------

# DBTITLE 1,Load the YAML config
import yaml
with open('../data_config/SolutionConfig.yaml', 'r') as solution_config:
    solution_config = yaml.safe_load(solution_config)

# COMMAND ----------

from MLCORE_SDK import mlclient
from pyspark.sql import functions as F
# GENERAL PARAMETERS
sdk_session_id = solution_config[f'sdk_session_id_{env}']
env = solution_config['ds_environment']
db_name = solution_config['database_name']

# DE SPECIFIC PARAMETERS
primary_keys = solution_config["data_engineering"]["data_engineering_ft"]["primary_keys"]
ground_truth_table_name = solution_config["data_engineering"]["data_engineering_gt"]["ground_truth_table_name"]
ground_truth_dbfs_path = solution_config["data_engineering"]["data_engineering_gt"]["ground_truth_dbfs_path"]

# COMMAND ----------

from MLCORE_SDK import mlclient
mlclient.log(operation_type="job_run_add", session_id = sdk_session_id, dbutils = dbutils, request_type = "DE")

# COMMAND ----------

pickle_file_path = f"/mnt/FileStore/{db_name}"
dbutils.fs.mkdirs(pickle_file_path)
print(f"Created directory : {pickle_file_path}")
pickle_file_path = f"/dbfs/{pickle_file_path}/{ground_truth_table_name}.pickle"

# COMMAND ----------

try : 
  with open(pickle_file_path, "rb") as handle:
      obj_properties = pickle.load(handle)
      print(f"Instance loaded successfully")
except Exception as e:
  print(f"Exception while loading cache : {e}")
  obj_properties = {}
print(f"Existing Cache : {obj_properties}")

# COMMAND ----------

ground_truth_df = spark.read.csv(ground_truth_dbfs_path)

# COMMAND ----------

# ground_truth_df = ground_truth_df.drop('id','date','timestamp')

# COMMAND ----------

from datetime import datetime
from pyspark.sql import (
    types as DT,
    functions as F,
    Window
)
def to_date_(col):
    """
    Checks col row-wise and returns first date format which returns non-null output for the respective column value
    """
    formats=(
             "MM-dd-yyyy", "dd-MM-yyyy",
             "MM/dd/yyyy", "yyyy-MM-dd", 
             "M/d/yyyy", "M/dd/yyyy",
             "MM/dd/yy", "MM.dd.yyyy",
             "dd.MM.yyyy", "yyyy-MM-dd",
             "yyyy-dd-MM"
            )
    return F.coalesce(*[F.to_date(col, f) for f in formats])

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

now = datetime.now()
date = now.strftime("%m-%d-%Y")
ground_truth_df = ground_truth_df.withColumn(
    "timestamp",
    F.expr("reflect('java.lang.System', 'currentTimeMillis')").cast("long"),
)
ground_truth_df = ground_truth_df.withColumn("date", F.lit(date))
ground_truth_df = ground_truth_df.withColumn("date", to_date_(F.col("date")))

# ADD A MONOTONICALLY INREASING COLUMN
if "id" not in ground_truth_df.columns : 
  window = Window.orderBy(F.monotonically_increasing_id())
  ground_truth_df = ground_truth_df.withColumn("id", F.row_number().over(window))

# COMMAND ----------

# DBTITLE 1,Creating hive database if it doesn't exist
spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}")
print(f"HIVE METASTORE DATABASE NAME : {db_name}")

# COMMAND ----------

# DBTITLE 1,ADD A MONOTONICALLY INREASING COLUMN - "id"
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ADD A MONOTONICALLY INREASING COLUMN
if "id" not in ground_truth_df.columns : 
  window = Window.orderBy(F.monotonically_increasing_id())
  ground_truth_df = ground_truth_df.withColumn("id", F.row_number().over(window))

# COMMAND ----------

# DBTITLE 1,Create/Update table in hive database
ground_truth_df.createOrReplaceTempView(ground_truth_table_name)

gt_table_exist = [True for table_data in spark.catalog.listTables(db_name) if table_data.name.lower() == ground_truth_table_name.lower() and not table_data.isTemporary]

if not any(gt_table_exist):
  print(f"CREATING GROUND TRUTH TABLE")
  spark.sql(f"CREATE TABLE IF NOT EXISTS hive_metastore.{db_name}.{ground_truth_table_name} AS SELECT * FROM {ground_truth_table_name}")
else :
  print(F"UPDATING GROUND TRUTH TABLE")
  spark.sql(f"INSERT INTO hive_metastore.{db_name}.{ground_truth_table_name} SELECT * FROM {ground_truth_table_name}")

# COMMAND ----------

# DBTITLE 1,Extracting dbfs path of hive table
from pyspark.sql import functions as F
gt_hive_table_path = spark.sql(f"desc formatted hive_metastore.{db_name}.{ground_truth_table_name}").filter(F.col("col_name") == "Location").select("data_type").collect()[0][0]
print(f"Ground Truth Hive Path : {gt_hive_table_path}")

# COMMAND ----------

stagemetrics.end()

# COMMAND ----------

stagemetrics.print_report()

# COMMAND ----------

compute_metrics = stagemetrics.aggregate_stagemetrics_DF().select("executorCpuTime", "peakExecutionMemory","memoryBytesSpilled","diskBytesSpilled").collect()[0].asDict()

# COMMAND ----------

compute_metrics['executorCpuTime'] = compute_metrics['executorCpuTime']/1000
compute_metrics['peakExecutionMemory'] = float(compute_metrics['peakExecutionMemory']) /(1024*1024)

# COMMAND ----------

# MAGIC %md <b> Use MLCore SDK to register Features and Ground Truth Tables

# COMMAND ----------

mlclient.log(operation_type = "register_table",
    sdk_session_id = sdk_session_id,
    dbutils = dbutils,
    spark = spark,
    table_name = ground_truth_table_name,
    num_rows = ground_truth_df.count(),
    cols = ground_truth_df.columns,
    column_datatype = ground_truth_df.dtypes,
    table_schema = ground_truth_df.schema,
    primary_keys = primary_keys,
    table_path = gt_hive_table_path,
    table_sub_type="Ground_Truth",
    env = "dev",
    compute_usage_metrics = compute_metrics)

# COMMAND ----------



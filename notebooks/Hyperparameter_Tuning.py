# Databricks notebook source
# MAGIC %md ## Hyperparameter_Tuning
# MAGIC

# COMMAND ----------

from sklearn.model_selection import train_test_split
from hyperopt import tpe, fmin, STATUS_OK, Trials, SparkTrials, space_eval
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, Trials
from hyperopt.spark import SparkTrials
import numpy as np
import json, time

# Disable auto-logging of runs in the mlflow
import mlflow
mlflow.autolog(disable=True)

# COMMAND ----------

import yaml
with open('../data_config/SolutionConfig.yaml', 'r') as solution_config:
    solution_config = yaml.safe_load(solution_config)

# COMMAND ----------

# GENERAL PARAMETERS
env = solution_config['ds_environment']
sdk_session_id = solution_config[f'sdk_session_id_{env}']
db_name = solution_config['database_name']

# JOB SPECIFIC PARAMETERS
feature_table_path = solution_config['train']["feature_table_path"]
ground_truth_path = solution_config['train']["ground_truth_path"]
primary_keys = solution_config['train']["primary_keys"]
feature_columns = solution_config['train']["feature_columns"]
target_columns = solution_config['train']["target_columns"]
test_size = solution_config['train']["test_size"]
primary_metric = solution_config['train']['hyperparameter_tuning']["primary_metric"]
search_range = solution_config['train']["hyperparameter_tuning"]["search_range"]
max_evaluations = solution_config['train']["hyperparameter_tuning"]["max_evaluations"]
stop_early = solution_config['train']["hyperparameter_tuning"]["stop_early"]
run_parallel = solution_config['train']["hyperparameter_tuning"]["run_parallel"]

# COMMAND ----------

ft_data = spark.sql(f"SELECT * FROM {db_name}.{feature_table_path}")
gt_data = spark.sql(f"SELECT * FROM {db_name}.{ground_truth_path}")

# COMMAND ----------

final_df = ft_data.join(gt_data.select(primary_keys+target_columns), on = primary_keys)

# COMMAND ----------

final_df_pandas = final_df.toPandas()

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(final_df_pandas[feature_columns], final_df_pandas[target_columns], test_size=test_size, random_state = 0)

# COMMAND ----------

X_train = X_train.fillna(X_train.mean())
X_train_np = X_train.to_numpy()
X_test_np = X_test.to_numpy()

# COMMAND ----------

def early_stop_function(iteration_stop_count=20, percent_increase=0.0):
        def stop_fn(trials, best_loss=None, iteration_no_progress=0):
            if (
                not trials
                or "loss" not in trials.trials[len(trials.trials) - 1]["result"]
            ):
                return False, [best_loss, iteration_no_progress + 1]
            new_loss = trials.trials[len(trials.trials) - 1]["result"]["loss"]
            if best_loss is None:
                return False, [new_loss, iteration_no_progress + 1]
            best_loss_threshold = best_loss - abs(
                best_loss * (percent_increase / 100.0)
            )
            if new_loss < best_loss_threshold:
                best_loss = new_loss
                iteration_no_progress = 0
            else:
                iteration_no_progress += 1
                print(
                    "No progress made: %d iteration on %d. best_loss=%.2f, best_loss_threshold=%.2f, new_loss=%.2f"
                    % (
                        iteration_no_progress,
                        iteration_stop_count,
                        best_loss,
                        best_loss_threshold,
                        new_loss,
                    )
                )

            return (
                iteration_no_progress >= iteration_stop_count,
                [best_loss, iteration_no_progress],
            )

        return stop_fn

def get_trial_data(trials, search_space):
    if not trials:
        return []

    trial_data = []
    trial_id = 0

    for trial in trials.trials:
        trial_id += 1
        trial["result"]["trial"] = trial_id
        trial["result"]["loss"] = (
            0
            if not np.isnan(trial["result"]["loss"])
            and abs(trial["result"]["loss"]) == np.inf
            else trial["result"]["loss"]
        )

        hp_vals = {}
        for hp, hp_val in trial["misc"]["vals"].items():
            hp_vals[hp] = hp_val[0]

        trial["result"]["hyper_parameters"] = space_eval(
            search_space, hp_vals
        )
        trial_data.append(trial["result"])
    return trial_data

def objective(params):
    start_time = time.time()
    metrics = {}
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)

    is_multiclass = len(np.unique(y_test)) > 2
    metrics_average = "weighted" if is_multiclass else "binary"

    f1 = f1_score(y_true=y_test, y_pred=y_test_pred, average=metrics_average)
    metrics["f1"] = f1

    precision = precision_score(
        y_true=y_test, y_pred=y_test_pred, average=metrics_average
    )
    metrics["precision"] = precision

    accuracy = accuracy_score(y_true=y_test, y_pred=y_test_pred)
    metrics["accuracy"] = accuracy

    recall = recall_score(
        y_true=y_test, y_pred=y_test_pred, average=metrics_average
    )
    metrics["recall"] = recall

    loss = - metrics[primary_metric]
    end_time = time.time()
    
    trail_out_put = {
        "loss": abs(loss),
        "metrics": metrics,
        "status": STATUS_OK,
        "duration" : end_time - start_time,
        "primary_metric":primary_metric,
        "max_evaluations":max_evaluations,
        "early_stopping":stop_early}

    return trail_out_put

def hyperparameter_tuning_with_trials(search_space,max_evals,run_parallel,stop_early):
    if run_parallel:
        trials = SparkTrials(parallelism=4)
    else:
        trials = Trials()

    best_config = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals= max_evals,
            trials=trials,
            early_stop_fn= early_stop_function(10, -0.01)
            if stop_early
            else None,
        )

    return best_config, trials


hyperopt_mapping = {
    bool: hp.choice,
    int: hp.uniform,
    float: hp.uniform,
    str: hp.choice
}

# Converted search space
search_space = {}

for key, value in search_range.items():
    value_type = type(value[0])
    if value_type in hyperopt_mapping:
        if value_type in [bool, str]:
            search_space[key] = hyperopt_mapping[value_type](key, value)
        else:
            search_space[key] = hyperopt_mapping[value_type](key, value[0], value[1])
    else:
        raise ValueError(f"Unsupported type for {key}")

# COMMAND ----------

best_hyperparameters , tuning_trails = hyperparameter_tuning_with_trials( search_space= search_space, max_evals=max_evaluations, run_parallel=run_parallel,stop_early=stop_early)

best_hyperparameters = space_eval(search_space, best_hyperparameters)
tuning_trails_all = get_trial_data(tuning_trails, search_space)


# COMMAND ----------

tuning_trails_all

# COMMAND ----------

hp_tuning_result = {
    "best_hyperparameters":best_hyperparameters,
    "tuning_trails":tuning_trails_all,
}

# COMMAND ----------

dbutils.notebook.exit(json.dumps(hp_tuning_result))

# COMMAND ----------



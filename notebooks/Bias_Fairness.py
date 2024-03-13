# Databricks notebook source
# MAGIC %pip install aequitas==0.42.0
# MAGIC %pip install google-auth
# MAGIC %pip install google-cloud-storage
# MAGIC %pip install azure-storage-blob
# MAGIC %pip install protobuf==3.17.2

# COMMAND ----------

def get_env_vault_scope():
    """
    Returns env and vault scope
    """
    import json
    env = (
        dbutils.notebook.entry_point.getDbutils()
        .notebook()
        .getContext()
        .notebookPath()
        .get()
    ).split("/")[2]

    if env == 'prod':
        vault_scope = 'qecdspdataprodqvc'
    elif env == 'qa':
        vault_scope = 'qecdspdataqaqvc'
    else:
        vault_scope = 'qecdspdatadevqvc'
    try:
        if len(dbutils.fs.ls('dbfs:/FileStore/jars/MLCORE_INIT/vault_check.json')) == 1:
            if env == "qa":
                with open("/dbfs/FileStore/jars/MLCORE_INIT/vault_check_qa.json", "r") as file:
                    vault_check_data = json.loads(file.read())
            else:
                with open("/dbfs/FileStore/jars/MLCORE_INIT/vault_check.json", "r") as file:
                    vault_check_data = json.loads(file.read())
            if "@" in env:
                return "qa", vault_check_data['client_name']
            return env, vault_check_data['client_name']
        else:
            return env, vault_scope
    except:
        return env, vault_scope

def get_access_tokens(client_id, scope, client_secret, vault_scope):
    """
    Returns a bearer token
    """

    headers = CaseInsensitiveDict()
    headers["Content-Type"] = "application/x-www-form-urlencoded"
    data = {}
    data["client_id"] = client_id
    data["grant_type"] = "client_credentials"
    data["scope"] = scope
    data["client_secret"] = client_secret
    tenant_id = dbutils.secrets.get(scope=vault_scope, key="az-directory-tenant")
    url = "https://login.microsoftonline.com/" + tenant_id + "/oauth2/v2.0/token"
    resp = requests.post(url, headers=headers, data=data).json()
    token = resp["access_token"]
    token_string = "Bearer" + " " + token
    return token_string

def get_app_url(env, vault_scope):
    """
    Returns DEV or QA base url endpoint
    """
    API_ENDPOINT = (
        "https://mlcoredevapi2.azurewebsites.net"
        + "/"
    )
    return API_ENDPOINT

def get_headers(vault_scope):
    """
    Returns API headers
    """
    h1 = CaseInsensitiveDict()
    client_id = dbutils.secrets.get(scope=vault_scope,key="az-api-client-id")
    scope = client_id + "/.default"
    client_secret = dbutils.secrets.get(scope=vault_scope,key="az-api-client-secret")
    h1["Authorization"]= get_access_tokens(client_id, scope, client_secret, vault_scope)
    h1["Content-Type"] = "application/json"
    return h1

def json_str_to_pythontype(param):
    param = param.replace("'", '"')
    param = json.loads(param)

    return param

def generate_run_notebook_url(job_id, run_id):
    """
    Generates the databricks job run notebook url in runtime
    """
    workspace_url = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
    workspace_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterOwnerOrgId")
    run_notebook_url = f"{workspace_url}/?o={workspace_id}#job/{job_id}/run/{run_id}"
    return run_notebook_url

def fetch_secret_from_dbutils(dbutils, key_name):
    _, vault_scope = get_env_vault_scope()
    return dbutils.secrets.get(scope=vault_scope, key=key_name)
    
def get_gcp_auth_credentials(dbutils):
    import google.auth

    _, vault_scope = get_env_vault_scope()
    client_id = dbutils.secrets.get(scope=vault_scope, key="gcp-api-client-id")
    client_secret = dbutils.secrets.get(scope=vault_scope, key="gcp-api-client-secret")
    quota_project_id = dbutils.secrets.get(
        scope=vault_scope, key="gcp-api-quota-project-id"
    )
    refresh_token = dbutils.secrets.get(scope=vault_scope, key="gcp-api-refresh-token")
    cred_dict = {
        "client_id": client_id,
        "client_secret": client_secret,
        "quota_project_id": quota_project_id,
        "refresh_token": refresh_token,
        "type": "authorized_user",
    }

    credentials, _ = google.auth.load_credentials_from_dict(cred_dict)

    return credentials

def __upload_blob_to_azure(dbutils, container_name, blob_path, target_path):
    from azure.storage.blob import BlobServiceClient

    try:
        connection_string = fetch_secret_from_dbutils(
            dbutils, "az-api-storage-connection-string"
        )
        # Initialize the BlobServiceClient using the connection string
        blob_service_client = BlobServiceClient.from_connection_string(
            connection_string
        )

        # Get the container client for the blob object
        container_client = blob_service_client.get_container_client(
            container=container_name
        )

        # Upload the blob file
        with open(blob_path, "rb") as data:
            blob_client = container_client.get_blob_client(target_path)
            blob_client.upload_blob(data)

    except Exception as e:
        print(f"Error came while uploading blob object from azure : {e}")
        raise e


def __upload_blob_to_gcp(dbutils, container_name, blob_path, target_path):
    from google.cloud import storage

    try:
        credentials = get_gcp_auth_credentials(dbutils)
        project = fetch_secret_from_dbutils(dbutils, "gcp-api-quota-project-id")

        # Use the obtained credentials to create a client to interact with GCP services
        storage_client = storage.Client(credentials=credentials, project=project)

        bucket_client = storage_client.bucket(container_name)

        # Upload the model file to GCS
        blob = bucket_client.blob(target_path)
        blob.upload_from_filename(blob_path)

    except Exception as e:
        print(f"Error came while uploading blob object from gcp : {e}")
        raise e


def upload_blob_to_cloud(**kwargs):
    """
    Upload the blob from the cloud storage.

    This function will help upload the blob from the cloud storage service like Azure, AWS, GCP.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments containing operation details, including the `resource_type`.

    Returns
    -------
    The result of the dispatched operation based on the `resource_type`.

    Notes
    -----
    - The function loads the blob object from cloud storage with below parameters :
    - For Azure
        - 'dbutils': The dbutils object to retrive the secrets needed for the APIs.
        - 'container_name': The container where the blob object is stored.
        - 'blob_path': The local file path where the blob is present.
        - 'target_path' : The target path where the blob has to be downloaded.

    - For GCP
        - 'dbutils': The dbutils object to retrive the secrets needed for the APIs.
        - 'container_namecontainer_name': The bucket where the blob  object is stored.
        - 'blob_path': The local file path where the blob is present.
        - 'target_path' : The target path where the blob has to be downloaded.

    - It is essential to provide the correct `resource_type`. Currently supported resources are : az, gcp
    """
    resource_type = kwargs.get("resource_type", None)
    if not resource_type or resource_type in [""]:
        raise Exception("Resource type is not passed or is empty.")

    del kwargs["resource_type"]  # Delete the key since it will not be used by modules

    if resource_type not in ["az", "gcp","azure"]:
        raise Exception(f"Uploading blob object from {resource_type} is not supported.")

    if resource_type.lower() in ["az","azure"]:
        return __upload_blob_to_azure(**kwargs)

    if resource_type.lower() == "gcp":
        return __upload_blob_to_gcp(**kwargs)
def get_gcp_auth_credentials(dbutils):
    import google.auth

    _, vault_scope = get_env_vault_scope()
    client_id = dbutils.secrets.get(scope=vault_scope, key="gcp-api-client-id")
    client_secret = dbutils.secrets.get(scope=vault_scope, key="gcp-api-client-secret")
    quota_project_id = dbutils.secrets.get(
        scope=vault_scope, key="gcp-api-quota-project-id"
    )
    refresh_token = dbutils.secrets.get(scope=vault_scope, key="gcp-api-refresh-token")
    cred_dict = {
        "client_id": client_id,
        "client_secret": client_secret,
        "quota_project_id": quota_project_id,
        "refresh_token": refresh_token,
        "type": "authorized_user",
    }

    credentials, _ = google.auth.load_credentials_from_dict(cred_dict)

    return credentials

def __upload_blob_to_azure(dbutils, container_name, blob_path, target_path):
    from azure.storage.blob import BlobServiceClient

    try:
        connection_string = fetch_secret_from_dbutils(
            dbutils, "az-api-storage-connection-string"
        )
        # Initialize the BlobServiceClient using the connection string
        blob_service_client = BlobServiceClient.from_connection_string(
            connection_string
        )

        # Get the container client for the blob object
        container_client = blob_service_client.get_container_client(
            container=container_name
        )

        # Upload the blob file
        with open(blob_path, "rb") as data:
            blob_client = container_client.get_blob_client(target_path)
            blob_client.upload_blob(data)

    except Exception as e:
        print(f"Error came while uploading blob object from azure : {e}")
        raise e


def __upload_blob_to_gcp(dbutils, container_name, blob_path, target_path):
    from google.cloud import storage

    try:
        credentials = get_gcp_auth_credentials(dbutils)
        project = fetch_secret_from_dbutils(dbutils, "gcp-api-quota-project-id")

        # Use the obtained credentials to create a client to interact with GCP services
        storage_client = storage.Client(credentials=credentials, project=project)

        bucket_client = storage_client.bucket(container_name)

        # Upload the model file to GCS
        blob = bucket_client.blob(target_path)
        blob.upload_from_filename(blob_path)

    except Exception as e:
        print(f"Error came while uploading blob object from gcp : {e}")
        raise e


def upload_blob_to_cloud(**kwargs):
    """
    Upload the blob from the cloud storage.

    This function will help upload the blob from the cloud storage service like Azure, AWS, GCP.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments containing operation details, including the `resource_type`.

    Returns
    -------
    The result of the dispatched operation based on the `resource_type`.

    Notes
    -----
    - The function loads the blob object from cloud storage with below parameters :
    - For Azure
        - 'dbutils': The dbutils object to retrive the secrets needed for the APIs.
        - 'container_name': The container where the blob object is stored.
        - 'blob_path': The local file path where the blob is present.
        - 'target_path' : The target path where the blob has to be downloaded.

    - For GCP
        - 'dbutils': The dbutils object to retrive the secrets needed for the APIs.
        - 'container_namecontainer_name': The bucket where the blob  object is stored.
        - 'blob_path': The local file path where the blob is present.
        - 'target_path' : The target path where the blob has to be downloaded.

    - It is essential to provide the correct `resource_type`. Currently supported resources are : az, gcp
    """
    resource_type = kwargs.get("resource_type", None)
    if not resource_type or resource_type in [""]:
        raise Exception("Resource type is not passed or is empty.")

    del kwargs["resource_type"]  # Delete the key since it will not be used by modules

    if resource_type not in ["az", "gcp","azure"]:
        raise Exception(f"Uploading blob object from {resource_type} is not supported.")

    if resource_type.lower() in ["az","azure"]:
        return __upload_blob_to_azure(**kwargs)

    if resource_type.lower() == "gcp":
        return __upload_blob_to_gcp(**kwargs)

# COMMAND ----------

from aequitas.preprocessing import preprocess_input_df
from tigerml.core.reports import create_report
from utils import utils

# COMMAND ----------

# train_output_path = "dbfs:/mnt/mlcore/dev/7820411c70ef4c1a94131e22266b07f3/Superstore/66002368769639/6570183776277bd1aa830866/report_output"
# features= ['Education', 'Marital_Status', 'Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'Complain', 'Age', 'Reg_year', 'Reg_month', 'Reg_week']
# target= ['Response']
# reserved_columns = ['prediction','dataset_type_71E4E76EB8C12230B6F51EA2214BD5FE', 'probability']

# sensitive_columns = ['Marital_Status','Education']
# report_directory = dbutils.widgets.get("report_directory")

# COMMAND ----------

train_output_path = dbutils.widgets.get("model_data_path")
features = dbutils.widgets.get("feature_columns").split(",")
target = dbutils.widgets.get("target_columns")
reserved_columns = ['prediction','dataset_type_71E4E76EB8C12230B6F51EA2214BD5FE', 'probability']
sensitive_columns = ['Marital_Status','Education']
# report_directory = dbutils.widgets.get("report_directory")
media_artifacts_path = dbutils.widgets.get("media_artifacts_path")

# COMMAND ----------

bias_check_df = spark.read.load(train_output_path).select(features + target + reserved_columns)
total_records = bias_check_df.count()

# COMMAND ----------

df = bias_check_df.toPandas()[sensitive_columns + target + ['prediction']]

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### AEQUITAS : Preprocessing

# COMMAND ----------

df = df.rename(columns={'prediction' : 'score', target[0] : 'label_value'})

# COMMAND ----------

from aequitas.preprocessing import preprocess_input_df

# double-check that categorical columns are of type 'string'
for col in sensitive_columns:
    df[col] = df[col].astype(str)

df, _ = preprocess_input_df(df)

# COMMAND ----------

from aequitas.group import Group
from aequitas.plotting import Plot
from aequitas.bias import Bias

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### AEQUITAS GROUPS : What biases exist in my model?

# COMMAND ----------

g = Group()
xtab, _ = g.get_crosstabs(df)

# COMMAND ----------

aqp = Plot()

# COMMAND ----------

import matplotlib.pyplot as plt

metrics = ['fnr', 'fpr', 'for', 'fdr', 'ppr', 'pprev']

num_rows = (len(metrics) + 1) // 2 

# Create a figure to contain all plots
fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(12, num_rows*3.5))  

# Flatten the axes array
axes = axes.flatten()

# Loop over each fnr and plot
for i, metric in enumerate(metrics):
    # Plot the group metric for the current fnr
    aqp.plot_group_metric(xtab, metric, ax=axes[i])
    axes[i].set_title(f'{metric.upper()}', fontsize=14)  # Set the font size of the title

# Adjust layout to prevent overlapping
plt.tight_layout()

plt.show()

# COMMAND ----------

# a = aqp.plot_group_metric_all(xtab, ncols=3)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### AEQUITAS GROUPS : What levels of disparity exist between population groups?

# COMMAND ----------

b = Bias()

# COMMAND ----------

majority_bdf = b.get_disparity_major_group(xtab, original_df=df, mask_significance=True)

# COMMAND ----------

majority_bdf

# COMMAND ----------

import matplotlib.pyplot as plt

metrics = ['tpr_disparity', 'tnr_disparity', 'for_disparity', 'fdr_disparity', 'fpr_disparity', 'fnr_disparity', 'npv_disparity', 'precision_disparity', 'ppr_disparity', 'pprev_disparity']

fig1 = {}
for metric in metrics:
    # Plot the disparity for the current metric
    fig1[metric] = aqp.plot_disparity_all(majority_bdf, metrics=[metric], significance_alpha=0.05)

# COMMAND ----------

min_metric_bdf = b.get_disparity_min_metric(df=xtab, original_df=df)

# COMMAND ----------

import matplotlib.pyplot as plt

metrics1 = ['for_disparity', 'fdr_disparity', 'fpr_disparity', 'fnr_disparity', 'ppr_disparity', 'pprev_disparity']

fig2 = {}
for metric1 in metrics1:
    # Plot the disparity for the current metric
    fig2[metric1] = aqp.plot_disparity_all(min_metric_bdf, metrics=[metric1],attributes=['Education', 'Marital_Status'], significance_alpha=0.05)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### AEQUITAS Fairness : How to assess model fairness?

# COMMAND ----------

from aequitas.fairness import Fairness

# COMMAND ----------

f = Fairness()
fdf = f.get_group_value_fairness(majority_bdf)

# COMMAND ----------

fg = aqp.plot_fairness_group_all(fdf, ncols=5, metrics = "all")

# COMMAND ----------

fdf.display()

# COMMAND ----------

fg.tight_layout()
axes = fg.get_axes()

# Set the extent of each axis to trim excess whitespace
for ax in axes:
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())

# Manually adjust the size of the figure
left, bottom, right, top = fg.bbox.bounds
fg.set_size_inches(right / 100, top / 100)

# COMMAND ----------

import matplotlib.pyplot as plt

metrics2 = ['tpr_disparity', 'tnr_disparity', 'for_disparity', 'fdr_disparity', 'fpr_disparity', 'fnr_disparity', 'npv_disparity', 'precision_disparity', 'ppr_disparity', 'pprev_disparity']

fig3 = {}
for metric2 in metrics2:
    # Plot the disparity for the current metric
    fig3[metric2] = aqp.plot_fairness_disparity_all(fdf, metrics=[metric2],attributes=['Education', 'Marital_Status'], significance_alpha=0.05)

# COMMAND ----------

import pandas as pd

# COMMAND ----------


data = {
    'Metric': ['True Positive Rate', 'True Negative Rate', 'False Omission Rate', 
               'False Discovery Rate', 'False Positive Rate', 'False Negative Rate', 
               'Negative Predictive Value', 'Precision', 'Predicted Positive Ratio_k', 
               'Predicted Positive Ratio_g', 'Group Prevalence'],
    'Column Name': ['tpr', 'tnr', 'for', 'fdr', 'fpr', 'fnr', 'npv', 'precision', 'ppr', 'pprev', 'prev']
}

Table_1 = pd.DataFrame(data)

# COMMAND ----------

Table_1.display()

# COMMAND ----------

Data = {
    'Metric': ['True Positive Rate Disparity', 'True Negative Rate Disparity', 
               'False Omission Rate Disparity', 'False Discovery Rate Disparity', 
               'False Positive Rate Disparity', 'False NegativeRate Disparity', 
               'Negative Predictive Value Disparity', 'Precision Disparity', 
               'Predicted Positive Ratio_k Disparity', 'Predicted Positive Ratio_g Disparity'],
    'Column Name': ['tpr_disparity', 'tnr_disparity', 'for_disparity', 'fdr_disparity', 
                    'fpr_disparity', 'fnr_disparity', 'npv_disparity', 'precision_disparity', 
                    'ppr_disparity', 'pprev_disparity']
}

Table_2 = pd.DataFrame(Data)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC 1. Prepare a case study : outcome of each chart has to be listed.
# MAGIC 2. Convert it into html

# COMMAND ----------

plot_dict1 = {}
for metric in metrics:
    plot_dict1[metric.upper()] = fig1[metric]
plot_dict2 = {}
for metric1 in metrics1:
    plot_dict2[metric1.upper()] = fig2[metric1]
plot_dict3 = {}
for metric2 in metrics2:
    plot_dict3[metric2.upper()] = fig3[metric2]


# COMMAND ----------

report = {
    "Category Level Bias": {
        "Individual Category Level Bias": fig,
        "Category Level Bias Table": (xtab,"<p style='text-align:left;'><strong>Summary:</strong> The chart above displays various metrics calculated across each attribute and colored based on the number of samples in the attribute group. Longer bars across attributes represent the highest value of the metric. Dark coloring across the attributes represents the largest populations in the data set.The table helps to deep dive into the data calculated and to see the correctness of the plots</p>"),
    },
    "Group Level Bias": {
        "Majority Group Level Bias": plot_dict1,
        "Majority Group Level Bias Table": (majority_bdf,"<p style='text-align:left;'><strong>Summary:</strong> The treemaps above display all the disparities (calculated in relation to the sample majority group for each attribute) between groups across attributes, colored based on disparity magnitude. The table helps to deep dive into the data calculated and to see the correctness of the plots.</p>"),
        "Minority Group Level Bias": plot_dict2,
        "Minority Group Level Bias Table": (min_metric_bdf,"<p style='text-align:left;'><strong>Summary:</strong>The treemaps above display all the disparities (calculated in relation to the sample minority group for each attribute) between groups across attributes, colored based on disparity magnitude. The table helps to deep dive into the data calculated and to see the correctness of the plots>"),
    },
    "Group Level Model Fairness": {
        "Fairness ": fg,
        "Fairness Table": (fdf, "<p style='text-align:left;'><strong>Summary:</strong> The chart above displays all calculated absolute group metrics across each attribute, colored based on fairness determination for that attribute group (green = ‘True’ and red = ‘False’). Immediately we can see that negative false rate, true negative parity, and negative predictive parity status are ‘True’ for all population groups.The table helps to deep dive into the data calculated and to see the correctness of the plots</p>"),
        "Group Level Fairness Disparity": plot_dict3,
    },
    "Appendix": {
        "Absolute Metrics Calculated": Table_1,
        "Disparities Calculated": Table_2,
    }
}

# report["Group Level Bias"]["Majority Group Level Bias"] = (plot_dict1,"<p style='text-align:left;'><strong>Summary:</strong> The treemaps above display all the disparities (calculated in relation to the sample majority group for each attribute) between groups across attributes, colored based on disparity magnitude.</p>")
# report["Group Level Bias"]["Minority Group Level Bias"] = (plot_dict2,"<p style='text-align:left;'><strong>Summary:</strong> The treemaps above display all the disparities (calculated in relation to the sample minority group for each attribute) between groups across attributes, colored based on disparity magnitude.</p>")
# report["Group Level Model Fairness"]["Group Level Fairness Disparity"] = plot_dict3



# COMMAND ----------

import time

# COMMAND ----------

report_name = f"BiasReport_{int(time.time())}"
Bias_report = create_report(
    report,
    name=report_name,
    path="/dbfs/FileStore/MONITORING/bias_report",
    format=".html",
    columns= 1 ,
)

# COMMAND ----------

utils.push_plots_to_mlcore(dbutils=dbutils,
						   figure_to_save=Bias_report,
						   plot_name="Bias_Reports",
						   ext='html',
						   folder_name='Model_Evaluation',
						   )

# COMMAND ----------

# from MLCORE_SDK.helpers.auth_helper import get_container_name

# az_container_name = get_container_name(dbutils)
# container_name=f"{az_container_name}_dev"

# report_path = report_path = f"/dbfs/FileStore/MONITORING/bias_report/{report_name}"

# upload_blob_to_cloud(
#     container_name=container_name,
#     blob_path=f"{report_path}.html",
#     dbutils = dbutils ,
#     target_path = f"{report_directory}/Bias_Evaluation/BiasReport_{int(time.time())}.html",
#     resource_type = "gcp")

# dbutils.fs.rm("dbfs:" + report_path.split('/dbfs')[-1], True)
# print(f"report_directory : {report_directory}")

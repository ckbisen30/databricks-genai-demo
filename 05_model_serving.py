# Databricks notebook source

# MAGIC %md
# MAGIC # 05 — Model Serving
# MAGIC
# MAGIC **Purpose:** Deploy the registered RAG chain to a Databricks Model Serving endpoint
# MAGIC and smoke-test it via a REST call.
# MAGIC
# MAGIC **What this notebook does:**
# MAGIC 1. Creates (or updates) a Model Serving endpoint pointing to `main.genai_demo.rag_chain`
# MAGIC 2. Waits for the endpoint to become `READY`
# MAGIC 3. Enables inference tables so all requests/responses are logged to a Delta table
# MAGIC 4. Sends two test questions via the REST API and prints the answers
# MAGIC
# MAGIC **Run after:** `04_rag_chain_mlflow.py`

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration

# COMMAND ----------

import os
from dotenv import load_dotenv
load_dotenv()

# ── CONFIGURATION (loaded from .env) ─────────────────────────────────────────
CATALOG               = os.getenv("CATALOG",               "main")
SCHEMA                = os.getenv("SCHEMA",                "genai_demo")
MODEL_VERSION         = int(os.getenv("MODEL_VERSION",     3))
SERVING_ENDPOINT_NAME = os.getenv("SERVING_ENDPOINT_NAME", "genai_demo_rag_endpoint")

# ── Derived names ─────────────────────────────────────────────────────────────
REGISTERED_MODEL = f"{CATALOG}.{SCHEMA}.rag_chain"

# Inference table — logs every request + response to Delta
INFERENCE_TABLE_CATALOG = CATALOG
INFERENCE_TABLE_SCHEMA  = SCHEMA

print("=" * 55)
print(f"  Model            : {REGISTERED_MODEL} v{MODEL_VERSION}")
print(f"  Endpoint         : {SERVING_ENDPOINT_NAME}")
print(f"  Inference table  : {INFERENCE_TABLE_CATALOG}.{INFERENCE_TABLE_SCHEMA}")
print("=" * 55)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Initialise SDK Client & Inject Auth for REST Calls

# COMMAND ----------

import os
from databricks.sdk import WorkspaceClient

_ws = WorkspaceClient()

# Inject credentials so we can make direct REST calls later
_auth_headers = _ws.config.authenticate()
_token        = _auth_headers.get("Authorization", "").replace("Bearer ", "")
DATABRICKS_HOST = _ws.config.host.rstrip("/")

os.environ["DATABRICKS_HOST"]  = DATABRICKS_HOST
os.environ["DATABRICKS_TOKEN"] = _token

print(f"Workspace : {DATABRICKS_HOST}")
print("SDK client ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create or Update the Model Serving Endpoint
# MAGIC
# MAGIC - If the endpoint does not exist → create it
# MAGIC - If it exists → update it to the specified model version

# COMMAND ----------

import time


def get_endpoint(name: str) -> dict | None:
    try:
        return _ws.api_client.do("GET", f"/api/2.0/serving-endpoints/{name}")
    except Exception:
        return None


def wait_for_endpoint(name: str, timeout: int = 600) -> bool:
    start = time.time()
    while True:
        ep     = get_endpoint(name)
        state  = ep.get("state", {}).get("ready", "NOT_READY") if ep else "NOT_READY"
        config = ep.get("state", {}).get("config_update", "") if ep else ""
        print(f"  Status: ready={state}  config_update={config}")
        if state == "READY":
            print("Endpoint is READY.")
            return True
        if time.time() - start > timeout:
            print(f"WARNING: Timed out after {timeout}s. Endpoint may still be starting.")
            return False
        time.sleep(20)


served_entity = {
    "name":                  "rag_chain",
    "entity_name":           REGISTERED_MODEL,
    "entity_version":        str(MODEL_VERSION),
    "workload_size":         "Small",
    "scale_to_zero_enabled": True,
}

existing = get_endpoint(SERVING_ENDPOINT_NAME)

if existing is None:
    print(f"Creating endpoint: {SERVING_ENDPOINT_NAME} ...")
    _ws.api_client.do(
        "POST",
        "/api/2.0/serving-endpoints",
        body={
            "name": SERVING_ENDPOINT_NAME,
            "config": {
                "served_entities": [served_entity],
            },
        },
    )
    print("Endpoint creation triggered. Waiting for READY state ...\n")
else:
    current_version = (
        existing.get("config", {})
                .get("served_entities", [{}])[0]
                .get("entity_version", "?")
    )
    if current_version == str(MODEL_VERSION):
        print(f"Endpoint already exists and is on version {MODEL_VERSION}.")
    else:
        print(f"Updating endpoint from version {current_version} → {MODEL_VERSION} ...")
        _ws.api_client.do(
            "PUT",
            f"/api/2.0/serving-endpoints/{SERVING_ENDPOINT_NAME}/config",
            body={"served_entities": [served_entity]},
        )
        print("Update triggered. Waiting for READY state ...\n")

wait_for_endpoint(SERVING_ENDPOINT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Enable Inference Tables
# MAGIC
# MAGIC Inference tables log every request and response to a Delta table in Unity Catalog.
# MAGIC This powers the SQL observability dashboard in `07_sql_dashboard.sql`.

# COMMAND ----------

try:
    _ws.api_client.do(
        "PUT",
        f"/api/2.0/serving-endpoints/{SERVING_ENDPOINT_NAME}/inference-tables",
        body={
            "auto_capture_config": {
                "catalog_name": INFERENCE_TABLE_CATALOG,
                "schema_name":  INFERENCE_TABLE_SCHEMA,
                "enabled":      True,
            }
        },
    )
    print(f"Inference tables enabled → {INFERENCE_TABLE_CATALOG}.{INFERENCE_TABLE_SCHEMA}")
except Exception as e:
    print(f"Note: Could not enable inference tables (may already be enabled): {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Smoke Test via REST API
# MAGIC
# MAGIC Call the endpoint directly using `requests` to verify it returns valid answers.

# COMMAND ----------

import requests
import json

ENDPOINT_URL = f"{DATABRICKS_HOST}/serving-endpoints/{SERVING_ENDPOINT_NAME}/invocations"
HEADERS      = {"Authorization": f"Bearer {_token}", "Content-Type": "application/json"}

test_questions = [
    "What is Chandra's experience with Databricks?",
    "Which certifications does Chandra hold?",
    "What programming languages does Chandra know?",
]

print(f"Calling endpoint: {ENDPOINT_URL}\n")
print("=" * 60)

for question in test_questions:
    payload  = {"dataframe_records": [{"question": question}]}
    response = requests.post(ENDPOINT_URL, headers=HEADERS, json=payload, timeout=120)
    response.raise_for_status()

    result   = response.json()
    # Response format: {"predictions": [{"answer": "...", "sources": "..."}]}
    answer   = result.get("predictions", [{}])[0].get("answer", result)

    print(f"Q: {question}")
    print(f"A: {answer}")
    print("-" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Endpoint Summary

# COMMAND ----------

ep = get_endpoint(SERVING_ENDPOINT_NAME)
print(f"Endpoint name    : {ep['name']}")
print(f"State            : {ep.get('state', {}).get('ready', 'UNKNOWN')}")
print(f"Creation time    : {ep.get('creation_timestamp', 'N/A')}")
served = ep.get("config", {}).get("served_entities", [{}])[0]
print(f"Model version    : {served.get('entity_version', 'N/A')}")
print(f"Workload size    : {served.get('workload_size', 'N/A')}")
print(f"Scale to zero    : {served.get('scale_to_zero_enabled', 'N/A')}")
print(f"\nEndpoint URL     : {DATABRICKS_HOST}/serving-endpoints/{SERVING_ENDPOINT_NAME}/invocations")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✓ Model Serving Complete
# MAGIC
# MAGIC The RAG chain is deployed and accessible via REST API.
# MAGIC
# MAGIC Proceed to **`app/streamlit_app.py`** to build the front-end UI.

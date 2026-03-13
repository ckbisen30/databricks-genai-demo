# Databricks notebook source

# MAGIC %md
# MAGIC # 06 — Workflow Definition: Ingestion Pipeline DAG
# MAGIC
# MAGIC **Purpose:** Create (or update) a Databricks Workflow that chains the three
# MAGIC ingestion notebooks as a scheduled DAG:
# MAGIC
# MAGIC ```
# MAGIC 01_ingest_batch  →  02_chunk_and_embed  →  03_vector_search
# MAGIC ```
# MAGIC
# MAGIC Drop a new document into the Volume and trigger this workflow — the index
# MAGIC will be updated end-to-end without any manual steps.
# MAGIC
# MAGIC **Run after:** `03_vector_search.py`

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration

# COMMAND ----------

import os
from dotenv import load_dotenv
load_dotenv()

# ── CONFIGURATION (loaded from .env) ─────────────────────────────────────────
CATALOG              = os.getenv("CATALOG",              "main")
SCHEMA               = os.getenv("SCHEMA",               "genai_demo")
NOTEBOOK_PATH_PREFIX = os.getenv("NOTEBOOK_PATH_PREFIX", "")
WORKFLOW_CLUSTER_ID  = os.getenv("WORKFLOW_CLUSTER_ID",  "").strip()

JOB_NAME = "genai_demo_ingestion_pipeline"

# Schedule: run once a month on the 1st at 02:00 UTC.
SCHEDULE_CRON     = "0 0 2 1 * ?"   # Quartz cron — 1st of every month at 02:00 UTC
SCHEDULE_TIMEZONE = "UTC"

if not NOTEBOOK_PATH_PREFIX:
    raise ValueError(
        "NOTEBOOK_PATH_PREFIX is not set in .env.\n"
        "Set it to the workspace path where your notebooks are stored,\n"
        "e.g. /Users/you@example.com/GenAI_Demo"
    )

print("=" * 55)
print(f"  Job name      : {JOB_NAME}")
print(f"  Notebook path : {NOTEBOOK_PATH_PREFIX}")
print(f"  Cluster       : {WORKFLOW_CLUSTER_ID or 'Serverless'}")
print(f"  Schedule      : {SCHEDULE_CRON} ({SCHEDULE_TIMEZONE})")
print("=" * 55)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Initialise SDK Client

# COMMAND ----------

from databricks.sdk import WorkspaceClient

_ws = WorkspaceClient()
print(f"Workspace : {_ws.config.host}")
print("SDK client ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Build Task Definitions
# MAGIC
# MAGIC Three notebook tasks chained in sequence. Each task runs one ingestion
# MAGIC notebook and only starts after the previous task succeeds.

# COMMAND ----------

def notebook_task(path: str) -> dict:
    return {"notebook_path": path, "source": "WORKSPACE"}


def cluster_spec() -> dict:
    """
    Use an existing cluster if WORKFLOW_CLUSTER_ID is set,
    otherwise omit — Databricks will use Serverless compute.
    """
    if WORKFLOW_CLUSTER_ID:
        return {"existing_cluster_id": WORKFLOW_CLUSTER_ID}
    return {}


tasks = [
    {
        "task_key":      "01_ingest_batch",
        "description":   "Read PDFs/TXT/MD from Volume to Bronze Delta table",
        "notebook_task": notebook_task(f"{NOTEBOOK_PATH_PREFIX}/01_ingest_batch"),
        **cluster_spec(),
    },
    {
        "task_key":      "02_chunk_and_embed",
        "description":   "Chunk Bronze docs, generate embeddings to Silver + Gold",
        "depends_on":    [{"task_key": "01_ingest_batch"}],
        "notebook_task": notebook_task(f"{NOTEBOOK_PATH_PREFIX}/02_chunk_and_embed"),
        **cluster_spec(),
    },
    {
        "task_key":      "03_vector_search",
        "description":   "Sync Gold table to Vector Search index",
        "depends_on":    [{"task_key": "02_chunk_and_embed"}],
        "notebook_task": notebook_task(f"{NOTEBOOK_PATH_PREFIX}/03_vector_search"),
        **cluster_spec(),
    },
]

print(f"Tasks defined: {[t['task_key'] for t in tasks]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create or Update the Workflow Job
# MAGIC
# MAGIC - Job does not exist → create it
# MAGIC - Job already exists → update settings in-place (preserves run history)

# COMMAND ----------

import time

job_body = {
    "name":  JOB_NAME,
    "tasks": tasks,
    "schedule": {
        "quartz_cron_expression": SCHEDULE_CRON,
        "timezone_id":            SCHEDULE_TIMEZONE,
        "pause_status":           "UNPAUSED",
    },
    "tags": {
        "project": "genai_demo",
        "catalog": CATALOG,
        "schema":  SCHEMA,
    },
    "email_notifications": {"no_alert_for_skipped_runs": True},
}

# Find existing job by name
existing = _ws.api_client.do("GET", "/api/2.1/jobs/list", query={"name": JOB_NAME})
jobs_list = existing.get("jobs", [])

if jobs_list:
    JOB_ID = jobs_list[0]["job_id"]
    print(f"Updating existing job (id={JOB_ID}) ...")
    _ws.api_client.do(
        "POST",
        "/api/2.1/jobs/reset",
        body={"job_id": JOB_ID, "new_settings": job_body},
    )
    print("Job updated.")
else:
    print("Creating new job ...")
    result = _ws.api_client.do("POST", "/api/2.1/jobs/create", body=job_body)
    JOB_ID = result["job_id"]
    print(f"Job created (id={JOB_ID}).")

DATABRICKS_HOST = _ws.config.host.rstrip("/")
JOB_URL = f"{DATABRICKS_HOST}/#job/{JOB_ID}"
print(f"\nJob URL : {JOB_URL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Trigger a Manual Run
# MAGIC
# MAGIC Run the pipeline once now to verify all three tasks execute successfully.

# COMMAND ----------

print(f"Triggering one-time run of job {JOB_ID} ...")
run_result = _ws.api_client.do(
    "POST",
    "/api/2.1/jobs/run-now",
    body={"job_id": JOB_ID},
)
RUN_ID  = run_result["run_id"]
RUN_URL = f"{DATABRICKS_HOST}/#job/{JOB_ID}/run/{RUN_ID}"
print(f"Run started : run_id={RUN_ID}")
print(f"Run URL     : {RUN_URL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Poll Until Run Completes

# COMMAND ----------

def wait_for_run(run_id: int, timeout: int = 900) -> str:
    start = time.time()
    while True:
        run        = _ws.api_client.do("GET", "/api/2.1/jobs/runs/get", query={"run_id": run_id})
        life_cycle = run.get("state", {}).get("life_cycle_state", "")
        result     = run.get("state", {}).get("result_state", "")
        elapsed    = int(time.time() - start)

        print(f"[{elapsed}s] run={life_cycle}")
        for t in run.get("tasks", []):
            t_state  = t.get("state", {}).get("life_cycle_state", "?")
            t_result = t.get("state", {}).get("result_state", "")
            suffix   = f" -> {t_result}" if t_result else ""
            print(f"  {t['task_key']}: {t_state}{suffix}")

        if life_cycle in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
            return result

        if time.time() - start > timeout:
            print(f"WARNING: Timed out after {timeout}s.")
            return "TIMED_OUT"

        time.sleep(15)
        print()


final_state = wait_for_run(RUN_ID)
print(f"\nFinal result : {final_state}")

if final_state == "SUCCESS":
    print("All three pipeline tasks completed successfully.")
else:
    print(f"Run ended with state: {final_state}. Check the Run URL for details:")
    print(f"  {RUN_URL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Workflow Summary

# COMMAND ----------

job   = _ws.api_client.do("GET", "/api/2.1/jobs/get", query={"job_id": JOB_ID})
sched = job.get("settings", {}).get("schedule", {})

print(f"Job name     : {job['settings']['name']}")
print(f"Job ID       : {JOB_ID}")
print(f"Schedule     : {sched.get('quartz_cron_expression', 'N/A')} ({sched.get('timezone_id', '')})")
print(f"Pause status : {sched.get('pause_status', 'N/A')}")
print(f"\nTasks:")
for t in job["settings"].get("tasks", []):
    deps    = [d["task_key"] for d in t.get("depends_on", [])]
    dep_str = f"  (after {', '.join(deps)})" if deps else "  (first task)"
    print(f"  {t['task_key']}{dep_str}")
print(f"\nJob URL      : {JOB_URL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✓ Workflow Complete
# MAGIC
# MAGIC The ingestion pipeline DAG is live and scheduled to run on the 1st of every month at 02:00 UTC.
# MAGIC Drop a new document into the Volume and the index will update automatically
# MAGIC on the next run — or trigger it manually from the Job URL above.
# MAGIC
# MAGIC Proceed to **`07_sql_dashboard.sql`** to build the observability dashboard.

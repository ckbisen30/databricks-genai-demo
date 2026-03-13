# Databricks notebook source

# MAGIC %md
# MAGIC # 03 — Vector Search: Create & Sync Index
# MAGIC
# MAGIC **Purpose:** Create a Databricks Vector Search index backed by the Gold Delta table,
# MAGIC then run a smoke-test query to confirm semantic retrieval is working.
# MAGIC
# MAGIC **Inputs:**  `gold_embeddings` table (populated by `02_chunk_and_embed.py`)
# MAGIC
# MAGIC **Output:**  A live Vector Search index registered in Unity Catalog
# MAGIC
# MAGIC **Run after:** `02_chunk_and_embed.py`
# MAGIC
# MAGIC **Can be re-run safely:** Yes — skips creation if the index already exists, and triggers
# MAGIC a sync to pick up any new Gold rows.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration

# COMMAND ----------

# ── CONFIGURATION (keep in sync with 00_setup.py) ────────────────────────────
CATALOG      = "main"
SCHEMA       = "genai_demo"

GOLD_TABLE   = f"{CATALOG}.{SCHEMA}.gold_embeddings"

VS_ENDPOINT  = "genai_demo_vs_endpoint"   # Must already exist in your workspace
VS_INDEX     = f"{CATALOG}.{SCHEMA}.gold_embeddings_index"

EMBED_MODEL  = "databricks-gte-large-en"

# Column names in the Gold table
PRIMARY_KEY_COL  = "chunk_id"
EMBEDDING_COL    = "embedding"
TEXT_COL         = "chunk_text"

# How many results to return in retrieval queries
TOP_K = 3

print(f"Gold table  : {GOLD_TABLE}")
print(f"VS endpoint : {VS_ENDPOINT}")
print(f"VS index    : {VS_INDEX}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Connect to Vector Search Client

# COMMAND ----------

from databricks.sdk import WorkspaceClient

_ws = WorkspaceClient()

# Helper wrappers around the Vector Search REST API.
# Uses WorkspaceClient which handles metadata-service / OAuth / PAT auth automatically.

def vs_get_index(index_name: str) -> dict:
    return _ws.api_client.do("GET", f"/api/2.0/vector-search/indexes/{index_name}")

def vs_create_delta_sync_index(
    endpoint_name: str,
    index_name: str,
    source_table: str,
    primary_key: str,
    embedding_col: str,
    embedding_dim: int,
) -> dict:
    return _ws.api_client.do(
        "POST",
        "/api/2.0/vector-search/indexes",
        body={
            "name": index_name,
            "endpoint_name": endpoint_name,
            "primary_key": primary_key,
            "index_type": "DELTA_SYNC",
            "delta_sync_index_spec": {
                "source_table": source_table,
                "pipeline_type": "TRIGGERED",
                "embedding_vector_columns": [
                    {"name": embedding_col, "embedding_dimension": embedding_dim}
                ],
            },
        },
    )

def vs_sync_index(index_name: str) -> dict:
    return _ws.api_client.do(
        "POST", f"/api/2.0/vector-search/indexes/{index_name}/sync"
    )

def vs_query_index(
    index_name: str, query_vector: list, columns: list, num_results: int
) -> dict:
    return _ws.api_client.do(
        "POST",
        f"/api/2.0/vector-search/indexes/{index_name}/query",
        body={
            "query_vector": query_vector,
            "columns": columns,
            "num_results": num_results,
        },
    )

print("Vector Search client ready (via WorkspaceClient REST API).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create or Verify the Vector Search Index
# MAGIC
# MAGIC Uses **Delta Sync** mode: the index stays in sync with the Gold Delta table
# MAGIC automatically (or on-demand via `sync()`).
# MAGIC
# MAGIC If the index already exists this cell skips creation and moves on.

# COMMAND ----------

def index_exists(index_name: str) -> bool:
    try:
        vs_get_index(index_name)
        return True
    except Exception:
        return False


if index_exists(VS_INDEX):
    print(f"Index already exists: {VS_INDEX}")
else:
    print(f"Creating index: {VS_INDEX} ...")
    vs_create_delta_sync_index(
        endpoint_name = VS_ENDPOINT,
        index_name    = VS_INDEX,
        source_table  = GOLD_TABLE,
        primary_key   = PRIMARY_KEY_COL,
        embedding_col = EMBEDDING_COL,
        embedding_dim = 1024,             # Must match databricks-gte-large-en output
    )
    print(f"Index created: {VS_INDEX}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Sync the Index
# MAGIC
# MAGIC Triggers an incremental sync to ensure the index reflects the latest Gold table rows.
# MAGIC Waits until the sync is complete before proceeding.

# COMMAND ----------

import time

def wait_for_index_ready(index_name: str, timeout_seconds: int = 300):
    """Poll until the index status is ONLINE or timeout is reached."""
    start = time.time()
    while True:
        idx    = vs_get_index(index_name)
        status = idx.get("status", {}).get("detailed_state", "UNKNOWN")

        print(f"  Index status: {status}")

        if "ONLINE" in status.upper():
            print("Index is ONLINE.")
            return True

        if time.time() - start > timeout_seconds:
            print(f"WARNING: Timed out waiting for index after {timeout_seconds}s")
            return False

        time.sleep(10)


print(f"Triggering sync for: {VS_INDEX}")
vs_sync_index(VS_INDEX)
print("Sync triggered. Waiting for index to come ONLINE ...\n")
wait_for_index_ready(VS_INDEX)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Smoke Test: Semantic Similarity Search
# MAGIC
# MAGIC Embed a test question and retrieve the top-K most relevant chunks.

# COMMAND ----------

TEST_QUESTION = "What is Chandra's experience with Databricks?"

def embed_query(text: str) -> list[float]:
    """Embed a single query string using the Foundation Model API."""
    response = _ws.api_client.do(
        "POST",
        f"/serving-endpoints/{EMBED_MODEL}/invocations",
        body={"input": [text]},
    )
    return [float(v) for v in response["data"][0]["embedding"]]


print(f"Query: {TEST_QUESTION}\n")
query_vector = embed_query(TEST_QUESTION)

results = vs_query_index(
    index_name   = VS_INDEX,
    query_vector = query_vector,
    columns      = [PRIMARY_KEY_COL, "filename", TEXT_COL],
    num_results  = TOP_K,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Display Retrieval Results

# COMMAND ----------

hits = results.get("result", {}).get("data_array", [])
cols = [c["name"] for c in results.get("manifest", {}).get("columns", [])]

print(f"Top {TOP_K} results for: '{TEST_QUESTION}'\n")
print("-" * 70)

for i, hit in enumerate(hits, 1):
    row = dict(zip(cols, hit))
    print(f"[{i}] chunk_id : {row.get(PRIMARY_KEY_COL, 'N/A')}")
    print(f"    filename : {row.get('filename', 'N/A')}")
    print(f"    score    : {hit[-1]:.4f}")
    print(f"    text     : {row.get(TEXT_COL, '')[:200]}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✓ Vector Search Complete
# MAGIC
# MAGIC The index is live and returning semantically relevant results.
# MAGIC Proceed to **`04_rag_chain_mlflow.py`** to build the RAG chain and register it with MLflow.

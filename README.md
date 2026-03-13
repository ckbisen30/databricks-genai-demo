# Databricks Gen-AI RAG Demo

An end-to-end **Retrieval-Augmented Generation (RAG)** pipeline built on Databricks. Upload documents to a Databricks Volume and ask questions about them through a Streamlit UI — powered by Databricks Vector Search, Foundation Model API, and MLflow.

---

## Architecture

```
 Documents (PDF / TXT / MD)
        │
        ▼
 Databricks Volume  (/Volumes/main/genai_demo/source_docs)
        │
        ▼  01_ingest_batch.py
 Bronze Table  (raw extracted text)
        │
        ▼  02_chunk_and_embed.py
 Silver Table  (text chunks)
        │
        ▼  02_chunk_and_embed.py
 Gold Table    (chunks + embedding vectors)
        │
        ▼  03_vector_search.py
 Vector Search Index  (semantic similarity search)
        │
        ▼  04_rag_chain_mlflow.py
 MLflow Model Registry  (RAG chain — embed → retrieve → generate)
        │
        ├──▶  05_model_serving.py  →  REST Endpoint
        │
        └──▶  app/streamlit_app.py  →  Streamlit Q&A UI
```

**Scheduled pipeline (06_workflow_definition.py):**
```
01_ingest_batch  →  02_chunk_and_embed  →  03_vector_search
```
Runs on the 1st of every month to pick up new documents automatically.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Databricks workspace | Free Edition or higher |
| Unity Catalog enabled | `main` catalog must exist |
| Databricks Vector Search enabled | Create one endpoint manually in the UI |
| Databricks Volume created | `main.genai_demo.source_docs` |
| Python 3.10+ | Local machine |
| Git | For cloning the repo |

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/ckbisen30/databricks-genai-demo.git
cd databricks-genai-demo
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install databricks-connect databricks-sdk mlflow pypdf python-dotenv streamlit
```

### 4. Configure Databricks Connect

```bash
databricks configure
```

Enter your workspace URL and a Personal Access Token (PAT) when prompted. This writes `~/.databrickscfg`.

### 5. Create your `.env` file

Copy the example and fill in your values:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Databricks Workspace
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com

# Unity Catalog
CATALOG=main
SCHEMA=genai_demo
VOLUME_NAME=source_docs

# Models (Foundation Model API)
LLM_MODEL=databricks-meta-llama-3-3-70b-instruct
EMBED_MODEL=databricks-gte-large-en

# Vector Search
VS_ENDPOINT=genai_demo_vs_endpoint

# Model Serving
SERVING_ENDPOINT_NAME=genai_demo_rag_endpoint
MODEL_VERSION=3

# Chunking
CHUNK_SIZE=512
CHUNK_OVERLAP=64
EMBED_BATCH_SIZE=100

# Retrieval
TOP_K=3

# Workflow
NOTEBOOK_PATH_PREFIX=/Users/you@example.com/GenAI_Demo
WORKFLOW_CLUSTER_ID=
```

> **Note:** `.env` is git-ignored and never committed. `.env.example` is the safe template.

---

## Running the Pipeline

Run each script in order. All scripts use **Databricks Connect** — they execute on your Databricks cluster from your local machine.

### Step 0 — Setup

Creates the Unity Catalog schema and Bronze / Silver / Gold Delta tables.

```bash
python 00_setup.py
```

**What it does:**
- Creates `main.genai_demo` schema
- Creates `bronze_documents`, `silver_chunks`, `gold_embeddings` tables
- Verifies Volume access at `/Volumes/main/genai_demo/source_docs`

---

### Step 1 — Ingest Documents

Reads all PDF, TXT, and MD files from the Volume into the Bronze table.

```bash
python 01_ingest_batch.py
```

**What it does:**
- Lists files in the Volume using `dbutils.fs.ls()`
- Reads binary content via `spark.read.format("binaryFile")`
- Extracts text from PDFs using `pypdf`
- Deduplicates by SHA-256 content hash — safe to re-run
- Writes new rows to `bronze_documents`

**Before running:** Upload your documents to the Databricks Volume at
`/Volumes/main/genai_demo/source_docs` via the Databricks UI.

---

### Step 2 — Chunk and Embed

Splits Bronze documents into overlapping chunks and generates embedding vectors.

```bash
python 02_chunk_and_embed.py
```

**What it does:**
- Splits each document into 512-character chunks with 64-character overlap → `silver_chunks`
- Calls `databricks-gte-large-en` (Foundation Model API) to embed each chunk → `gold_embeddings`
- Embeds in batches of 100 — safe to re-run (skips already-embedded chunks)
- Embeddings are 1024-dimensional float vectors

---

### Step 3 — Vector Search Index

Creates and syncs a Databricks Vector Search Delta Sync index on the Gold table.

```bash
python 03_vector_search.py
```

**What it does:**
- Creates a Delta Sync index `main.genai_demo.gold_embeddings_index` backed by the Gold table
- Triggers an incremental sync and waits for `ONLINE` status
- Runs a smoke-test query to confirm semantic retrieval works

**Prerequisite:** Create a Vector Search endpoint named `genai_demo_vs_endpoint` in the Databricks UI
(**Compute → Vector Search → Create endpoint**) before running this script.

---

### Step 4 — RAG Chain + MLflow

Builds the end-to-end RAG chain and registers it in the MLflow Unity Catalog Model Registry.

```bash
python 04_rag_chain_mlflow.py
```

**What it does:**
- Defines three helpers: `embed_query` → `retrieve_chunks` → `generate_answer`
- Wraps the chain as `mlflow.pyfunc.PythonModel` (`RAGChain` class)
- Logs it to MLflow experiment `/Users/{you}/genai_demo_rag_experiment`
- Registers as `main.genai_demo.rag_chain` in Unity Catalog
- Loads the registered model and runs two test questions to verify

---

### Step 5 — Model Serving (Optional)

Deploys the registered RAG chain as a Databricks Model Serving REST endpoint.

```bash
python 05_model_serving.py
```

**What it does:**
- Creates (or updates) endpoint `genai_demo_rag_endpoint`
- Waits for `READY` state
- Enables inference tables — logs every request/response to Delta
- Smoke-tests the endpoint with three questions via REST API

> **Note:** The Streamlit app (Step 7) calls Databricks APIs directly and does **not** require this endpoint. Skip this step if you are on the Databricks free tier and want to conserve resources.

---

### Step 6 — Workflow DAG

Creates a Databricks Workflow that automates the ingestion pipeline on a schedule.

```bash
python 06_workflow_definition.py
```

**What it does:**
- Creates a 3-task job: `01_ingest_batch` → `02_chunk_and_embed` → `03_vector_search`
- Schedules it to run on the **1st of every month at 02:00 UTC**
- Triggers a manual run immediately and polls each task's status
- Prints a direct link to monitor the run in the Databricks UI

**Before running:** Set `NOTEBOOK_PATH_PREFIX` in `.env` to the workspace path where your notebooks are stored, e.g.:
```
NOTEBOOK_PATH_PREFIX=/Users/you@example.com/GenAI_Demo
```

---

### Step 7 — Streamlit Q&A UI

Launches the front-end chat interface.

```bash
cd databricks-genai-demo
streamlit run app/streamlit_app.py
```

Opens at **http://localhost:8501**.

**What it does:**
- Takes a question from the user
- Calls `databricks-gte-large-en` to embed the question
- Queries the Vector Search index for the top-3 relevant chunks
- Calls `databricks-meta-llama-3-3-70b-instruct` with the retrieved context
- Displays the answer and an expandable "Sources" section with chunk previews and similarity scores

**Authentication:** Uses `WorkspaceClient()` which reads credentials from `~/.databrickscfg` automatically — no extra configuration needed.

---

### Step 8 — SQL Observability Dashboard

Run the queries in `07_sql_dashboard.sql` in **Databricks SQL** to monitor the RAG endpoint.

**Queries included:**

| # | Query | What it shows |
|---|-------|---------------|
| 1 | Overview | Total requests, success rate, avg / p95 latency |
| 2 | Daily trend | Requests per day — successful vs failed |
| 3 | Hourly latency | p50 / p95 / p99 / max per hour |
| 4 | Recent Q&A | Last 25 questions + answers extracted from JSON |
| 5 | Top sources | Most cited documents + avg relevance score |
| 6 | Question frequency | Most repeated questions |
| 7 | Error analysis | Failed requests grouped by status code |
| 8 | Latency histogram | Distribution across `< 0.5s`, `0.5–1s`, `1–2s` … buckets |

**Inference table:** `main.genai_demo.genai_demo_rag_endpoint_payload`
(auto-created by Model Serving when inference tables are enabled)

**To build a live dashboard:**
1. Open **Databricks SQL → Dashboards → New Dashboard**
2. Paste each query as a widget (table, bar chart, or counter)
3. Set auto-refresh to keep metrics current

---

## Project Structure

```
databricks-genai-demo/
├── 00_setup.py                  # UC schema + Delta table DDL
├── 01_ingest_batch.py           # Volume → Bronze
├── 02_chunk_and_embed.py        # Bronze → Silver → Gold
├── 03_vector_search.py          # Gold → Vector Search index
├── 04_rag_chain_mlflow.py       # RAG chain → MLflow registry
├── 05_model_serving.py          # Registered model → REST endpoint
├── 06_workflow_definition.py    # Scheduled ingestion pipeline DAG
├── 07_sql_dashboard.sql         # Observability SQL queries
├── app/
│   └── streamlit_app.py         # Streamlit Q&A front-end
├── .env.example                 # Environment variable template (commit this)
├── .env                         # Your actual values (git-ignored)
├── .gitignore
└── databricks.yml               # Databricks Asset Bundle definition
```

---

## Environment Variables Reference

| Variable | Description | Default |
|---|---|---|
| `DATABRICKS_HOST` | Workspace URL | — |
| `CATALOG` | Unity Catalog name | `main` |
| `SCHEMA` | Schema name | `genai_demo` |
| `VOLUME_NAME` | Volume name under the schema | `source_docs` |
| `LLM_MODEL` | Foundation Model API — LLM endpoint | `databricks-meta-llama-3-3-70b-instruct` |
| `EMBED_MODEL` | Foundation Model API — embedding endpoint | `databricks-gte-large-en` |
| `VS_ENDPOINT` | Vector Search endpoint name | `genai_demo_vs_endpoint` |
| `SERVING_ENDPOINT_NAME` | Model Serving endpoint name | `genai_demo_rag_endpoint` |
| `MODEL_VERSION` | Registered model version to deploy | `3` |
| `CHUNK_SIZE` | Characters per chunk | `512` |
| `CHUNK_OVERLAP` | Overlap between chunks (characters) | `64` |
| `EMBED_BATCH_SIZE` | Chunks per embedding API call | `100` |
| `TOP_K` | Chunks to retrieve per query | `3` |
| `NOTEBOOK_PATH_PREFIX` | Workspace path for workflow notebooks | — |
| `WORKFLOW_CLUSTER_ID` | Cluster ID for workflow tasks (blank = Serverless) | `` |

---

## Delta Table Schema

### `bronze_documents`
| Column | Type | Description |
|---|---|---|
| `doc_id` | STRING | SHA-256 hash of raw text — dedup key |
| `filename` | STRING | Original filename |
| `file_path` | STRING | Full Volume path |
| `raw_text` | STRING | Extracted text content |
| `file_size_bytes` | LONG | File size in bytes |
| `file_type` | STRING | `pdf`, `txt`, or `md` |
| `ingested_at` | TIMESTAMP | UTC ingestion time |

### `silver_chunks`
| Column | Type | Description |
|---|---|---|
| `chunk_id` | STRING | `{doc_id}_{chunk_index}` |
| `doc_id` | STRING | FK → `bronze_documents.doc_id` |
| `filename` | STRING | Source filename |
| `chunk_index` | INT | Position within document |
| `chunk_text` | STRING | Chunk content |
| `created_at` | TIMESTAMP | UTC creation time |

### `gold_embeddings`
| Column | Type | Description |
|---|---|---|
| `chunk_id` | STRING | PK — matches `silver_chunks.chunk_id` |
| `doc_id` | STRING | FK → `bronze_documents.doc_id` |
| `filename` | STRING | Source filename |
| `chunk_text` | STRING | Chunk content |
| `embedding` | ARRAY\<FLOAT\> | 1024-dimensional embedding vector |

---

## Re-running Safely

All ingestion scripts are **idempotent** — they can be re-run without creating duplicates:

- `01_ingest_batch.py` — skips documents whose SHA-256 hash already exists in Bronze
- `02_chunk_and_embed.py` — skips `chunk_id` values already in Silver / Gold
- `03_vector_search.py` — skips index creation if it already exists; always triggers an incremental sync

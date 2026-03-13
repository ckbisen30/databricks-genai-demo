# Databricks notebook source

# MAGIC %md
# MAGIC # 00 — Setup & Configuration
# MAGIC
# MAGIC **Purpose:** Define all shared configuration and create the Unity Catalog schema,
# MAGIC Delta tables, and Volume reference used by every subsequent notebook.
# MAGIC
# MAGIC **Run this notebook once** before running any other notebook in the demo.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Unity Catalog is enabled in your workspace
# MAGIC - Databricks Vector Search is enabled in your workspace
# MAGIC - You have `CREATE SCHEMA`, `CREATE TABLE`, and `USE CATALOG` privileges on the target catalog
# MAGIC - A Databricks Volume already exists (or you have permission to create one)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration
# MAGIC
# MAGIC Edit the values in this cell to match your workspace. All other notebooks import this config.

# COMMAND ----------

import os
from dotenv import load_dotenv
load_dotenv()

# ── CONFIGURATION (loaded from .env) ─────────────────────────────────────────
CATALOG      = os.getenv("CATALOG",      "main")
SCHEMA       = os.getenv("SCHEMA",       "genai_demo")
VOLUME_NAME  = os.getenv("VOLUME_NAME",  "source_docs")

LLM_MODEL    = os.getenv("LLM_MODEL",    "databricks-meta-llama-3-3-70b-instruct")
EMBED_MODEL  = os.getenv("EMBED_MODEL",  "databricks-gte-large-en")

VS_ENDPOINT           = os.getenv("VS_ENDPOINT",           "genai_demo_vs_endpoint")
SERVING_ENDPOINT_NAME = os.getenv("SERVING_ENDPOINT_NAME", "genai_demo_rag_endpoint")

CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE",    512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 64))

# ── Derived names ─────────────────────────────────────────────────────────────
BRONZE_TABLE = f"{CATALOG}.{SCHEMA}.bronze_documents"
SILVER_TABLE = f"{CATALOG}.{SCHEMA}.silver_chunks"
GOLD_TABLE   = f"{CATALOG}.{SCHEMA}.gold_embeddings"
VS_INDEX     = f"{CATALOG}.{SCHEMA}.gold_embeddings_index"
VOLUME_PATH  = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}"

# ── Print summary ─────────────────────────────────────────────────────────────
print("=" * 60)
print("CONFIGURATION SUMMARY")
print("=" * 60)
print(f"  Catalog        : {CATALOG}")
print(f"  Schema         : {SCHEMA}")
print(f"  Volume path    : {VOLUME_PATH}")
print(f"  Bronze table   : {BRONZE_TABLE}")
print(f"  Silver table   : {SILVER_TABLE}")
print(f"  Gold table     : {GOLD_TABLE}")
print(f"  VS index       : {VS_INDEX}")
print(f"  LLM model      : {LLM_MODEL}")
print(f"  Embed model    : {EMBED_MODEL}")
print(f"  Chunk size     : {CHUNK_SIZE} chars  |  Overlap: {CHUNK_OVERLAP} chars")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create Catalog Schema

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
print(f"Schema ready: {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create Bronze Table
# MAGIC
# MAGIC Stores one row per raw document read from the Volume.

# COMMAND ----------

spark.sql(f"""
  CREATE TABLE IF NOT EXISTS {BRONZE_TABLE} (
    doc_id          STRING       COMMENT 'SHA-256 hash of raw text — stable unique key',
    filename        STRING       COMMENT 'Original filename from the Volume',
    file_path       STRING       COMMENT 'Full Volume path of the source file',
    raw_text        STRING       COMMENT 'Full extracted text content of the document',
    file_size_bytes LONG         COMMENT 'Size of source file in bytes',
    file_type       STRING       COMMENT 'File extension: pdf | txt | md',
    ingested_at     TIMESTAMP    COMMENT 'UTC timestamp when this row was written'
  )
  USING DELTA
  COMMENT 'Bronze layer: raw document text extracted from Databricks Volume'
  TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
""")
print(f"Bronze table ready: {BRONZE_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create Silver Table
# MAGIC
# MAGIC Stores chunked text segments derived from Bronze documents.

# COMMAND ----------

spark.sql(f"""
  CREATE TABLE IF NOT EXISTS {SILVER_TABLE} (
    chunk_id    STRING       COMMENT 'Unique ID: doc_id + chunk index',
    doc_id      STRING       COMMENT 'Foreign key to bronze_documents.doc_id',
    filename    STRING       COMMENT 'Source filename (denormalised for convenience)',
    chunk_index INT          COMMENT 'Sequential position of this chunk within the document',
    chunk_text  STRING       COMMENT 'Text content of this chunk',
    created_at  TIMESTAMP    COMMENT 'UTC timestamp when this chunk was written'
  )
  USING DELTA
  COMMENT 'Silver layer: chunked text segments ready for embedding'
  TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
""")
print(f"Silver table ready: {SILVER_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Create Gold Table
# MAGIC
# MAGIC Stores chunks with their embedding vectors. This table backs the Vector Search index.

# COMMAND ----------

spark.sql(f"""
  CREATE TABLE IF NOT EXISTS {GOLD_TABLE} (
    chunk_id   STRING        COMMENT 'Primary key — matches silver_chunks.chunk_id',
    doc_id     STRING        COMMENT 'Foreign key to bronze_documents.doc_id',
    filename   STRING        COMMENT 'Source filename',
    chunk_text STRING        COMMENT 'Text content of this chunk',
    embedding  ARRAY<FLOAT>  COMMENT 'Embedding vector from {EMBED_MODEL}'
  )
  USING DELTA
  COMMENT 'Gold layer: chunks with embedding vectors — source for Vector Search index'
  TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
""")
print(f"Gold table ready: {GOLD_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Verify Volume Access

# COMMAND ----------

try:
    files = dbutils.fs.ls(VOLUME_PATH)
    print(f"Volume accessible: {VOLUME_PATH}")
    print(f"Files found: {len(files)}")
    for f in files[:10]:
        print(f"  - {f.name}  ({f.size:,} bytes)")
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more")
except Exception as e:
    print(f"WARNING: Could not access Volume: {VOLUME_PATH}")
    print(f"Error: {e}")
    print("Please verify the Volume exists and the path config is correct.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Setup Complete ✓
# MAGIC
# MAGIC All resources are ready. Proceed to **`01_ingest_batch.py`** to read documents
# MAGIC from the Volume and populate the Bronze table.
# MAGIC
# MAGIC > **Tip:** Re-run this notebook at any time to verify schema/table state without modifying data.

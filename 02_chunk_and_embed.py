# Databricks notebook source

# MAGIC %md
# MAGIC # 02 — Chunking & Embedding: Bronze → Silver → Gold
# MAGIC
# MAGIC **Purpose:**
# MAGIC 1. Read raw document text from the Bronze table
# MAGIC 2. Split each document into overlapping text chunks → Silver table
# MAGIC 3. Generate embedding vectors for each chunk via the Databricks Foundation Model API → Gold table
# MAGIC
# MAGIC **Inputs:**  `bronze_documents` table
# MAGIC
# MAGIC **Outputs:** `silver_chunks` table, `gold_embeddings` table
# MAGIC
# MAGIC **Run after:** `01_ingest_batch.py`
# MAGIC
# MAGIC **Can be re-run safely:** Yes — only processes chunks not already in Silver/Gold.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration

# COMMAND ----------

import os
from dotenv import load_dotenv
load_dotenv()

# ── CONFIGURATION (loaded from .env) ─────────────────────────────────────────
CATALOG     = os.getenv("CATALOG",    "main")
SCHEMA      = os.getenv("SCHEMA",     "genai_demo")
EMBED_MODEL = os.getenv("EMBED_MODEL","databricks-gte-large-en")

CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE",       512))
CHUNK_OVERLAP    = int(os.getenv("CHUNK_OVERLAP",    64))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", 100))

# ── Derived names ─────────────────────────────────────────────────────────────
BRONZE_TABLE = f"{CATALOG}.{SCHEMA}.bronze_documents"
SILVER_TABLE = f"{CATALOG}.{SCHEMA}.silver_chunks"
GOLD_TABLE   = f"{CATALOG}.{SCHEMA}.gold_embeddings"

print(f"Bronze  : {BRONZE_TABLE}")
print(f"Silver  : {SILVER_TABLE}")
print(f"Gold    : {GOLD_TABLE}")
print(f"Model   : {EMBED_MODEL}")
print(f"Chunk   : {CHUNK_SIZE} chars  |  Overlap: {CHUNK_OVERLAP} chars")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Helper: Text Chunker
# MAGIC
# MAGIC Splits a document into overlapping fixed-size character windows.
# MAGIC Simple and dependency-free — no LangChain required for this step.

# COMMAND ----------

from datetime import datetime, timezone


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split `text` into overlapping windows of `chunk_size` characters
    with `overlap` characters of context carried over between chunks.
    Returns a list of non-empty chunk strings.
    """
    chunks = []
    start  = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == length:
            break
        start += chunk_size - overlap

    return chunks

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load Bronze Documents

# COMMAND ----------

bronze_df = spark.table(BRONZE_TABLE)
bronze_rows = bronze_df.select("doc_id", "filename", "raw_text").collect()

print(f"Documents loaded from Bronze: {len(bronze_rows)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Chunk Documents → Silver Table
# MAGIC
# MAGIC Deduplicates by `chunk_id` so re-runs are safe.

# COMMAND ----------

from pyspark.sql import Row

# Load existing chunk_ids to avoid duplicates
existing_chunk_ids = set(
    row.chunk_id
    for row in spark.table(SILVER_TABLE).select("chunk_id").collect()
)
print(f"Existing chunks in Silver: {len(existing_chunk_ids)}")

# Build new chunk records
now = datetime.now(timezone.utc)
new_chunks = []

for doc in bronze_rows:
    chunks = chunk_text(doc.raw_text, CHUNK_SIZE, CHUNK_OVERLAP)
    for i, chunk_text_val in enumerate(chunks):
        chunk_id = f"{doc.doc_id}_{i}"
        if chunk_id in existing_chunk_ids:
            continue
        new_chunks.append(Row(
            chunk_id    = chunk_id,
            doc_id      = doc.doc_id,
            filename    = doc.filename,
            chunk_index = i,
            chunk_text  = chunk_text_val,
            created_at  = now,
        ))

print(f"New chunks to write : {len(new_chunks)}")
print(f"Duplicates skipped  : {sum(1 for doc in bronze_rows for i in range(len(chunk_text(doc.raw_text, CHUNK_SIZE, CHUNK_OVERLAP))) if f'{doc.doc_id}_{i}' in existing_chunk_ids)}")

# COMMAND ----------

if new_chunks:
    from pyspark.sql.functions import col
    from pyspark.sql.types import IntegerType

    silver_df = (
        spark.createDataFrame(new_chunks)
             .withColumn("chunk_index", col("chunk_index").cast(IntegerType()))
    )
    (
        silver_df.write
                 .format("delta")
                 .mode("append")
                 .saveAsTable(SILVER_TABLE)
    )
    print(f"Written {len(new_chunks)} chunks to {SILVER_TABLE}")
else:
    print("Silver table already up to date.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Verify Silver Table

# COMMAND ----------

silver_df = spark.table(SILVER_TABLE)
total_chunks = silver_df.count()
print(f"Total chunks in Silver: {total_chunks}\n")

silver_df \
    .select("chunk_id", "filename", "chunk_index", "chunk_text") \
    .orderBy("filename", "chunk_index") \
    .show(5, truncate=80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Generate Embeddings → Gold Table
# MAGIC
# MAGIC Calls the Databricks Foundation Model API in batches.
# MAGIC Only embeds chunks not already present in the Gold table.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

# WorkspaceClient uses the same auth as Databricks Connect (metadata-service, OAuth, PAT)
_ws = WorkspaceClient()

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Call the Databricks Foundation Model embedding endpoint for a batch of texts.
    Returns a list of embedding vectors in the same order as the input.
    """
    response = _ws.api_client.do(
        "POST",
        f"/serving-endpoints/{EMBED_MODEL}/invocations",
        body={"input": texts},
    )
    # Response format: {"data": [{"embedding": [...], "index": N}, ...]}
    sorted_items = sorted(response["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in sorted_items]

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6a. Load chunks that need embedding

# COMMAND ----------

# Find chunk_ids already in Gold
existing_gold_ids = set(
    row.chunk_id
    for row in spark.table(GOLD_TABLE).select("chunk_id").collect()
)
print(f"Chunks already in Gold: {len(existing_gold_ids)}")

# Load Silver chunks not yet in Gold
silver_rows = spark.table(SILVER_TABLE).collect()
to_embed = [r for r in silver_rows if r.chunk_id not in existing_gold_ids]
print(f"Chunks to embed       : {len(to_embed)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6b. Embed in batches and write to Gold

# COMMAND ----------

gold_records = []

for batch_start in range(0, len(to_embed), EMBED_BATCH_SIZE):
    batch      = to_embed[batch_start : batch_start + EMBED_BATCH_SIZE]
    batch_texts = [r.chunk_text for r in batch]

    print(f"  Embedding batch {batch_start // EMBED_BATCH_SIZE + 1} "
          f"({len(batch)} chunks) ...", end=" ")

    vectors = get_embeddings(batch_texts)

    for row, vector in zip(batch, vectors):
        gold_records.append(Row(
            chunk_id   = row.chunk_id,
            doc_id     = row.doc_id,
            filename   = row.filename,
            chunk_text = row.chunk_text,
            embedding  = vector,
        ))

    print("done")

print(f"\nTotal gold records prepared: {len(gold_records)}")

# COMMAND ----------

if gold_records:
    from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType

    gold_schema = StructType([
        StructField("chunk_id",   StringType(),         False),
        StructField("doc_id",     StringType(),         False),
        StructField("filename",   StringType(),         False),
        StructField("chunk_text", StringType(),         False),
        StructField("embedding",  ArrayType(FloatType()), False),
    ])

    # Ensure all embedding values are Python floats (API can return int for zero values)
    gold_dicts = [
        {**r.asDict(), "embedding": [float(v) for v in r["embedding"]]}
        for r in gold_records
    ]

    gold_df = spark.createDataFrame(gold_dicts, schema=gold_schema)
    (
        gold_df.write
               .format("delta")
               .mode("append")
               .saveAsTable(GOLD_TABLE)
    )
    print(f"Written {len(gold_records)} embedded chunks to {GOLD_TABLE}")
else:
    print("Gold table already up to date.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Verify Gold Table

# COMMAND ----------

gold_df = spark.table(GOLD_TABLE)
total_gold = gold_df.count()
print(f"Total rows in Gold table: {total_gold}\n")

gold_df \
    .select("chunk_id", "filename", "chunk_text") \
    .orderBy("filename", "chunk_id") \
    .show(5, truncate=80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Sanity Check: Inspect One Embedding

# COMMAND ----------

sample = gold_df.select("chunk_id", "embedding").limit(1).collect()
if sample:
    row = sample[0]
    vec = row.embedding
    print(f"chunk_id        : {row.chunk_id}")
    print(f"Embedding dims  : {len(vec)}")
    print(f"First 5 values  : {vec[:5]}")
else:
    print("Gold table is empty.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✓ Chunking & Embedding Complete
# MAGIC
# MAGIC Silver and Gold tables are populated. Proceed to **`03_vector_search.py`**
# MAGIC to create and sync the Databricks Vector Search index.

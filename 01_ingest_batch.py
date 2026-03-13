# Databricks notebook source

# MAGIC %md
# MAGIC # 01 — Batch Ingestion: Volume → Bronze Delta Table
# MAGIC
# MAGIC **Purpose:** Read all supported documents (PDF, TXT, MD) from a Databricks Volume
# MAGIC using a standard Spark batch job and write them as rows to the Bronze Delta table.
# MAGIC
# MAGIC **Inputs:**  Files in `VOLUME_PATH` (configured in `00_setup.py`)
# MAGIC
# MAGIC **Output:**  `BRONZE_TABLE` — one row per document with extracted raw text
# MAGIC
# MAGIC **Run after:** `00_setup.py`
# MAGIC
# MAGIC **Can be re-run safely:** Yes — uses `doc_id` (SHA-256 hash) to skip already-ingested documents.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Configuration
# MAGIC
# MAGIC Re-declare config here (mirrors `00_setup.py`). If you changed any values there, update them here too.

# COMMAND ----------

import os
from dotenv import load_dotenv
load_dotenv()

# ── CONFIGURATION (loaded from .env) ─────────────────────────────────────────
CATALOG     = os.getenv("CATALOG",     "main")
SCHEMA      = os.getenv("SCHEMA",      "genai_demo")
VOLUME_NAME = os.getenv("VOLUME_NAME", "source_docs")

# ── Derived names ─────────────────────────────────────────────────────────────
BRONZE_TABLE = f"{CATALOG}.{SCHEMA}.bronze_documents"
VOLUME_PATH  = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}"

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

print(f"Reading from : {VOLUME_PATH}")
print(f"Writing to   : {BRONZE_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Helper: Extract Text from a Single File

# COMMAND ----------

import io
import os
import hashlib
from datetime import datetime, timezone

from pypdf import PdfReader


def extract_text_from_bytes(content: bytes, filename: str) -> str:
    """
    Extract plain text from raw file bytes.
    Supports PDF, TXT, and MD based on filename extension.
    """
    ext = os.path.splitext(filename)[1].lower()
    try:
        if ext == ".pdf":
            reader = PdfReader(io.BytesIO(content))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages).strip()
        elif ext in {".txt", ".md"}:
            return content.decode("utf-8", errors="replace").strip()
    except Exception as e:
        print(f"  WARNING: Could not extract text from {filename}: {e}")
    return ""


def make_doc_id(text: str) -> str:
    """Stable unique key: SHA-256 of the raw text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Scan Volume and Build Document Records
# MAGIC
# MAGIC Uses `dbutils.fs.ls()` to list files and `spark.read.format("binaryFile")`
# MAGIC to read file contents — both work over Databricks Connect.

# COMMAND ----------

print(f"Scanning: {VOLUME_PATH}\n")

# List files in the Volume
volume_files = dbutils.fs.ls(VOLUME_PATH)

# Filter to supported extensions
supported = [
    f for f in volume_files
    if os.path.splitext(f.name)[1].lower() in SUPPORTED_EXTENSIONS
]
skipped = [
    f for f in volume_files
    if os.path.splitext(f.name)[1].lower() not in SUPPORTED_EXTENSIONS
]

for f in skipped:
    print(f"  SKIP  (unsupported type): {f.name}")

print(f"  Files to process: {len(supported)}\n")

# Read all supported files as binary in one Spark job
file_paths = [f.path for f in supported]

if file_paths:
    binary_df = (
        spark.read.format("binaryFile")
             .load(file_paths)
             .select("path", "content", "length")
             .collect()
    )
else:
    binary_df = []

# Extract text from binary content
records = []
now = datetime.now(timezone.utc)

for row in binary_df:
    fname    = os.path.basename(row.path)
    ext      = os.path.splitext(fname)[1].lower()
    content  = bytes(row.content)

    print(f"  READ  ({ext[1:].upper()}, {row.length:,} bytes): {fname}")

    raw_text = extract_text_from_bytes(content, fname)

    if not raw_text:
        print(f"         └─ WARNING: no text extracted, skipping.")
        continue

    records.append({
        "doc_id":          make_doc_id(raw_text),
        "filename":        fname,
        "file_path":       row.path,
        "raw_text":        raw_text,
        "file_size_bytes": row.length,
        "file_type":       ext.lstrip("."),
        "ingested_at":     now,
    })

print(f"\nDocuments ready to ingest: {len(records)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Deduplicate Against Existing Bronze Data
# MAGIC
# MAGIC Skip any document whose `doc_id` (content hash) is already in the Bronze table.
# MAGIC This makes the notebook safe to re-run without creating duplicates.

# COMMAND ----------

from pyspark.sql import Row

if records:
    # Load existing doc_ids from Bronze
    existing_ids = set(
        row.doc_id
        for row in spark.table(BRONZE_TABLE).select("doc_id").collect()
    )

    new_records = [r for r in records if r["doc_id"] not in existing_ids]

    print(f"Already in Bronze : {len(existing_ids)}")
    print(f"New documents     : {len(new_records)}")
    print(f"Duplicates skipped: {len(records) - len(new_records)}")
else:
    new_records = []
    print("No records to process.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Write New Documents to Bronze Table

# COMMAND ----------

if new_records:
    df = spark.createDataFrame([Row(**r) for r in new_records])

    (
        df.write
          .format("delta")
          .mode("append")
          .saveAsTable(BRONZE_TABLE)
    )
    print(f"Written {len(new_records)} new document(s) to {BRONZE_TABLE}")
else:
    print("Nothing to write — Bronze table is already up to date.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Verify: Preview Bronze Table

# COMMAND ----------

bronze_df = spark.table(BRONZE_TABLE)

total = bronze_df.count()
print(f"Total rows in Bronze table: {total}\n")

bronze_df \
    .select("doc_id", "filename", "file_type", "file_size_bytes", "ingested_at") \
    .orderBy("ingested_at", ascending=False) \
    .show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Quick Sanity Check: Inspect One Document

# COMMAND ----------

sample = bronze_df.select("filename", "raw_text").limit(1).collect()

if sample:
    row = sample[0]
    print(f"File   : {row.filename}")
    print(f"Length : {len(row.raw_text):,} characters")
    print(f"\n--- First 500 characters ---\n")
    print(row.raw_text[:500])
else:
    print("Bronze table is empty.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✓ Ingestion Complete
# MAGIC
# MAGIC Bronze table is populated. Proceed to **`02_chunk_and_embed.py`** to split documents
# MAGIC into chunks and generate embedding vectors.

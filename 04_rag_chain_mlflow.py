# Databricks notebook source

# MAGIC %md
# MAGIC # 04 — RAG Chain + MLflow
# MAGIC
# MAGIC **Purpose:** Build a Retrieval-Augmented Generation (RAG) chain that:
# MAGIC 1. Embeds the user's question via the Foundation Model API
# MAGIC 2. Retrieves the top-K relevant chunks from the Vector Search index
# MAGIC 3. Assembles a prompt with retrieved context
# MAGIC 4. Calls `databricks-meta-llama-3-1-70b-instruct` to generate a grounded answer
# MAGIC
# MAGIC The chain is then logged and registered to the Unity Catalog Model Registry via MLflow.
# MAGIC
# MAGIC **Run after:** `03_vector_search.py`

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration

# COMMAND ----------

# ── CONFIGURATION (keep in sync with 00_setup.py) ────────────────────────────
CATALOG      = "main"
SCHEMA       = "genai_demo"

VS_ENDPOINT  = "genai_demo_vs_endpoint"
VS_INDEX     = f"{CATALOG}.{SCHEMA}.gold_embeddings_index"

LLM_MODEL    = "databricks-meta-llama-3-3-70b-instruct"
EMBED_MODEL  = "databricks-gte-large-en"

TOP_K        = 3   # Number of chunks to retrieve per query

from databricks.sdk import WorkspaceClient as _WC
_current_user     = _WC().current_user.me().user_name
MLFLOW_EXPERIMENT = f"/Users/{_current_user}/genai_demo_rag_experiment"
REGISTERED_MODEL  = f"{CATALOG}.{SCHEMA}.rag_chain"

DATABRICKS_HOST = spark.conf.get("spark.databricks.workspaceUrl", "")
if not DATABRICKS_HOST.startswith("http"):
    DATABRICKS_HOST = f"https://{DATABRICKS_HOST}"

print("=" * 55)
print(f"  VS index       : {VS_INDEX}")
print(f"  LLM            : {LLM_MODEL}")
print(f"  Embed model    : {EMBED_MODEL}")
print(f"  Top-K          : {TOP_K}")
print(f"  MLflow model   : {REGISTERED_MODEL}")
print("=" * 55)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Initialise Clients

# COMMAND ----------

from databricks.sdk import WorkspaceClient

_ws = WorkspaceClient()

print("WorkspaceClient ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Helper Functions
# MAGIC
# MAGIC Three pure-Python helpers that together form the RAG pipeline:
# MAGIC - `embed_query` — embed the user question
# MAGIC - `retrieve_chunks` — query Vector Search for top-K chunks
# MAGIC - `generate_answer` — call the LLM with retrieved context

# COMMAND ----------

import json


def embed_query(question: str) -> list:
    """Embed a single question string using the Foundation Model API."""
    response = _ws.api_client.do(
        "POST",
        f"/serving-endpoints/{EMBED_MODEL}/invocations",
        body={"input": [question]},
    )
    return response["data"][0]["embedding"]


def retrieve_chunks(question_embedding: list, top_k: int = TOP_K) -> list:
    """
    Query the Vector Search index with a pre-computed embedding.
    Returns a list of dicts with keys: chunk_id, filename, chunk_text, score.
    """
    response = _ws.api_client.do(
        "POST",
        f"/api/2.0/vector-search/indexes/{VS_INDEX}/query",
        body={
            "query_vector":    question_embedding,
            "columns":         ["chunk_id", "filename", "chunk_text"],
            "num_results":     top_k,
        },
    )

    results    = response.get("result", {})
    data_array = results.get("data_array", [])
    manifest   = results.get("manifest", {})

    # VS returns columns as a flat list; score is appended as the last column
    col_names = [c["name"] for c in manifest.get("columns", [])]

    chunks = []
    for row in data_array:
        record = dict(zip(col_names, row))
        # Fallback: if column mapping failed, try positional access
        if not record and len(row) >= 3:
            record = {
                "chunk_id":   row[0],
                "filename":   row[1],
                "chunk_text": row[2],
                "score":      row[-1] if len(row) > 3 else 0.0,
            }
        chunks.append({
            "chunk_id":   record.get("chunk_id", ""),
            "filename":   record.get("filename", ""),
            "chunk_text": record.get("chunk_text", ""),
            "score":      record.get("score", 0.0),
        })
    return chunks


def generate_answer(question: str, chunks: list) -> str:
    """
    Build a prompt from retrieved chunks and call the LLM.
    Returns the generated answer string.
    """
    context_blocks = "\n\n".join(
        f"[Source {i+1}: {c['filename']}]\n{c['chunk_text']}"
        for i, c in enumerate(chunks)
    )

    system_prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY "
        "the provided context. If the context does not contain enough information, "
        "say so clearly. Do not make up information."
    )

    user_message = (
        f"Context:\n{context_blocks}\n\n"
        f"Question: {question}"
    )

    response = _ws.api_client.do(
        "POST",
        f"/serving-endpoints/{LLM_MODEL}/invocations",
        body={
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            "max_tokens": 512,
            "temperature": 0.1,
        },
    )
    return response["choices"][0]["message"]["content"].strip()


def rag(question: str) -> dict:
    """End-to-end RAG: embed → retrieve → generate. Returns full result dict."""
    embedding = embed_query(question)
    chunks    = retrieve_chunks(embedding)
    answer    = generate_answer(question, chunks)
    return {
        "question": question,
        "answer":   answer,
        "sources":  [{"filename": c["filename"], "chunk_text": c["chunk_text"], "score": c["score"]} for c in chunks],
    }

print("RAG helper functions defined.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Smoke Test the RAG Chain

# COMMAND ----------

test_question = "What is Chandra's experience with Databricks?"

print(f"Question: {test_question}\n")
result = rag(test_question)

print(f"Answer:\n{result['answer']}\n")
print("Sources:")
for i, s in enumerate(result["sources"]):
    print(f"  [{i+1}] {s['filename']}  (score: {s['score']:.4f})")
    print(f"       {s['chunk_text'][:120].strip()} ...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Wrap as an MLflow PyFunc Model
# MAGIC
# MAGIC Wrapping the chain as `mlflow.pyfunc` makes it portable — it can be logged,
# MAGIC versioned, and deployed to Model Serving with a standard `predict()` interface.

# COMMAND ----------

import mlflow
import mlflow.pyfunc
import pandas as pd


class RAGChain(mlflow.pyfunc.PythonModel):
    """
    MLflow PyFunc wrapper for the RAG chain.

    Config is stored in the instance so it survives CloudPickle serialisation
    and is available when the model is loaded from the registry in any context.

    Input:  pandas DataFrame with a single column `question` (string)
    Output: pandas DataFrame with columns `answer` and `sources` (JSON string)
    """

    def __init__(self, embed_model, vs_index, llm_model, top_k):
        self.embed_model = embed_model
        self.vs_index    = vs_index
        self.llm_model   = llm_model
        self.top_k       = top_k

    def load_context(self, context):
        from databricks.sdk import WorkspaceClient
        self._ws = WorkspaceClient()

    def _embed(self, question: str) -> list:
        resp = self._ws.api_client.do(
            "POST",
            f"/serving-endpoints/{self.embed_model}/invocations",
            body={"input": [question]},
        )
        return resp["data"][0]["embedding"]

    def _retrieve(self, embedding: list) -> list:
        resp = self._ws.api_client.do(
            "POST",
            f"/api/2.0/vector-search/indexes/{self.vs_index}/query",
            body={"query_vector": embedding, "columns": ["chunk_id", "filename", "chunk_text"], "num_results": self.top_k},
        )
        results  = resp.get("result", {})
        data     = results.get("data_array", [])
        manifest = results.get("manifest", {})
        cols     = [c["name"] for c in manifest.get("columns", [])]
        chunks   = [dict(zip(cols, row)) for row in data]
        # Fallback to positional access if column mapping fails
        if data and not chunks[0]:
            chunks = [{"chunk_id": r[0], "filename": r[1], "chunk_text": r[2], "score": r[-1] if len(r) > 3 else 0.0} for r in data]
        return chunks

    def _generate(self, question: str, chunks: list) -> str:
        context_blocks = "\n\n".join(
            f"[Source {i+1}: {c.get('filename','')}]\n{c.get('chunk_text','')}"
            for i, c in enumerate(chunks)
        )
        resp = self._ws.api_client.do(
            "POST",
            f"/serving-endpoints/{self.llm_model}/invocations",
            body={
                "messages": [
                    {"role": "system", "content": "Answer using ONLY the provided context. Be concise."},
                    {"role": "user",   "content": f"Context:\n{context_blocks}\n\nQuestion: {question}"},
                ],
                "max_tokens": 512,
                "temperature": 0.1,
            },
        )
        return resp["choices"][0]["message"]["content"].strip()

    def predict(self, context, model_input):
        import json, pandas as pd

        if isinstance(model_input, pd.DataFrame):
            questions = model_input["question"].tolist()
        elif isinstance(model_input, dict):
            questions = [model_input.get("question", "")]
        else:
            questions = [str(model_input)]

        rows = []
        for q in questions:
            emb    = self._embed(q)
            chunks = self._retrieve(emb)
            answer = self._generate(q, chunks)
            rows.append({
                "answer":  answer,
                "sources": json.dumps([
                    {"filename": c.get("filename"), "chunk_text": c.get("chunk_text"), "score": c.get("score")}
                    for c in chunks
                ]),
            })
        return pd.DataFrame(rows)


print("RAGChain MLflow model class defined.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Log the Chain to MLflow

# COMMAND ----------

import os
import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

# Inject auth credentials from the SDK so MLflow can authenticate
# (required when running locally via Databricks Connect metadata-service auth)
_auth_headers = _ws.config.authenticate()
_token = _auth_headers.get("Authorization", "").replace("Bearer ", "")
os.environ["DATABRICKS_HOST"]  = _ws.config.host.rstrip("/")
os.environ["DATABRICKS_TOKEN"] = _token

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(MLFLOW_EXPERIMENT)

input_schema  = Schema([ColSpec("string", "question")])
output_schema = Schema([ColSpec("string", "answer"), ColSpec("string", "sources")])
signature     = ModelSignature(inputs=input_schema, outputs=output_schema)

sample_input  = pd.DataFrame({"question": ["What is Chandra's experience with Databricks?"]})
sample_output = pd.DataFrame({"answer": ["Sample answer."], "sources": ["[]"]})

with mlflow.start_run(run_name="rag_chain_v1") as run:
    mlflow.log_param("llm_model",   LLM_MODEL)
    mlflow.log_param("embed_model", EMBED_MODEL)
    mlflow.log_param("vs_index",    VS_INDEX)
    mlflow.log_param("top_k",       TOP_K)

    model_info = mlflow.pyfunc.log_model(
        artifact_path    = "rag_chain",
        python_model     = RAGChain(embed_model=EMBED_MODEL, vs_index=VS_INDEX, llm_model=LLM_MODEL, top_k=TOP_K),
        signature        = signature,
        input_example    = sample_input,
        registered_model_name = REGISTERED_MODEL,
        pip_requirements = ["databricks-sdk", "pandas"],
    )

    RUN_ID    = run.info.run_id
    MODEL_URI = model_info.model_uri

print(f"Run ID        : {RUN_ID}")
print(f"Model URI     : {MODEL_URI}")
print(f"Registered as : {REGISTERED_MODEL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Verify: Load from Registry and Run Inference

# COMMAND ----------

loaded_model = mlflow.pyfunc.load_model(MODEL_URI)

test_df = pd.DataFrame({"question": [
    "What is Chandra's experience with Databricks?",
    "Which certifications does Chandra hold?",
]})

predictions = loaded_model.predict(test_df)

for i, row in predictions.iterrows():
    print(f"\nQ: {test_df.iloc[i]['question']}")
    print(f"A: {row['answer']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✓ RAG Chain Complete
# MAGIC
# MAGIC - Chain logged under MLflow experiment: `{MLFLOW_EXPERIMENT}`
# MAGIC - Model registered in Unity Catalog: `{REGISTERED_MODEL}`
# MAGIC
# MAGIC Proceed to **`05_model_serving.py`** to deploy the registered model as a REST endpoint.

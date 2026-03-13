"""
Streamlit front-end for the Databricks RAG Demo.

Calls the Foundation Model API (embed + LLM) and Vector Search index
directly via the Databricks SDK — no Model Serving endpoint required.

Run with:
    cd GenAI_Demo
    streamlit run app/streamlit_app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv
from databricks.sdk import WorkspaceClient

load_dotenv()

# ── CONFIGURATION (loaded from .env) ─────────────────────────────────────────
CATALOG     = os.getenv("CATALOG",     "main")
SCHEMA      = os.getenv("SCHEMA",      "genai_demo")
LLM_MODEL   = os.getenv("LLM_MODEL",   "databricks-meta-llama-3-3-70b-instruct")
EMBED_MODEL = os.getenv("EMBED_MODEL", "databricks-gte-large-en")
TOP_K       = int(os.getenv("TOP_K",   3))

# ── Derived names ─────────────────────────────────────────────────────────────
VS_INDEX = f"{CATALOG}.{SCHEMA}.gold_embeddings_index"

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Databricks RAG Demo",
    page_icon="🔍",
    layout="centered",
)

st.title("🔍 Databricks RAG — Document Q&A")
st.caption("Ask questions about the documents ingested into the Vector Search index.")

# ── SDK CLIENT (cached so it's only created once per session) ─────────────────
@st.cache_resource(show_spinner=False)
def get_ws() -> WorkspaceClient:
    return WorkspaceClient()


# ── RAG HELPERS ───────────────────────────────────────────────────────────────
def embed_query(ws: WorkspaceClient, question: str) -> list:
    resp = ws.api_client.do(
        "POST",
        f"/serving-endpoints/{EMBED_MODEL}/invocations",
        body={"input": [question]},
    )
    return resp["data"][0]["embedding"]


def retrieve_chunks(ws: WorkspaceClient, embedding: list) -> list:
    resp = ws.api_client.do(
        "POST",
        f"/api/2.0/vector-search/indexes/{VS_INDEX}/query",
        body={
            "query_vector": embedding,
            "columns":      ["chunk_id", "filename", "chunk_text"],
            "num_results":  TOP_K,
        },
    )
    results  = resp.get("result", {})
    data     = results.get("data_array", [])
    manifest = results.get("manifest", {})
    cols     = [c["name"] for c in manifest.get("columns", [])]
    chunks   = [dict(zip(cols, row)) for row in data]
    if data and not chunks[0]:
        chunks = [
            {"chunk_id": r[0], "filename": r[1], "chunk_text": r[2],
             "score": r[-1] if len(r) > 3 else 0.0}
            for r in data
        ]
    return chunks


def generate_answer(ws: WorkspaceClient, question: str, chunks: list) -> str:
    context_blocks = "\n\n".join(
        f"[Source {i+1}: {c.get('filename', '')}]\n{c.get('chunk_text', '')}"
        for i, c in enumerate(chunks)
    )
    resp = ws.api_client.do(
        "POST",
        f"/serving-endpoints/{LLM_MODEL}/invocations",
        body={
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Answer the user's question using ONLY "
                        "the provided context. If the context does not contain enough information, "
                        "say so clearly. Do not make up information."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context_blocks}\n\nQuestion: {question}",
                },
            ],
            "max_tokens":  512,
            "temperature": 0.1,
        },
    )
    return resp["choices"][0]["message"]["content"].strip()


def rag(question: str) -> dict:
    ws      = get_ws()
    emb     = embed_query(ws, question)
    chunks  = retrieve_chunks(ws, emb)
    answer  = generate_answer(ws, question, chunks)
    return {"answer": answer, "chunks": chunks}


# ── UI ────────────────────────────────────────────────────────────────────────
question = st.text_input(
    "Your question",
    placeholder="e.g. What is Chandra's experience with Databricks?",
)

if st.button("Ask", type="primary", disabled=not question.strip()):
    with st.spinner("Retrieving and generating answer..."):
        try:
            result = rag(question.strip())
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    st.subheader("Answer")
    st.write(result["answer"])

    with st.expander(f"Sources ({len(result['chunks'])} chunks retrieved)"):
        for i, chunk in enumerate(result["chunks"]):
            score = chunk.get("score", 0.0)
            fname = chunk.get("filename", "unknown")
            text  = chunk.get("chunk_text", "")
            st.markdown(f"**[{i+1}] {fname}** — score: `{score:.4f}`")
            st.text(text[:400] + ("..." if len(text) > 400 else ""))
            st.divider()

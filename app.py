"""
app.py — Streamlit UI for the RAG chatbot.

Features:
    - Reranker toggle: Cohere (API) or BGE (local) — live switchable
    - Chat interface with multi-turn conversation memory
    - Source panel showing normalised rerank score (authoritative)
      and raw embedding score (reference only)
    - Synthesis flag, page number, and chunk ID displayed per source
    - Settings sidebar for retrieval tuning (k, top_n)
    - Clear conversation button with memory reset
"""

import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from src.chain import initialise_pipeline, run_query

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="📄 RAG Chatbot",
    page_icon="📄",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Configuration
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    st.subheader("Reranker")
    reranker_choice = st.radio(
        "Select reranker",
        options=["Cohere (API)", "BGE (Local)"],
        index=0,
        help=(
            "**Cohere**: Fast, API-based. Requires COHERE_API_KEY.\n\n"
            "**BGE**: Local BAAI/bge-reranker-large. No API key needed, "
            "but requires ~1.5GB RAM and a first-run model download."
        ),
    )
    reranker = "cohere" if reranker_choice == "Cohere (API)" else "bge"

    st.divider()

    st.subheader("Retrieval Tuning")
    retrieval_k = st.slider(
        "FAISS candidates (k)",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
        help="Number of candidates fetched by FAISS before reranking. "
             "Higher = better recall, slower reranking.",
    )
    top_n = st.slider(
        "Chunks passed to LLM",
        min_value=2,
        max_value=10,
        value=6,
        step=1,
        help="Number of top reranked chunks injected into the prompt. "
             "Higher = more context, more tokens.",
    )

    st.divider()

    st.subheader("Vectorstore")
    vectorstore_dir = st.text_input(
        "Vectorstore path",
        value="./vectorstore",
        help="Path to the FAISS vectorstore built by ingest.py",
    )

    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.pop("messages", None)
        st.session_state.pop("pipeline", None)
        st.rerun()

    st.caption(
        "**Score labels:**\n"
        "- **Rerank score** — normalised [0–1], task relevance. Authoritative.\n"
        "- *Embed score* — raw FAISS cosine distance. Reference only.\n\n"
        "A chunk with embed score 0.88 may rank below one with 0.74 "
        "if the reranker judges it less relevant to the actual question."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Initialisation (cached per session)
# ─────────────────────────────────────────────────────────────────────────────

def load_pipeline(vs_dir: str) -> dict:
    """Loads vectorstore and initialises the pipeline. Cached in session state."""
    vs_path = Path(vs_dir)
    if not vs_path.exists():
        return None

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
    )
    vectorstore = FAISS.load_local(
        str(vs_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return initialise_pipeline(vectorstore)


if "pipeline" not in st.session_state:
    with st.spinner("Loading vectorstore and initialising pipeline..."):
        pipeline = load_pipeline(vectorstore_dir)
        if pipeline is None:
            st.error(
                f"Vectorstore not found at `{vectorstore_dir}`. "
                "Run `python ingest.py` first to build it."
            )
            st.stop()
        st.session_state.pipeline = pipeline

if "messages" not in st.session_state:
    st.session_state.messages = []


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.title("📄 RAG Chatbot")
st.caption(
    f"Reranker: **{reranker_choice}** · "
    f"FAISS k: **{retrieval_k}** · "
    f"Chunks to LLM: **{top_n}**"
)
st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Chat History Display
# ─────────────────────────────────────────────────────────────────────────────

def render_sources(source_docs: list):
    """Renders the source panel for a single response."""
    for doc in source_docs:
        m = doc.metadata

        rerank_score  = m.get("rerank_score_norm", "N/A")
        embed_score   = m.get("embedding_score", "N/A")
        synthesised   = m.get("synthesised", False)
        source        = m.get("source", "Unknown Document")
        page          = m.get("page", None)
        chunk_id      = m.get("chunk_id", "?")
        reranker_used = m.get("reranker", "unknown")

        synth_badge   = "⚠️ **SYNTHESISED**  " if synthesised else ""
        page_str      = f" · Page **{page}**" if page is not None else ""

        st.markdown(
            f"{synth_badge}"
            f"**[Rerank: {rerank_score}]**  "
            f"*Embed: {embed_score} (reference only)*  \n"
            f"📄 `{source}`{page_str} · Chunk {chunk_id} · _{reranker_used} reranker_"
        )
        st.caption(doc.page_content[:350].strip() + "…")
        st.divider()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📄 Retrieved Sources", expanded=False):
                render_sources(msg["sources"])


# ─────────────────────────────────────────────────────────────────────────────
# Query Handler
# ─────────────────────────────────────────────────────────────────────────────

if query := st.chat_input("Ask anything about your document..."):

    # Display user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Run pipeline
    with st.chat_message("assistant"):
        with st.spinner("Retrieving, reranking, and generating..."):
            try:
                result = run_query(
                    query=query,
                    vectorstore=st.session_state.pipeline["vectorstore"],
                    memory=st.session_state.pipeline["memory"],
                    generator_llm=st.session_state.pipeline["generator_llm"],
                    reranker=reranker,
                    retrieval_k=retrieval_k,
                    top_n=top_n,
                )
                answer = result["answer"]
                sources = result["source_documents"]

            except Exception as e:
                answer = f"⚠️ Pipeline error: `{type(e).__name__}: {e}`"
                sources = []

        st.markdown(answer)

        if sources:
            with st.expander("📄 Retrieved Sources", expanded=False):
                render_sources(sources)

    # Save to session history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })

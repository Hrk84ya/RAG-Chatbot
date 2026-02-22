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
import json
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO

load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from src.chain import initialise_pipeline, run_query


# ─────────────────────────────────────────────────────────────────────────────
# Export Functions
# ─────────────────────────────────────────────────────────────────────────────

def export_chat_json():
    """Export chat history as JSON."""
    def serialize_message(msg):
        if msg["role"] == "assistant" and "sources" in msg:
            sources = []
            for doc in msg["sources"]:
                sources.append({
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                })
            return {
                "role": msg["role"],
                "content": msg["content"],
                "sources": sources
            }
        return msg
    return json.dumps([serialize_message(msg) for msg in st.session_state.messages], indent=2)

def export_chat_text():
    """Export chat history as formatted text."""
    lines = []
    for msg in st.session_state.messages:
        role = msg["role"].capitalize()
        content = msg["content"]
        lines.append(f"{role}: {content}")
        if msg["role"] == "assistant" and "sources" in msg:
            lines.append("Sources:")
            for doc in msg["sources"]:
                m = doc.metadata
                rerank_score = m.get("rerank_score_norm", "N/A")
                embed_score = m.get("embedding_score", "N/A")
                source = m.get("source", "Unknown")
                page = m.get("page", "")
                chunk_id = m.get("chunk_id", "")
                lines.append(f"  - {source} (Page {page}, Chunk {chunk_id}) - Rerank: {rerank_score}, Embed: {embed_score}")
                lines.append(f"    {doc.page_content[:200]}...")
            lines.append("")
        lines.append("")
    return "\n".join(lines)

def export_chat_pdf():
    """Export chat history as PDF with proper text wrapping and formatting."""
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_LEFT
    from html import escape
    
    buffer = BytesIO()
    pdf_doc = SimpleDocTemplate(buffer, pagesize=letter, 
                           rightMargin=0.75*inch, leftMargin=0.75*inch,
                           topMargin=0.75*inch, bottomMargin=0.75*inch)
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=12
    )
    
    user_style = ParagraphStyle(
        'UserMessage',
        parent=styles['Normal'],
        fontSize=11,
        leftIndent=0,
        spaceAfter=6,
        fontName='Helvetica-Bold'
    )
    
    assistant_style = ParagraphStyle(
        'AssistantMessage',
        parent=styles['Normal'],
        fontSize=11,
        leftIndent=0,
        spaceAfter=6,
        fontName='Helvetica-Bold'
    )
    
    content_style = ParagraphStyle(
        'Content',
        parent=styles['Normal'],
        fontSize=10,
        leftIndent=20,
        spaceAfter=12,
        alignment=TA_LEFT
    )
    
    source_style = ParagraphStyle(
        'Source',
        parent=styles['Normal'],
        fontSize=8,
        leftIndent=40,
        spaceAfter=6
    )
    
    story = []
    
    # Title
    story.append(Paragraph("Chat History Export", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Check if there are messages
    if not st.session_state.messages:
        story.append(Paragraph("No chat history available.", content_style))
    
    # Chat messages
    for msg in st.session_state.messages:
        role = msg["role"].capitalize()
        content = escape(msg["content"])
        
        # Role header
        if msg["role"] == "user":
            story.append(Paragraph(f"User:", user_style))
        else:
            story.append(Paragraph(f"Assistant:", assistant_style))
        
        # Message content - split into paragraphs for better formatting
        for para in content.split('\n'):
            if para.strip():
                story.append(Paragraph(para, content_style))
        
        # Sources (if assistant message)
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            story.append(Spacer(1, 0.1*inch))
            story.append(Paragraph("<b>Sources:</b>", content_style))
            for doc in msg["sources"]:
                m = doc.metadata
                rerank_score = m.get("rerank_score_norm", "N/A")
                embed_score = m.get("embedding_score", "N/A")
                source = m.get("source", "Unknown")
                page = m.get("page", "")
                chunk_id = m.get("chunk_id", "")
                
                source_text = escape(f"{source} (Page {page}, Chunk {chunk_id}) - Rerank: {rerank_score}, Embed: {embed_score}")
                story.append(Paragraph(source_text, source_style))
                
                # Chunk preview
                chunk_preview = escape(doc.page_content[:200])
                story.append(Paragraph(f"<i>{chunk_preview}...</i>", source_style))
        
        story.append(Spacer(1, 0.2*inch))
    
    pdf_doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


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

# Export Chat History
st.subheader("💾 Export Chat History")
col1, col2, col3 = st.columns(3)
with col1:
    st.download_button(
        label="📄 JSON",
        data=export_chat_json(),
        file_name="chat_history.json",
        mime="application/json",
        use_container_width=True
    )
with col2:
    st.download_button(
        label="📝 Text",
        data=export_chat_text(),
        file_name="chat_history.txt",
        mime="text/plain",
        use_container_width=True
    )
with col3:
    st.download_button(
        label="📖 PDF",
        data=export_chat_pdf(),
        file_name="chat_history.pdf",
        mime="application/pdf",
        use_container_width=True
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

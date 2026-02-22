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
import base64
import tempfile

load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from src.chain import initialise_pipeline, run_query
import time
from datetime import datetime
from collections import Counter
import pandas as pd

# Voice integration imports
try:
    import speech_recognition as sr
    from gtts import gTTS
    from audio_recorder_streamlit import audio_recorder
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    audio_recorder = None
from collections import Counter
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Analytics Functions
# ─────────────────────────────────────────────────────────────────────────────

def init_analytics():
    """Initialize analytics tracking in session state."""
    if "analytics" not in st.session_state:
        st.session_state.analytics = {
            "queries": [],
            "response_times": [],
            "source_usage": [],
            "reranker_usage": {"cohere": 0, "bge": 0},
            "timestamps": [],
            "retrieval_k_values": [],
            "top_n_values": [],
        }

def log_query_analytics(query, response_time, sources, reranker, retrieval_k, top_n):
    """Log analytics data for a query."""
    if "analytics" not in st.session_state:
        init_analytics()
    
    st.session_state.analytics["queries"].append(query)
    st.session_state.analytics["response_times"].append(response_time)
    st.session_state.analytics["timestamps"].append(datetime.now())
    st.session_state.analytics["retrieval_k_values"].append(retrieval_k)
    st.session_state.analytics["top_n_values"].append(top_n)
    st.session_state.analytics["reranker_usage"][reranker] += 1
    
    # Track source usage
    for doc in sources:
        source_name = doc.metadata.get("source", "Unknown")
        st.session_state.analytics["source_usage"].append(source_name)

def get_analytics_summary():
    """Generate analytics summary statistics."""
    if "analytics" not in st.session_state or not st.session_state.analytics["queries"]:
        return None
    
    analytics = st.session_state.analytics
    
    # Calculate statistics
    total_queries = len(analytics["queries"])
    avg_response_time = sum(analytics["response_times"]) / total_queries if total_queries > 0 else 0
    min_response_time = min(analytics["response_times"]) if analytics["response_times"] else 0
    max_response_time = max(analytics["response_times"]) if analytics["response_times"] else 0
    
    # Source usage frequency
    source_counter = Counter(analytics["source_usage"])
    
    # Average retrieval settings
    avg_k = sum(analytics["retrieval_k_values"]) / total_queries if total_queries > 0 else 0
    avg_top_n = sum(analytics["top_n_values"]) / total_queries if total_queries > 0 else 0
    
    return {
        "total_queries": total_queries,
        "avg_response_time": avg_response_time,
        "min_response_time": min_response_time,
        "max_response_time": max_response_time,
        "source_usage": dict(source_counter.most_common()),
        "reranker_usage": analytics["reranker_usage"],
        "avg_k": avg_k,
        "avg_top_n": avg_top_n,
        "timestamps": analytics["timestamps"],
        "response_times": analytics["response_times"],
    }

def render_analytics_dashboard():
    """Render the analytics dashboard."""
    st.title("📊 Analytics Dashboard")
    
    summary = get_analytics_summary()
    
    if summary is None:
        st.info("No analytics data available yet. Start chatting to generate analytics!")
        return
    
    # Overview metrics
    st.subheader("📈 Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Queries", summary["total_queries"])
    with col2:
        st.metric("Avg Response Time", f"{summary['avg_response_time']:.2f}s")
    with col3:
        st.metric("Min Response Time", f"{summary['min_response_time']:.2f}s")
    with col4:
        st.metric("Max Response Time", f"{summary['max_response_time']:.2f}s")
    
    st.divider()
    
    # Response time chart
    st.subheader("⏱️ Response Time Trends")
    if summary["response_times"]:
        df_response = pd.DataFrame({
            "Query #": range(1, len(summary["response_times"]) + 1),
            "Response Time (s)": summary["response_times"]
        })
        st.line_chart(df_response.set_index("Query #"))
    
    st.divider()
    
    # Source usage
    st.subheader("📄 Source Document Usage")
    if summary["source_usage"]:
        df_sources = pd.DataFrame(
            list(summary["source_usage"].items()),
            columns=["Document", "Times Retrieved"]
        ).sort_values("Times Retrieved", ascending=False)
        
        st.bar_chart(df_sources.set_index("Document"))
        
        with st.expander("View Detailed Source Statistics"):
            st.dataframe(df_sources, use_container_width=True)
    else:
        st.info("No source usage data available.")
    
    st.divider()
    
    # Reranker usage
    st.subheader("🔄 Reranker Usage")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Cohere API", summary["reranker_usage"]["cohere"])
    with col2:
        st.metric("BGE Local", summary["reranker_usage"]["bge"])
    
    if summary["reranker_usage"]["cohere"] > 0 or summary["reranker_usage"]["bge"] > 0:
        df_reranker = pd.DataFrame({
            "Reranker": ["Cohere", "BGE"],
            "Usage Count": [summary["reranker_usage"]["cohere"], summary["reranker_usage"]["bge"]]
        })
        st.bar_chart(df_reranker.set_index("Reranker"))
    
    st.divider()
    
    # Retrieval settings
    st.subheader("⚙️ Average Retrieval Settings")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Avg FAISS k", f"{summary['avg_k']:.1f}")
    with col2:
        st.metric("Avg Chunks to LLM", f"{summary['avg_top_n']:.1f}")
    
    st.divider()
    
    # Query history
    st.subheader("📝 Recent Queries")
    if st.session_state.analytics["queries"]:
        recent_queries = list(zip(
            st.session_state.analytics["timestamps"][-10:],
            st.session_state.analytics["queries"][-10:],
            st.session_state.analytics["response_times"][-10:]
        ))
        
        df_queries = pd.DataFrame(
            recent_queries,
            columns=["Timestamp", "Query", "Response Time (s)"]
        )
        df_queries["Timestamp"] = df_queries["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        st.dataframe(df_queries, use_container_width=True)
    
    st.divider()
    
    # Export and reset options
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Export Analytics as JSON", use_container_width=True):
            analytics_json = json.dumps({
                "total_queries": summary["total_queries"],
                "avg_response_time": summary["avg_response_time"],
                "min_response_time": summary["min_response_time"],
                "max_response_time": summary["max_response_time"],
                "source_usage": summary["source_usage"],
                "reranker_usage": summary["reranker_usage"],
                "queries": st.session_state.analytics["queries"],
                "response_times": st.session_state.analytics["response_times"],
            }, indent=2)
            st.download_button(
                "Download Analytics JSON",
                analytics_json,
                "analytics.json",
                "application/json",
                use_container_width=True
            )
    
    with col2:
        if st.button("🗑️ Reset Analytics", use_container_width=True):
            st.session_state.pop("analytics", None)
            init_analytics()
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Voice Integration Functions
# ─────────────────────────────────────────────────────────────────────────────

def text_to_speech(text, tld='co.uk'):
    """Convert text to speech and return audio file.
    
    Args:
        text: Text to convert to speech
        tld: Top-level domain for accent (co.uk for British, com for American, etc.)
    """
    if not VOICE_AVAILABLE:
        return None
    
    try:
        # Create a temporary file for the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            # Use British English accent with tld='co.uk'
            tts = gTTS(text=text, lang='en', tld=tld, slow=False)
            tts.save(fp.name)
            
            # Read the audio file
            with open(fp.name, 'rb') as audio_file:
                audio_bytes = audio_file.read()
            
            # Clean up temp file
            os.unlink(fp.name)
            
            return audio_bytes
    except Exception as e:
        st.error(f"Text-to-speech error: {e}")
        return None

def create_audio_player(audio_bytes):
    """Create an HTML audio player for the given audio bytes."""
    if audio_bytes:
        audio_base64 = base64.b64encode(audio_bytes).decode()
        audio_html = f"""
        <audio controls autoplay style="width: 100%;">
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            Your browser does not support the audio element.
        </audio>
        """
        return audio_html
    return None

def speech_to_text_from_file(audio_file):
    """Convert uploaded audio file to text using speech recognition."""
    if not VOICE_AVAILABLE:
        return None
    
    try:
        recognizer = sr.Recognizer()
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as fp:
            fp.write(audio_file.read())
            temp_path = fp.name
        
        # Convert speech to text
        with sr.AudioFile(temp_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        
        # Clean up
        os.unlink(temp_path)
        
        return text
    except sr.UnknownValueError:
        st.error("Could not understand audio")
        return None
    except sr.RequestError as e:
        st.error(f"Speech recognition error: {e}")
        return None
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

def speech_to_text_from_bytes(audio_bytes):
    """Convert audio bytes to text using speech recognition with improved preprocessing."""
    if not VOICE_AVAILABLE or not audio_bytes:
        return None
    
    try:
        recognizer = sr.Recognizer()
        
        # Adjust recognizer settings for better accuracy
        recognizer.energy_threshold = 300  # Minimum audio energy to consider for recording
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.8  # Seconds of non-speaking audio before phrase is complete
        
        # Save audio bytes temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as fp:
            fp.write(audio_bytes)
            temp_path = fp.name
        
        # Convert speech to text
        with sr.AudioFile(temp_path) as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            # Record the audio
            audio_data = recognizer.record(source)
            
            # Try Google Speech Recognition
            try:
                text = recognizer.recognize_google(audio_data, language='en-US', show_all=False)
                os.unlink(temp_path)
                return text
            except sr.UnknownValueError:
                # Try with UK English
                try:
                    text = recognizer.recognize_google(audio_data, language='en-GB', show_all=False)
                    os.unlink(temp_path)
                    return text
                except sr.UnknownValueError:
                    os.unlink(temp_path)
                    return "Could not understand audio. Please speak clearly and try again."
        
    except sr.RequestError as e:
        return f"Speech recognition service error: {e}. Please check your internet connection."
    except Exception as e:
        return f"Error processing audio: {str(e)}"


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

# Initialize analytics
init_analytics()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Configuration
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")
    
    # Navigation
    st.subheader("📍 Navigation")
    page = st.radio(
        "Go to",
        ["💬 Chat", "📊 Analytics"],
        index=0,
        key="nav_radio"
    )
    
    st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Analytics Page
# ─────────────────────────────────────────────────────────────────────────────

if page == "📊 Analytics":
    render_analytics_dashboard()
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Chat Page - Sidebar Configuration Continued
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    # Define vectorstore_dir early so it can be used in upload section
    vectorstore_dir_default = "./vectorstore"


    st.subheader("📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload documents to add to knowledge base",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Upload PDF, DOCX, or TXT files. They will be automatically processed and added to the vectorstore."
    )
    
    if uploaded_files:
        if st.button("🔄 Process Uploaded Files", use_container_width=True):
            with st.spinner("Processing uploaded documents..."):
                try:
                    from pathlib import Path
                    from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
                    from langchain_text_splitters import RecursiveCharacterTextSplitter
                    
                    # Save uploaded files temporarily
                    docs_dir = Path("./docs")
                    docs_dir.mkdir(exist_ok=True)
                    
                    saved_files = []
                    for uploaded_file in uploaded_files:
                        file_path = docs_dir / uploaded_file.name
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        saved_files.append(file_path)
                    
                    # Load and process documents
                    LOADER_MAP = {
                        ".pdf": PyPDFLoader,
                        ".docx": Docx2txtLoader,
                        ".txt": TextLoader,
                    }
                    
                    all_docs = []
                    for file_path in saved_files:
                        ext = file_path.suffix.lower()
                        loader_cls = LOADER_MAP[ext]
                        loader = loader_cls(str(file_path))
                        docs = loader.load()
                        
                        for doc in docs:
                            doc.metadata["source"] = file_path.name
                            doc.metadata["file_path"] = str(file_path)
                        
                        all_docs.extend(docs)
                    
                    # Chunk documents
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=512,
                        chunk_overlap=64,
                        length_function=len,
                        separators=["\n\n", "\n", ". ", " ", ""],
                    )
                    chunks = splitter.split_documents(all_docs)
                    
                    for i, chunk in enumerate(chunks):
                        chunk.metadata["chunk_id"] = i
                    
                    # Add to existing vectorstore or create new one
                    embeddings = OpenAIEmbeddings(
                        model="text-embedding-3-small",
                        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
                    )
                    
                    vectorstore_path = Path(vectorstore_dir_default)
                    if vectorstore_path.exists():
                        # Load existing and merge
                        existing_vs = FAISS.load_local(
                            str(vectorstore_path),
                            embeddings,
                            allow_dangerous_deserialization=True,
                        )
                        new_vs = FAISS.from_documents(chunks, embeddings)
                        existing_vs.merge_from(new_vs)
                        existing_vs.save_local(str(vectorstore_path))
                    else:
                        # Create new vectorstore
                        vectorstore_path.mkdir(parents=True, exist_ok=True)
                        new_vs = FAISS.from_documents(chunks, embeddings)
                        new_vs.save_local(str(vectorstore_path))
                    
                    st.success(f"✅ Successfully processed {len(uploaded_files)} file(s) and added {len(chunks)} chunks to the vectorstore!")
                    st.info("🔄 Reloading vectorstore... Please refresh the page to use the updated knowledge base.")
                    
                    # Clear pipeline to force reload
                    st.session_state.pop("pipeline", None)
                    
                except Exception as e:
                    st.error(f"❌ Error processing files: {type(e).__name__}: {e}")
    
    st.divider()

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
        value=vectorstore_dir_default,
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

# Voice Controls
if VOICE_AVAILABLE:
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader("🎤 Voice Input")
    
    with col2:
        enable_tts = st.checkbox(
            "🔊 TTS",
            value=st.session_state.get("enable_tts", False),
            help="Enable Text-to-Speech for responses"
        )
        st.session_state.enable_tts = enable_tts
    
    with col3:
        voice_accent = st.selectbox(
            "Voice",
            options=["British 🇬🇧", "American 🇺🇸", "Australian 🇦🇺", "Indian 🇮🇳"],
            index=0,
            help="Select voice accent"
        )
        # Map accent to TLD
        accent_map = {
            "British 🇬🇧": "co.uk",
            "American 🇺🇸": "com",
            "Australian 🇦🇺": "com.au",
            "Indian 🇮🇳": "co.in"
        }
        st.session_state.voice_tld = accent_map[voice_accent]
    
    # Audio recorder
    st.markdown("**Click the microphone to record your question:**")
    
    with st.expander("💡 Tips for better voice recognition", expanded=False):
        st.markdown("""
        - Speak clearly and at a moderate pace
        - Reduce background noise
        - Hold the microphone/device at a consistent distance
        - Record for 2-5 seconds minimum
        - Ensure good internet connection (uses Google Speech API)
        - Try speaking in a quiet environment
        """)
    
    audio_bytes = audio_recorder(
        text="",
        recording_color="#e74c3c",
        neutral_color="#3498db",
        icon_name="microphone",
        icon_size="2x",
    )
    
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("🎤 Convert to Text and Send", use_container_width=True, type="primary"):
                with st.spinner("Converting speech to text..."):
                    transcribed_text = speech_to_text_from_bytes(audio_bytes)
                    if transcribed_text and not transcribed_text.startswith("Could not") and not transcribed_text.startswith("Speech recognition") and not transcribed_text.startswith("Error"):
                        st.session_state.voice_input = transcribed_text
                        st.success(f"✅ Transcribed: {transcribed_text}")
                        st.rerun()
                    else:
                        st.error(transcribed_text if transcribed_text else "Failed to transcribe audio")
        
        with col_b:
            if st.button("🔍 Test Transcription Only", use_container_width=True):
                with st.spinner("Testing transcription..."):
                    transcribed_text = speech_to_text_from_bytes(audio_bytes)
                    if transcribed_text and not transcribed_text.startswith("Could not") and not transcribed_text.startswith("Speech recognition") and not transcribed_text.startswith("Error"):
                        st.info(f"📝 Transcribed text: **{transcribed_text}**")
                    else:
                        st.error(transcribed_text if transcribed_text else "Failed to transcribe audio")

else:
    st.warning("⚠️ Voice features unavailable. Install required packages: `pip install SpeechRecognition gTTS audio-recorder-streamlit`")

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
        
        # Add TTS playback for assistant messages
        if msg["role"] == "assistant" and VOICE_AVAILABLE and st.session_state.get("enable_tts", False):
            if st.button(f"🔊 Play Audio", key=f"tts_{st.session_state.messages.index(msg)}"):
                with st.spinner("Generating audio..."):
                    tld = st.session_state.get("voice_tld", "co.uk")
                    audio_bytes = text_to_speech(msg["content"], tld=tld)
                    if audio_bytes:
                        audio_html = create_audio_player(audio_bytes)
                        st.markdown(audio_html, unsafe_allow_html=True)
        
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📄 Retrieved Sources", expanded=False):
                render_sources(msg["sources"])


# ─────────────────────────────────────────────────────────────────────────────
# Query Handler
# ─────────────────────────────────────────────────────────────────────────────

# Check for voice input
if "voice_input" in st.session_state and st.session_state.voice_input:
    query = st.session_state.voice_input
    st.session_state.pop("voice_input")
else:
    query = st.chat_input("Ask anything about your document...")

if query:

    # Display user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Run pipeline
    with st.chat_message("assistant"):
        with st.spinner("Retrieving, reranking, and generating..."):
            start_time = time.time()
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
                response_time = time.time() - start_time
                
                # Log analytics
                log_query_analytics(query, response_time, sources, reranker, retrieval_k, top_n)

            except Exception as e:
                answer = f"⚠️ Pipeline error: `{type(e).__name__}: {e}`"
                sources = []
                response_time = time.time() - start_time

        st.markdown(answer)
        
        # Auto-play TTS if enabled
        if VOICE_AVAILABLE and st.session_state.get("enable_tts", False):
            with st.spinner("Generating audio..."):
                tld = st.session_state.get("voice_tld", "co.uk")
                audio_bytes = text_to_speech(answer, tld=tld)
                if audio_bytes:
                    audio_html = create_audio_player(audio_bytes)
                    st.markdown(audio_html, unsafe_allow_html=True)

        if sources:
            with st.expander("📄 Retrieved Sources", expanded=False):
                render_sources(sources)

    # Save to session history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })

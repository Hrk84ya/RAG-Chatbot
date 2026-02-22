# 📄 RAG Chatbot

A production-hardened Retrieval-Augmented Generation chatbot with advanced features:

- **Dual reranker support**: Cohere API or BGE local model, switchable live in the UI
- **Reranker-conditional score normalisation**: prevents rank collapse with BGE logits
- **Stable synthesis detection**: deterministic span hashing, not LLM self-assessment
- **Lost-in-the-middle mitigation**: 1,N,2,N-1 interleaved chunk ordering
- **Decoupled summariser**: prevents memory drift over long conversations
- **Prompt injection defence**: retrieved content treated as untrusted data
- **Voice integration**: Speech-to-text input and text-to-speech output with multiple accents
- **Analytics dashboard**: Track queries, response times, source usage, and retrieval patterns
- **Document upload**: Add documents directly through the UI without manual file management
- **Export capabilities**: Export chat history as JSON, text, or formatted PDF

---

## Project Structure

```
rag_chatbot/
├── app.py              # Streamlit UI
├── ingest.py           # Document ingestion → FAISS vectorstore
├── requirements.txt
├── .env.example        # Copy to .env and fill in API keys
├── docs/               # ← Put your documents here
├── vectorstore/        # ← Auto-created by ingest.py
└── src/
    ├── __init__.py
    ├── chain.py        # Full pipeline (8 stages)
    └── prompts.py      # System prompt + human message template
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on BGE:** If using the BGE reranker, `sentence-transformers` will
> download `BAAI/bge-reranker-large` (~1.5GB) on first run. This is automatic.

> **Note on Voice Features:** Voice input/output requires `SpeechRecognition`, `gTTS`, and `audio-recorder-streamlit`. These are optional - the app will run without them if not installed.

### 2. Configure API keys

```bash
cp .env.example .env
```

Edit `.env`:
```
OPENAI_API_KEY=sk-...          # Required for embeddings + generation
COHERE_API_KEY=...             # Required only if using Cohere reranker
```

### 3. Add your documents

You have two options:

**Option A: Manual file placement**
```bash
mkdir docs
cp your_paper.pdf docs/
cp your_manual.docx docs/
```

**Option B: Upload through UI** (after launching the app)
- Use the "Upload Documents" section in the sidebar
- Supports PDF, DOCX, and TXT files
- Automatically processes and adds to vectorstore

### 4. Run ingestion (if using Option A)

```bash
python ingest.py
```

With custom paths or chunk settings:
```bash
python ingest.py --input_dir ./my_docs --output_dir ./vectorstore --chunk_size 256 --chunk_overlap 32
```

### 5. Launch the chatbot

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

---

## Key Features

### Voice Integration

The chatbot supports voice input and output for hands-free interaction:

- **Speech-to-Text**: Record questions using your microphone
- **Text-to-Speech**: Hear responses read aloud with multiple accent options (British, American, Australian, Indian)
- **Auto-play**: Optionally enable automatic audio playback for all responses
- Uses Google Speech Recognition API for transcription
- Powered by gTTS (Google Text-to-Speech) for audio generation

### Analytics Dashboard

Track and analyze your chatbot usage:

- **Query metrics**: Total queries, average/min/max response times
- **Response time trends**: Visualize performance over time
- **Source usage**: See which documents are retrieved most frequently
- **Reranker statistics**: Compare Cohere vs BGE usage
- **Retrieval settings**: Monitor average k and top_n values
- **Export analytics**: Download usage data as JSON

### Export Chat History

Save your conversations in multiple formats:

- **JSON**: Structured data with full metadata and sources
- **Text**: Plain text format for easy reading
- **PDF**: Professionally formatted document with proper text wrapping

### Document Upload

Add documents without restarting the app:

- Upload PDF, DOCX, or TXT files directly through the UI
- Automatic chunking and embedding
- Seamlessly merges with existing vectorstore
- No need to manually run `ingest.py`

---

## Reranker Comparison

| | Cohere | BGE (Local) |
|---|---|---|
| **Type** | API (cloud) | Local model |
| **Model** | rerank-english-v3.0 | BAAI/bge-reranker-large |
| **API key needed** | ✅ COHERE_API_KEY | ❌ None |
| **Latency** | ~200–400ms | ~500ms–2s (CPU) |
| **Cost** | ~$0.001/1K docs | Free |
| **Score range** | [-1.0, 1.0] | Unbounded logits |
| **Normalisation** | Linear: `(raw+1)/2` | Sigmoid: `1/(1+e^-raw)` |

> Reranker can be switched live in the sidebar — no restart needed.

---

## Determinism & Reproducibility

For audit logging and offline evaluation, ensure the following are held constant across runs:

- Identical document set under `./docs/`
- Same chunking parameters: `--chunk_size` and `--chunk_overlap`
- Same embedding model version (`text-embedding-3-small` by default)
- Same reranker selection (Cohere or BGE)
- Same FAISS index — re-run `ingest.py` if any of the above change

Generation is seeded (`seed=42`, `temperature=0`) on both the generator and summariser LLMs, which reduces but does not fully eliminate output variance (OpenAI does not guarantee exact reproducibility across API versions).

> **Note:** FAISS indices are not portable across embedding model changes.
> Re-ingest all documents after upgrading the embedding model.

---

## ⚠️ FAISS Index Compatibility

The stored FAISS index is tightly coupled to the embedding model used during ingestion — specifically its output dimensionality and vector space geometry.

If you change any of the following, you **must** re-run `python ingest.py`:

- Embedding model (e.g. `text-embedding-3-small` → `text-embedding-3-large`)
- Embedding dimensionality
- Chunking parameters (`--chunk_size`, `--chunk_overlap`)

Loading a FAISS index built with a different embedding model **will not raise an error** — the index will load and return results, but those results will be geometrically meaningless. Retrieval quality will silently collapse with no warning.

As a safeguard, consider storing the embedding model name alongside the vectorstore:

```bash
echo "text-embedding-3-small" > ./vectorstore/embedding_model.txt
```

And validating it at startup in `app.py` before loading the index.

---

## Runtime Notes

- **BGE reranker** runs locally via `sentence-transformers`. No API key required.
  - CPU latency: ~500ms–2s per query (depending on hardware and `top_n`)
  - GPU latency: ~150–300ms (recommended for workloads above ~5 QPS)
  - First run downloads `BAAI/bge-reranker-large` (~1.5GB) automatically

- **Cohere reranker** is network-bound but supports request batching and has predictable sub-500ms latency at low QPS.

- **For production workloads:**
  - Use **Cohere** for low-latency cloud deployments where network egress is acceptable
  - Use **BGE + GPU** for on-premises or cost-sensitive environments where API costs or data residency are constraints
  - Both rerankers produce directly comparable normalised `[0–1]` scores in the prompt — switching rerankers does not require any prompt or pipeline changes

---

## Tuning Reference

| Behaviour | Where to change |
|---|---|
| Retrieve more candidates | `k` slider in sidebar (or `retrieval_k` in `run_query()`) |
| Fewer chunks in prompt | `top_n` slider in sidebar (or `top_n` in `run_query()`) |
| Larger chunks | `--chunk_size` flag in `ingest.py` |
| More boundary context | `--chunk_overlap` flag in `ingest.py` |
| Tighter synthesis detection | `[:128]` → `[:64]` in `stable_fingerprint()` |
| Cheaper summariser | Change `gpt-4o-mini` → `gpt-3.5-turbo` in `build_summariser_llm()` |
| Longer memory window | Increase `max_token_limit` in `build_memory()` |
| Disable interleaving | Comment out `reranked = interleave_extremes(reranked)` in `run_query()` |
| Change voice accent | Select from dropdown in UI (British, American, Australian, Indian) |
| Enable/disable TTS | Toggle "TTS" checkbox in UI |

---

## UI Navigation

The app has two main pages accessible via the sidebar:

1. **💬 Chat**: Main conversation interface with document Q&A
2. **📊 Analytics**: Dashboard showing usage statistics and trends

---

## How Scores Work

Two scores are tracked per chunk:

- **Rerank score** `[0.00–1.00]` — Normalised task relevance from the cross-encoder. This is what the LLM uses for conflict resolution (Rule C2). **Authoritative.**
- **Embed score** — Raw FAISS cosine similarity. Stored for UI display only. Never shown to the LLM.

A chunk with embed score `0.88` may rank *below* one with `0.74` if the reranker judges it less relevant to the specific question asked. This is expected and correct behaviour.

---

## Bug Fixes

| Bug | Fix |
|---|---|
| BGE logits all collapse to `[Score: 1.00]` | Sigmoid normalisation for BGE; linear for Cohere |
| `hash()` non-deterministic across process restarts | `stable_fingerprint()` with CRLF normalisation + strip + explicit UTF-8 |
| LLM unreliably self-reports synthesis | Deterministic `span_hash` tagging by pipeline |
| Memory drift over long conversations | Decoupled `gpt-4o-mini` summariser |
| Mid-ranked chunks ignored (lost-in-middle) | Interleaved 1,N,2,N-1 ordering |

---

## Architecture

The RAG pipeline consists of 8 stages:

1. **Dense retrieval**: FAISS top-k candidates with cosine similarity
2. **Cross-encoder rerank**: Cohere API or BGE local model
3. **Score normalisation**: Reranker-specific formulas (linear for Cohere, sigmoid for BGE)
4. **Span hash tagging**: Deterministic synthesis detection via MD5 fingerprints
5. **Interleaved ordering**: 1,N,2,N-1 pattern to combat attention decay
6. **Context formatting**: Scored chunks with metadata for LLM
7. **Generation**: GPT-4o with temperature=0 and seed=42 for reproducibility
8. **Memory save**: Compressed by dedicated GPT-4o-mini summariser

### Models Used

- **Embeddings**: `text-embedding-3-small` (OpenAI)
- **Generator**: `gpt-4o` (OpenAI)
- **Summariser**: `gpt-4o-mini` (OpenAI) - decoupled to prevent memory drift
- **Reranker (Cohere)**: `rerank-english-v3.0`
- **Reranker (BGE)**: `BAAI/bge-reranker-large` (local)

---

## Dependencies

Core dependencies:
- `openai>=1.0.0` - OpenAI API client
- `langchain>=0.1.0,<0.2.0` - LLM framework
- `faiss-cpu>=1.7.4` - Vector similarity search
- `sentence-transformers>=2.2.2` - BGE reranker
- `cohere>=4.0.0` - Cohere reranker API
- `streamlit>=1.32.0` - Web UI framework

Optional voice features:
- `SpeechRecognition>=3.10.0` - Speech-to-text
- `gTTS>=2.4.0` - Text-to-speech
- `audio-recorder-streamlit>=0.0.8` - Audio recording widget

Document processing:
- `pypdf>=3.0.0` - PDF parsing
- `python-docx>=1.1.0` - DOCX parsing

Export and analytics:
- `reportlab>=4.0.0` - PDF generation
- `pandas>=2.0.0` - Data analysis

---

## License

See [LICENSE](LICENSE) file for details.

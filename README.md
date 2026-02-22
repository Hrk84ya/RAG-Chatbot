# 📄 RAG Chatbot

A production-hardened Retrieval-Augmented Generation chatbot with:
- **Dual reranker support**: Cohere API or BGE local model, switchable live in the UI
- **Reranker-conditional score normalisation**: prevents rank collapse with BGE logits
- **Stable synthesis detection**: deterministic span hashing, not LLM self-assessment
- **Lost-in-the-middle mitigation**: 1,N,2,N-1 interleaved chunk ordering
- **Decoupled summariser**: prevents memory drift over long conversations
- **Prompt injection defence**: retrieved content treated as untrusted data

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

Drop any `.pdf`, `.docx`, or `.txt` files into the `./docs/` folder:
```bash
mkdir docs
cp your_paper.pdf docs/
cp your_manual.docx docs/
```

### 4. Run ingestion

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

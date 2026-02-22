"""
chain.py — Full RAG pipeline.

Stages:
  1. Dense retrieval    — FAISS top-20, embedding scores stored for UI only
  2. Cross-encoder rerank — Cohere API or BGE local model (configurable)
  3. Score normalisation  — Cohere: linear (raw+1)/2 | BGE: sigmoid
  4. Span hash tagging    — stable_fingerprint() → [⚠ SYNTHESISED] flag
  5. Interleaved ordering — 1,N,2,N-1 pattern counteracts lost-in-the-middle
  6. Context formatting   — Only normalised rerank score exposed to LLM
  7. Generation           — gpt-4o, temperature=0, seed=42
  8. Memory save          — Compressed by dedicated gpt-4o-mini summariser
"""

import os
import math
import hashlib

from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory

from src.prompts import build_chat_prompt

chat_prompt = build_chat_prompt()


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Dense Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_with_scores(vectorstore: FAISS, query: str, k: int = 20) -> list:
    """
    Over-fetches top-k candidates using FAISS cosine similarity.
    Raw embedding scores are stored in metadata for UI display ONLY —
    never surfaced to the LLM. The reranker's normalised score is what
    the LLM sees.
    """
    results = vectorstore.similarity_search_with_score(query, k=k)
    docs = []
    for doc, score in results:
        doc.metadata["embedding_score"] = round(float(score), 4)
        docs.append(doc)
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Cross-Encoder Reranking
# ─────────────────────────────────────────────────────────────────────────────

def rerank_with_cohere(query: str, docs: list, top_n: int = 6) -> list:
    """
    Reranks candidates using Cohere's rerank-english-v3.0 cross-encoder.
    Returns bounded relevance scores in [-1.0, 1.0].
    Requires COHERE_API_KEY environment variable.
    """
    import cohere
    co = cohere.Client(os.environ["COHERE_API_KEY"])
    texts = [doc.page_content for doc in docs]
    response = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=texts,
        top_n=top_n,
    )
    reranked = []
    for result in response.results:
        doc = docs[result.index]
        doc.metadata["_raw_rerank_score"] = result.relevance_score
        doc.metadata["reranker"] = "cohere"
        reranked.append(doc)
    return reranked


def rerank_with_bge(query: str, docs: list, top_n: int = 6) -> list:
    """
    Reranks candidates using BAAI/bge-reranker-large (local CrossEncoder).
    Returns unbounded logits — typically [-10, +10]. Sigmoid normalisation
    is applied in Stage 3 to map these to (0, 1).
    """
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder("BAAI/bge-reranker-large")
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    reranked = []
    for doc, score in scored[:top_n]:
        doc.metadata["_raw_rerank_score"] = float(score)
        doc.metadata["reranker"] = "bge"
        reranked.append(doc)
    return reranked


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Score Normalisation
# ─────────────────────────────────────────────────────────────────────────────
#
# CRITICAL: Do NOT use a single formula for both rerankers.
#   • Cohere: relevance_score is bounded [-1.0, 1.0]  → linear map (raw+1)/2
#   • BGE:    logits are unbounded [~-10, +10]         → sigmoid 1/(1+e^-raw)
#
# Applying the Cohere formula to BGE logits causes all scores > 1.0 to
# clamp to [Score: 1.00], silently destroying rank separability (Rule C2).

def normalise_rerank_scores(docs: list, reranker: str) -> list:
    """
    Normalises raw reranker scores to a unified [0.0, 1.0] scale.

    Args:
        docs:      Documents with `_raw_rerank_score` in metadata.
        reranker:  "cohere" | "bge" | "other"
                   Must be passed explicitly — never inferred at runtime.

    Stores result as `rerank_score_norm` — the ONLY score the LLM sees.
    """
    for doc in docs:
        raw = doc.metadata.get("_raw_rerank_score", 0.0)

        if reranker == "cohere":
            norm = (raw + 1) / 2                      # bounded [-1,1] → [0,1]
        elif reranker == "bge":
            norm = 1 / (1 + math.exp(-raw))           # logits → smooth (0,1)
        else:
            norm = raw
            doc.metadata["_norm_warning"] = (
                "reranker type unknown — score passed through unnormalised"
            )

        doc.metadata["rerank_score_norm"] = round(
            max(0.0, min(1.0, norm)), 4               # clamp for float safety
        )

    return docs


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — Span Hash Tagging
# ─────────────────────────────────────────────────────────────────────────────
#
# Rather than asking the LLM to self-report when it synthesises across chunks
# (unreliable), the pipeline deterministically computes a stable content
# fingerprint per chunk. If >1 unique fingerprint is found, [⚠ SYNTHESISED]
# is injected into the context header before the LLM sees it.
#
# stable_fingerprint() canonicalises text before hashing to ensure the same
# logical passage always produces the same hash regardless of OS line endings,
# FAISS loader whitespace differences, or UTF-8 encoding variations.

def stable_fingerprint(text: str) -> str:
    """
    Returns a deterministic MD5 fingerprint of a text passage, stable
    across OS line endings, whitespace variations, and FAISS loader
    encoding differences.

    Canonicalisation (order matters):
        1. CRLF → LF       (Windows loader compatibility)
        2. strip()          (remove loader-introduced whitespace)
        3. encode("utf-8")  (explicit encoding, no locale dependency)
        4. [:128]           (fingerprint first 128 canonical bytes)
    """
    canonical = (
        text
        .replace("\r\n", "\n")
        .strip()
        .encode("utf-8")
    )
    return hashlib.md5(canonical[:128]).hexdigest()


def tag_synthesis(docs: list) -> list:
    """
    Assigns a stable `span_hash` to each doc. If >1 unique hash exists
    across the retrieved set, marks all docs as `synthesised=True`,
    which triggers [⚠ SYNTHESISED] injection in the context formatter.
    """
    for doc in docs:
        doc.metadata["span_hash"] = stable_fingerprint(doc.page_content)

    unique_hashes = {doc.metadata["span_hash"] for doc in docs}
    is_synthesised = len(unique_hashes) > 1

    for doc in docs:
        doc.metadata["synthesised"] = is_synthesised

    return docs


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — Interleaved Ordering
# ─────────────────────────────────────────────────────────────────────────────
#
# Transformers exhibit a lost-in-the-middle effect: chunks at positions 1 and N
# receive the highest attention; positions 3-4 in a 6-chunk window receive the
# least. Passing chunks in simple descending score order means the 3rd and 4th
# most relevant chunks are systematically underutilised.
#
# Fix: Interleave in a 1,N,2,N-1,3,N-2 pattern so that high-scoring chunks
# occupy both the first AND last positions in the prompt context window.

def interleave_extremes(docs: list) -> list:
    """
    Reorders docs in a 1,N,2,N-1,3,N-2 pattern.

    Example (6 chunks ranked A–F by score):
        Input:  [A(1), B(2), C(3), D(4), E(5), F(6)]
        Output: [A(1), F(6), B(2), E(5), C(3), D(4)]
    """
    if len(docs) <= 2:
        return docs

    left, right = 0, len(docs) - 1
    interleaved = []
    while left <= right:
        interleaved.append(docs[left])
        left += 1
        if left <= right:
            interleaved.append(docs[right])
            right -= 1

    return interleaved


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6 — Context Formatter
# ─────────────────────────────────────────────────────────────────────────────
#
# Only `rerank_score_norm` is included in the LLM-visible prompt.
# `embedding_score` is available in metadata for the Streamlit UI only.

def format_context(docs: list) -> str:
    """
    Formats normalised, interleaved, hash-tagged docs into a scored
    context string for injection into the LLM prompt.

    Output example:
        ─────────────────────────────────────────────────────
        [Score: 0.94] [⚠ SYNTHESISED] guide.pdf | Page 12 | Chunk: 42
        Batch size should be set to 32 for standard single-GPU training...
    """
    blocks = []
    for i, doc in enumerate(docs):
        m = doc.metadata

        score      = m.get("rerank_score_norm", "N/A")
        synth_tag  = "[⚠ SYNTHESISED] " if m.get("synthesised", False) else ""
        source     = m.get("source", m.get("file_path", "Unknown Document"))
        page       = f"Page {m['page']}" if "page" in m else ""
        section    = f"Section: \"{m['section']}\"" if "section" in m else ""
        chunk_id   = f"Chunk: {m.get('chunk_id', i)}"

        meta_parts = [p for p in [source, page, section, chunk_id] if p]
        header = f"[Score: {score}] {synth_tag}" + " | ".join(meta_parts)

        blocks.append(
            f"{'─' * 53}\n"
            f"{header}\n"
            f"{doc.page_content.strip()}\n"
        )

    return "\n".join(blocks)


# ─────────────────────────────────────────────────────────────────────────────
# LLMs — Generator and Summariser (Decoupled)
# ─────────────────────────────────────────────────────────────────────────────
#
# Using the same model for both generation and memory summarisation causes
# referent drift over ~5–7 compression cycles ("loss function" →
# "optimisation method" → "training setup"). Fix: dedicated gpt-4o-mini
# at temperature=0 for summarisation only — faithful compression, no
# creative reinterpretation.

def build_generator_llm() -> ChatOpenAI:
    """Main response generator. Full capability, strict factual mode."""
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        seed=42,
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )


def build_summariser_llm() -> ChatOpenAI:
    """
    Dedicated memory summariser — decoupled from generator.
    gpt-4o-mini: faithful compression, minimal reinterpretation,
    ~15× cheaper per token than gpt-4o.
    """
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        seed=42,
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Memory
# ─────────────────────────────────────────────────────────────────────────────

def build_memory(summariser_llm: ChatOpenAI) -> ConversationSummaryBufferMemory:
    """
    Rolling summary memory with a dedicated low-drift summariser.
    Referents are preserved beyond the token window via compression,
    without the reinterpretation drift of using the generator model.
    """
    return ConversationSummaryBufferMemory(
        llm=summariser_llm,
        max_token_limit=1000,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Full Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_query(
    query: str,
    vectorstore: FAISS,
    memory: ConversationSummaryBufferMemory,
    generator_llm: ChatOpenAI,
    reranker: str = "cohere",   # "cohere" | "bge"
    retrieval_k: int = 20,
    top_n: int = 6,
) -> dict:
    """
    Full pipeline: retrieve → rerank → normalise → tag → interleave
                       → format → generate → save memory

    Args:
        query:          User's question.
        vectorstore:    Loaded FAISS vectorstore.
        memory:         ConversationSummaryBufferMemory instance.
        generator_llm:  ChatOpenAI generator (gpt-4o).
        reranker:       "cohere" or "bge" — selects reranker and
                        normalisation formula. Must be explicit.
        retrieval_k:    Number of FAISS candidates to over-fetch.
        top_n:          Final chunks after reranking (passed to LLM).

    Returns:
        {
            "answer":            str,
            "source_documents":  list[Document],   # full metadata
            "formatted_context": str               # LLM-visible context
        }
    """
    if reranker not in ("cohere", "bge"):
        raise ValueError(f"reranker must be 'cohere' or 'bge', got: {reranker!r}")

    # Stage 1: Dense retrieval
    candidates = retrieve_with_scores(vectorstore, query, k=retrieval_k)

    # Stage 2: Rerank
    if reranker == "cohere":
        reranked = rerank_with_cohere(query, candidates, top_n=top_n)
    else:
        reranked = rerank_with_bge(query, candidates, top_n=top_n)

    # Stage 3: Normalise scores (reranker-conditional formula)
    reranked = normalise_rerank_scores(reranked, reranker)

    # Stage 4: Tag synthesis deterministically via stable span hashes
    reranked = tag_synthesis(reranked)

    # Stage 5: Interleave to counteract lost-in-the-middle attention decay
    reranked = interleave_extremes(reranked)

    # Stage 6: Format context (normalised scores only — embedding scores excluded)
    formatted_context = format_context(reranked)

    # Stage 7: Build prompt and generate
    chat_history = memory.load_memory_variables({}).get("chat_history", "")
    prompt = chat_prompt.format_messages(
        chat_history=chat_history,
        context=formatted_context,
        question=query,
    )
    response = generator_llm(prompt)
    answer = response.content

    # Stage 8: Compress and save to summary memory
    memory.save_context({"input": query}, {"answer": answer})

    return {
        "answer": answer,
        "source_documents": reranked,
        "formatted_context": formatted_context,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Initialisation
# ─────────────────────────────────────────────────────────────────────────────

def initialise_pipeline(vectorstore: FAISS) -> dict:
    """
    Call once at app startup. Builds two separate LLM instances and memory.
    Returns a pipeline dict for use with run_query().
    """
    generator_llm  = build_generator_llm()
    summariser_llm = build_summariser_llm()
    memory         = build_memory(summariser_llm)

    return {
        "generator_llm":  generator_llm,
        "summariser_llm": summariser_llm,
        "memory":         memory,
        "vectorstore":    vectorstore,
    }

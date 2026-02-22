"""
ingest.py — Document ingestion pipeline for the RAG chatbot.

Supports: PDF (.pdf), Word (.docx), plain text (.txt)

Usage:
    python ingest.py --input_dir ./docs --output_dir ./vectorstore

What it does:
    1. Loads all supported documents from --input_dir (recursively)
    2. Splits each document into overlapping chunks
    3. Embeds chunks using OpenAI text-embedding-3-small
    4. Saves a FAISS vectorstore to --output_dir

Notes:
    - OPENAI_API_KEY must be set in environment or .env file
    - Re-run to rebuild the vectorstore after adding new documents
    - chunk_size and chunk_overlap can be tuned via CLI flags
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


# ─────────────────────────────────────────────────────────────────────────────
# Supported loaders by file extension
# ─────────────────────────────────────────────────────────────────────────────

LOADER_MAP = {
    ".pdf":  PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt":  TextLoader,
}


def load_documents(input_dir: Path) -> list:
    """
    Recursively loads all supported documents from input_dir.
    Each document is tagged with its source filename in metadata.

    Supported extensions: .pdf, .docx, .txt

    Returns a flat list of LangChain Document objects.
    """
    all_docs = []
    found = list(input_dir.rglob("*"))
    supported = [f for f in found if f.suffix.lower() in LOADER_MAP and f.is_file()]

    if not supported:
        print(f"[!] No supported documents found in: {input_dir}")
        print(f"    Supported types: {', '.join(LOADER_MAP.keys())}")
        sys.exit(1)

    print(f"[→] Found {len(supported)} document(s) to ingest:")

    for filepath in supported:
        ext = filepath.suffix.lower()
        loader_cls = LOADER_MAP[ext]

        try:
            loader = loader_cls(str(filepath))
            docs = loader.load()

            # Normalise source metadata to just the filename for cleaner citations
            for doc in docs:
                doc.metadata["source"] = filepath.name
                doc.metadata["file_path"] = str(filepath)

            all_docs.extend(docs)
            print(f"    ✓ {filepath.name}  ({len(docs)} page/section(s))")

        except Exception as e:
            print(f"    ✗ {filepath.name} — skipped ({type(e).__name__}: {e})")

    print(f"\n[✓] Loaded {len(all_docs)} raw page/section(s) from {len(supported)} file(s).")
    return all_docs


# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────

def chunk_documents(
    docs: list,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list:
    """
    Splits documents into overlapping chunks using RecursiveCharacterTextSplitter.

    Default values:
        chunk_size=512    — balances context richness with retrieval precision
        chunk_overlap=64  — ensures boundary sentences are not lost

    Each chunk inherits the source document's metadata, plus a `chunk_id`
    for stable identification in citations.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(docs)

    # Add sequential chunk_id for citation traceability
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    print(f"[→] Split into {len(chunks)} chunk(s)  "
          f"(size={chunk_size}, overlap={chunk_overlap})")
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Embedding + FAISS Indexing
# ─────────────────────────────────────────────────────────────────────────────

def build_vectorstore(chunks: list, output_dir: Path) -> FAISS:
    """
    Embeds all chunks using OpenAI text-embedding-3-small and saves
    a FAISS vectorstore to output_dir.

    text-embedding-3-small is chosen for:
        - Cost efficiency (~5× cheaper than ada-002)
        - 1536-dimensional embeddings with strong semantic accuracy
        - Native support for normalised cosine similarity in FAISS

    Requires OPENAI_API_KEY in environment.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[✗] OPENAI_API_KEY not set. Add it to your .env file or environment.")
        sys.exit(1)

    print(f"[→] Embedding {len(chunks)} chunks with text-embedding-3-small ...")
    print("    (This may take a moment for large document sets.)")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key,
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    output_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(output_dir))

    index_size = sum(
        f.stat().st_size for f in output_dir.iterdir() if f.is_file()
    ) / 1024

    print(f"[✓] Vectorstore saved to: {output_dir}  ({index_size:.1f} KB)")
    return vectorstore


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest documents into a FAISS vectorstore for the RAG chatbot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest.py --input_dir ./docs --output_dir ./vectorstore
  python ingest.py --input_dir ./docs --chunk_size 256 --chunk_overlap 32
        """,
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("./docs"),
        help="Directory containing documents to ingest (default: ./docs)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./vectorstore"),
        help="Directory to save the FAISS vectorstore (default: ./vectorstore)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=512,
        help="Maximum tokens per chunk (default: 512)",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=64,
        help="Overlapping tokens between adjacent chunks (default: 64)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  RAG Chatbot — Document Ingestion")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  Input dir:     {args.input_dir.resolve()}")
    print(f"  Output dir:    {args.output_dir.resolve()}")
    print(f"  Chunk size:    {args.chunk_size}")
    print(f"  Chunk overlap: {args.chunk_overlap}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    if not args.input_dir.exists():
        print(f"[✗] Input directory not found: {args.input_dir}")
        sys.exit(1)

    docs   = load_documents(args.input_dir)
    chunks = chunk_documents(docs, args.chunk_size, args.chunk_overlap)
    build_vectorstore(chunks, args.output_dir)

    print("\n[✓] Ingestion complete. Run `streamlit run app.py` to start the chatbot.")


if __name__ == "__main__":
    main()

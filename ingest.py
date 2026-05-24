"""
ingest.py - Document Ingestion and Indexing Module

Loads documents → splits into chunks → generates embeddings → saves FAISS index.
"""

import os
import sys
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DOCUMENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents")
FAISS_INDEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_index")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "text-embedding-3-small"


def load_documents(documents_dir: str) -> list:
    """Load all .txt and .pdf documents from the specified directory."""
    all_documents = []

    if not os.path.exists(documents_dir):
        print(f"Error: Documents directory '{documents_dir}' does not exist.")
        sys.exit(1)

    files = os.listdir(documents_dir)
    if not files:
        print(f"Error: No files found in '{documents_dir}'.")
        sys.exit(1)

    for filename in sorted(files):
        filepath = os.path.join(documents_dir, filename)
        if filename.endswith(".txt"):
            print(f"  Loading TXT: {filename}")
            loader = TextLoader(filepath, encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = filename
            all_documents.extend(docs)
        elif filename.endswith(".pdf"):
            print(f"  Loading PDF: {filename}")
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = filename
            all_documents.extend(docs)
        else:
            print(f"  Skipping: {filename}")

    print(f"\nTotal documents loaded: {len(all_documents)}")
    return all_documents


def split_documents(documents: list) -> list:
    """Split documents into semantic chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    return chunks


def create_vector_store(chunks: list) -> FAISS:
    """Generate embeddings and save FAISS index to disk."""
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_BASE_URL")
    )

    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(FAISS_INDEX_DIR)
    print(f"FAISS index saved to '{FAISS_INDEX_DIR}'")
    return vector_store


def run_ingestion():
    """Run the full ingestion pipeline."""
    print("=" * 60)
    print("FINANCE RAG - Document Ingestion Pipeline")
    print("=" * 60)

    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set. Add it to your .env file.")
        sys.exit(1)

    print("\n[Step 1/3] Loading documents...")
    documents = load_documents(DOCUMENTS_DIR)

    print("\n[Step 2/3] Splitting documents into chunks...")
    chunks = split_documents(documents)

    print("\n[Step 3/3] Creating embeddings and saving FAISS index...")
    create_vector_store(chunks)

    print("\n" + "=" * 60)
    print("Ingestion complete!")
    print(f"  Documents: {len(documents)} | Chunks: {len(chunks)}")
    print(f"  Index saved: {FAISS_INDEX_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    run_ingestion()
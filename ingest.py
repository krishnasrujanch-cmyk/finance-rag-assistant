"""
ingest.py - Document Ingestion and Indexing Module
Loads documents -> splits into chunks -> generates embeddings -> saves FAISS index.
"""

import os
import sys
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")
FAISS_INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "text-embedding-3-small"


def load_documents(documents_dir: str) -> list:
    all_documents = []
    if not os.path.exists(documents_dir):
        print(f"Error: '{documents_dir}' not found.")
        sys.exit(1)

    for filename in sorted(os.listdir(documents_dir)):
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

    print(f"Total documents loaded: {len(all_documents)}")
    return all_documents


def split_documents(documents: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks: {len(chunks)}")
    return chunks


def create_vector_store(chunks: list) -> FAISS:
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
    print("=" * 60)
    print("FINANCE RAG - Document Ingestion Pipeline")
    print("=" * 60)
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set.")
        sys.exit(1)

    documents = load_documents(DOCUMENTS_DIR)
    chunks = split_documents(documents)
    create_vector_store(chunks)

    print("Ingestion complete!")


if __name__ == "__main__":
    run_ingestion()
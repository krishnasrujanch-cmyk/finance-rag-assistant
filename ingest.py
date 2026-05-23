"""
ingest.py - Document Ingestion and Indexing Module

This module handles:
1. Loading documents from the 'documents' folder (supports .txt and .pdf files)
2. Splitting content into semantic chunks using RecursiveCharacterTextSplitter
3. Generating embeddings using OpenAI text-embedding-3-small model
4. Storing embeddings in ChromaDB vector store for retrieval
"""

import os
import sys
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DOCUMENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents")
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")

# Chunking parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Embedding model
EMBEDDING_MODEL = "text-embedding-3-small"


def load_documents(documents_dir: str) -> list:
    """
    Load all .txt and .pdf documents from the specified directory.
    
    Args:
        documents_dir: Path to the folder containing documents.
        
    Returns:
        List of LangChain Document objects.
    """
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
            # Add source metadata
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
            print(f"  Skipping unsupported file: {filename}")
    
    print(f"\nTotal documents loaded: {len(all_documents)}")
    return all_documents


def split_documents(documents: list) -> list:
    """
    Split documents into smaller semantic chunks.
    
    Uses RecursiveCharacterTextSplitter which tries to split on natural
    boundaries (paragraphs, sentences, words) to keep semantic meaning intact.
    
    Args:
        documents: List of LangChain Document objects.
        
    Returns:
        List of chunked Document objects.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    return chunks


def create_vector_store(chunks: list) -> Chroma:
    """
    Generate embeddings and store them in ChromaDB.
    
    Args:
        chunks: List of chunked Document objects.
        
    Returns:
        ChromaDB vector store instance.
    """
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
          openai_api_base=os.getenv("OPENAI_BASE_URL")
    )
    
    # Remove existing ChromaDB directory if it exists (fresh ingestion)
    if os.path.exists(CHROMA_PERSIST_DIR):
        import shutil
        shutil.rmtree(CHROMA_PERSIST_DIR)
        print(f"Cleared existing vector store at '{CHROMA_PERSIST_DIR}'")
    
    # Create ChromaDB vector store from document chunks
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name="finance_docs"
    )
    
    print(f"Vector store created and persisted at '{CHROMA_PERSIST_DIR}'")
    return vector_store


def run_ingestion():
    """
    Run the full ingestion pipeline:
    1. Load documents
    2. Split into chunks
    3. Create embeddings and store in ChromaDB
    """
    print("=" * 60)
    print("FINANCE RAG - Document Ingestion Pipeline")
    print("=" * 60)
    
    # Load environment variables
    load_dotenv()
    
    # Validate API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please create a .env file with your OpenAI API key:")
        print('  OPENAI_API_KEY=your-api-key-here')
        sys.exit(1)
    
    # Step 1: Load documents
    print("\n[Step 1/3] Loading documents...")
    documents = load_documents(DOCUMENTS_DIR)
    
    # Step 2: Split into chunks
    print("\n[Step 2/3] Splitting documents into chunks...")
    chunks = split_documents(documents)
    
    # Step 3: Create vector store
    print("\n[Step 3/3] Creating embeddings and storing in ChromaDB...")
    vector_store = create_vector_store(chunks)
    
    print("\n" + "=" * 60)
    print("Ingestion complete!")
    print(f"  Documents processed: {len(documents)}")
    print(f"  Chunks created:      {len(chunks)}")
    print(f"  Vector store:        {CHROMA_PERSIST_DIR}")
    print("=" * 60)
    
    return vector_store


if __name__ == "__main__":
    run_ingestion()

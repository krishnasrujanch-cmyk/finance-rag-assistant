"""
app.py - Streamlit UI for the Finance RAG Chatbot

Works both locally and on Streamlit Community Cloud.
- Locally: reads OPENAI_API_KEY from .env file
- Cloud:   reads OPENAI_API_KEY from Streamlit Secrets (Settings → Secrets)

If the ChromaDB vector store does not exist, it auto-runs ingestion on startup.
"""

import os
import sys

# Fix for Streamlit Cloud: sqlite3 version issue with ChromaDB
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
TOP_K = 4
TEMPERATURE = 0.2


# ---------------------------------------------------------------------------
# API Key Helper — works with .env (local) and st.secrets (cloud)
# ---------------------------------------------------------------------------
def get_api_key() -> str:
    """Retrieve OpenAI API key from environment or Streamlit secrets."""
    # First try .env file (local development)
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")
    if key and key != "your-openai-api-key-here":
        return key

    # Then try Streamlit secrets (cloud deployment)
    try:
        key = st.secrets["OPENAI_API_KEY"]
        if key:
            os.environ["OPENAI_API_KEY"] = key  # set for LangChain
            return key
    except (KeyError, FileNotFoundError):
        pass

    return ""


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
QA_PROMPT_TEMPLATE = """You are a helpful Finance Domain Support Assistant. Your job is to
help users with questions about investments, mutual funds, stock markets, personal finance,
savings instruments, tax planning, and related financial topics.

INSTRUCTIONS:
1. Use the context provided below to answer the user's question accurately.
2. If the user sends a greeting or a general request (e.g. "help me with mutual funds"),
   welcome them warmly and provide a useful overview using the information in the context.
3. If the context contains relevant information, give a clear, detailed, and well-structured
   answer. Do NOT refuse to answer when the context has relevant content.
4. Only say "I don't have enough information in the provided documents to answer this
   question." when the context truly contains NO relevant information at all.
5. Mention which document or section your answer is based on when possible.
6. Include relevant disclaimers from the source documents when applicable.

Context from retrieved documents:
{context}

User's Question: {question}

Helpful Answer:"""


CONDENSE_PROMPT_TEMPLATE = """Given the following conversation history and a new follow-up
question, rephrase the follow-up question to be a standalone question that captures the
full intent. If the question is already standalone or is a greeting, return it as-is.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""


# ---------------------------------------------------------------------------
# Auto-Ingestion — runs once if chroma_db/ doesn't exist
# ---------------------------------------------------------------------------
def auto_ingest(api_key: str):
    """Run document ingestion to create the vector store."""
    from langchain_community.document_loaders import TextLoader, PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    st.info("🔄 First-time setup: Ingesting documents and creating vector store...")

    # Load documents
    all_documents = []
    for filename in sorted(os.listdir(DOCUMENTS_DIR)):
        filepath = os.path.join(DOCUMENTS_DIR, filename)
        if filename.endswith(".txt"):
            loader = TextLoader(filepath, encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = filename
            all_documents.extend(docs)
        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = filename
            all_documents.extend(docs)

    if not all_documents:
        st.error("No documents found in the 'documents/' folder.")
        st.stop()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(all_documents)

    # Create vector store
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=api_key)

    if os.path.exists(CHROMA_PERSIST_DIR):
        import shutil
        shutil.rmtree(CHROMA_PERSIST_DIR)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name="finance_docs"
    )

    st.success(f"✅ Ingestion complete! Loaded {len(all_documents)} documents → {len(chunks)} chunks.")


# ---------------------------------------------------------------------------
# Initialize RAG Chain
# ---------------------------------------------------------------------------
@st.cache_resource
def initialize_chain(_api_key: str):
    """Initialize and cache the RAG chain. _api_key is prefixed with _ to skip hashing."""

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=_api_key)

    vector_store = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name="finance_docs"
    )

    llm = ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE, openai_api_key=_api_key)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )

    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=QA_PROMPT_TEMPLATE
    )

    condense_question_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template=CONDENSE_PROMPT_TEMPLATE
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        condense_question_prompt=condense_question_prompt,
        return_source_documents=True,
        verbose=False
    )

    return chain


# ---------------------------------------------------------------------------
# Main Streamlit App
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Finance RAG Assistant",
        page_icon="💰",
        layout="wide"
    )

    # ---- Sidebar ----
    with st.sidebar:
        st.title("💰 Finance RAG Assistant")
        st.markdown("---")
        st.markdown("""
**Domain:** Finance

**Documents Covered:**
- 📄 Investment Basics FAQ
- 📄 Mutual Fund Guide
- 📄 Stock Market Basics
- 📄 Personal Finance Planning
- 📄 Fixed Deposits & Savings
        """)
        st.markdown("---")
        st.markdown("""
**Sample questions to try:**
- What is a mutual fund?
- How do I start investing in stocks?
- What is the 50-30-20 budgeting rule?
- Tell me about PPF and its tax benefits
- What are the types of fixed deposits?
- What is the difference between active and passive investing?
        """)
        st.markdown("---")

        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.pop("chain", None)
            st.cache_resource.clear()
            st.rerun()

        st.markdown("---")
        st.caption("Built with LangChain · OpenAI · ChromaDB · Streamlit")

    # ---- Main Area ----
    st.title("💰 Finance Domain Support Assistant")
    st.caption("Ask questions about investments, mutual funds, stocks, tax planning, and more!")

    # Get API key
    api_key = get_api_key()
    if not api_key:
        st.warning("⚠️ OpenAI API key not found.")
        api_key_input = st.text_input(
            "Enter your OpenAI API Key:",
            type="password",
            placeholder="sk-..."
        )
        if api_key_input:
            os.environ["OPENAI_API_KEY"] = api_key_input
            api_key = api_key_input
            st.rerun()
        else:
            st.info("💡 You can also set it via `.env` file locally or in Streamlit Cloud → Settings → Secrets.")
            st.stop()

    # Auto-ingest if vector store doesn't exist
    if not os.path.exists(CHROMA_PERSIST_DIR):
        auto_ingest(api_key)

    # Initialize chain
    chain = initialize_chain(api_key)

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("📄 Source Documents"):
                    st.markdown(message["sources"])

    # Chat input
    if user_input := st.chat_input("Ask a finance question..."):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                try:
                    result = chain.invoke({"question": user_input})
                    answer = result.get("answer", "I encountered an error.")
                    source_docs = result.get("source_documents", [])

                    st.markdown(answer)

                    # Format and show sources
                    sources_text = ""
                    if source_docs:
                        sources = set()
                        for doc in source_docs:
                            src = doc.metadata.get("source", "Unknown")
                            sources.add(src)
                        sources_text = "**Referenced Documents:**\n" + "\n".join(
                            [f"- {s}" for s in sorted(sources)]
                        )
                        with st.expander("📄 Source Documents"):
                            st.markdown(sources_text)
                            st.markdown("---")
                            for i, doc in enumerate(source_docs):
                                st.markdown(
                                    f"**Chunk {i+1}** (from `{doc.metadata.get('source', 'Unknown')}`):"
                                )
                                preview = doc.page_content[:500]
                                if len(doc.page_content) > 500:
                                    preview += "..."
                                st.text(preview)
                                st.markdown("---")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources_text
                    })

                except Exception as e:
                    error_msg = f"An error occurred: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })


if __name__ == "__main__":
    main()
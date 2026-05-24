"""
app.py - Streamlit UI for the Finance RAG Chatbot

Works on Streamlit Community Cloud and locally.
Uses FAISS vector store (no ChromaDB / SQLite dependency issues).
Auto-ingests documents on first run if faiss_index/ doesn't exist.
"""

import os
import sys
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")
FAISS_INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
TOP_K = 4
TEMPERATURE = 0.2

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
# API Key Helper
# ---------------------------------------------------------------------------
def get_api_key() -> str:
    """Get OpenAI API key from .env (local) or st.secrets (cloud)."""
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY", "")
    if key and key != "your-openai-api-key-here":
        return key
    try:
        key = st.secrets.get("OPENAI_API_KEY", "")
        if key:
            os.environ["OPENAI_API_KEY"] = key
            return key
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# Auto-Ingestion
# ---------------------------------------------------------------------------
def auto_ingest(api_key: str):
    """Run document ingestion to build the FAISS index."""
    from langchain_community.document_loaders import TextLoader, PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    with st.status("⚙️ First-time setup: Building vector index from documents...", expanded=True) as status:
        # Load docs
        st.write("📂 Loading documents...")
        all_docs = []
        for filename in sorted(os.listdir(DOCUMENTS_DIR)):
            filepath = os.path.join(DOCUMENTS_DIR, filename)
            if filename.endswith(".txt"):
                loader = TextLoader(filepath, encoding="utf-8")
                docs = loader.load()
                for d in docs:
                    d.metadata["source"] = filename
                all_docs.extend(docs)
            elif filename.endswith(".pdf"):
                loader = PyPDFLoader(filepath)
                docs = loader.load()
                for d in docs:
                    d.metadata["source"] = filename
                all_docs.extend(docs)

        if not all_docs:
            st.error("No documents found in the 'documents/' folder.")
            st.stop()

        # Split
        st.write(f"✂️ Splitting {len(all_docs)} documents into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(all_docs)

        # Embed and save
        st.write(f"🔢 Generating embeddings for {len(chunks)} chunks...")
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=api_key)
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(FAISS_INDEX_DIR)

        status.update(label=f"✅ Index built: {len(all_docs)} docs → {len(chunks)} chunks", state="complete")


# ---------------------------------------------------------------------------
# Initialize RAG Chain
# ---------------------------------------------------------------------------
@st.cache_resource
def initialize_chain(_api_key: str):
    """Load FAISS index and create the RAG chain (cached)."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=_api_key)

    vector_store = FAISS.load_local(
        FAISS_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
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
# Main App
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Finance RAG Assistant",
        page_icon="💰",
        layout="wide"
    )

    # Sidebar
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
**Try asking:**
- What is a mutual fund?
- How do I start investing in stocks?
- What is the 50-30-20 budgeting rule?
- Tell me about PPF and tax benefits
- Types of fixed deposits?
- Difference between active and passive investing?
        """)
        st.markdown("---")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.cache_resource.clear()
            st.rerun()
        st.markdown("---")
        st.caption("Built with LangChain · OpenAI · FAISS · Streamlit")

    # Main
    st.title("💰 Finance Domain Support Assistant")
    st.caption("Ask questions about investments, mutual funds, stocks, tax planning, and more!")

    # API Key
    api_key = get_api_key()
    if not api_key:
        st.warning("⚠️ OpenAI API key not found.")
        api_key_input = st.text_input("Enter your OpenAI API Key:", type="password", placeholder="sk-...")
        if api_key_input:
            os.environ["OPENAI_API_KEY"] = api_key_input
            api_key = api_key_input
            st.rerun()
        else:
            st.info("Set it in `.env` locally, or in Streamlit Cloud → Settings → Secrets as `OPENAI_API_KEY`.")
            st.stop()

    # Auto-ingest if needed
    if not os.path.exists(FAISS_INDEX_DIR):
        auto_ingest(api_key)

    # Load chain
    chain = initialize_chain(api_key)

    # Chat state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📄 Source Documents"):
                    st.markdown(msg["sources"])

    # Input
    if user_input := st.chat_input("Ask a finance question..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                try:
                    result = chain.invoke({"question": user_input})
                    answer = result.get("answer", "Error generating response.")
                    source_docs = result.get("source_documents", [])

                    st.markdown(answer)

                    sources_text = ""
                    if source_docs:
                        sources = sorted({d.metadata.get("source", "Unknown") for d in source_docs})
                        sources_text = "**Referenced Documents:**\n" + "\n".join(f"- {s}" for s in sources)
                        with st.expander("📄 Source Documents"):
                            st.markdown(sources_text)
                            st.markdown("---")
                            for i, doc in enumerate(source_docs):
                                st.markdown(f"**Chunk {i+1}** — `{doc.metadata.get('source', 'Unknown')}`")
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
                    err = f"Error: {str(e)}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})


if __name__ == "__main__":
    main()
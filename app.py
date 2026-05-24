"""
app.py - Streamlit Finance RAG Assistant
Uses LCEL (LangChain Expression Language) - compatible with Python 3.14.
No ChromaDB. No ConversationalRetrievalChain. Works on Streamlit Cloud.
"""

import os
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

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
# System Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a helpful Finance Domain Support Assistant. Help users with
questions about investments, mutual funds, stock markets, personal finance, savings
instruments, tax planning, and related financial topics.

INSTRUCTIONS:
1. Answer using the context below. If the user sends a greeting or general request,
   welcome them and give a helpful overview from the context.
2. Give clear, detailed answers when the context is relevant.
3. Only say "I don't have enough information in the provided documents to answer this."
   when the context has NO relevant information at all.
4. Cite the source document when possible.
5. Include relevant disclaimers from source documents when applicable.

Context from retrieved documents:
{context}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_api_key() -> str:
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


def format_docs(docs) -> str:
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def get_sources(docs) -> str:
    sources = sorted({d.metadata.get("source", "Unknown") for d in docs})
    return "**Referenced Documents:**\n" + "\n".join(f"- {s}" for s in sources)


# ---------------------------------------------------------------------------
# Auto-Ingestion
# ---------------------------------------------------------------------------
def auto_ingest(api_key: str):
    from langchain_community.document_loaders import TextLoader, PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    with st.status("⚙️ First-time setup: Building vector index...", expanded=True) as status:
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
            st.error("No documents found in 'documents/' folder.")
            st.stop()

        st.write(f"✂️ Splitting {len(all_docs)} documents into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(all_docs)

        st.write(f"🔢 Generating embeddings for {len(chunks)} chunks...")
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=api_key)
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(FAISS_INDEX_DIR)

        status.update(
            label=f"✅ Ready! {len(all_docs)} docs → {len(chunks)} chunks indexed.",
            state="complete"
        )


# ---------------------------------------------------------------------------
# Initialize chain (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def initialize_chain(_api_key: str):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=_api_key)
    vector_store = FAISS.load_local(
        FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})
    llm = ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE, openai_api_key=_api_key)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"]))
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Finance RAG Assistant", page_icon="💰", layout="wide")

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
- Active vs passive investing?
        """)
        st.markdown("---")
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.cache_resource.clear()
            st.rerun()
        st.markdown("---")
        st.caption("Built with LangChain · OpenAI · FAISS · Streamlit")

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
            st.info("Set it in `.env` locally, or Streamlit Cloud → Settings → Secrets as `OPENAI_API_KEY`.")
            st.stop()

    # Auto-ingest on first run
    if not os.path.exists(FAISS_INDEX_DIR):
        auto_ingest(api_key)

    # Initialize chain
    chain, retriever = initialize_chain(api_key)

    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📄 Source Documents"):
                    st.markdown(msg["sources"])

    # Chat input
    if user_input := st.chat_input("Ask a finance question..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                try:
                    # Get retrieved docs for source display
                    retrieved_docs = retriever.invoke(user_input)

                    # Run chain
                    answer = chain.invoke({
                        "question": user_input,
                        "chat_history": st.session_state.chat_history
                    })

                    st.markdown(answer)

                    # Show sources
                    sources_text = ""
                    if retrieved_docs:
                        sources_text = get_sources(retrieved_docs)
                        with st.expander("📄 Source Documents"):
                            st.markdown(sources_text)
                            st.markdown("---")
                            for i, doc in enumerate(retrieved_docs):
                                st.markdown(f"**Chunk {i+1}** — `{doc.metadata.get('source', 'Unknown')}`")
                                preview = doc.page_content[:500]
                                if len(doc.page_content) > 500:
                                    preview += "..."
                                st.text(preview)
                                st.markdown("---")

                    # Update history
                    st.session_state.chat_history.append(HumanMessage(content=user_input))
                    st.session_state.chat_history.append(AIMessage(content=answer))
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
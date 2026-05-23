"""
app.py - Streamlit UI for the Finance RAG Chatbot (Bonus Feature)

This module provides a web-based interface for the Finance RAG chatbot
using Streamlit. It includes:
1. Chat interface with conversation history
2. Source document citation display
3. Sidebar with information and controls
"""

import os
import sys
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
TOP_K = 4
TEMPERATURE = 0.2

# ---------------------------------------------------------------------------
# Prompts (same as chatbot.py)
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


@st.cache_resource
def initialize_chain():
    """Initialize and cache the RAG chain."""
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY not found. Please set it in your .env file.")
        st.stop()

    # Load vector store
    if not os.path.exists(CHROMA_PERSIST_DIR):
        st.error("Vector store not found. Please run `python ingest.py` first.")
        st.stop()

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=api_key)

    vector_store = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name="finance_docs"
    )

    llm = ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE, openai_api_key=api_key)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )

    # QA prompt — only {context} and {question}
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


def main():
    """Main Streamlit application."""

    # Page configuration
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
        - Investment Basics FAQ
        - Mutual Fund Guide
        - Stock Market Basics
        - Personal Finance Planning
        - Fixed Deposits & Savings
        """)
        st.markdown("---")
        st.markdown("""
        **How to use:**
        1. Type your finance question below
        2. The assistant retrieves relevant info from documents
        3. Get grounded answers with source citations
        4. Ask follow-up questions for deeper understanding
        """)
        st.markdown("---")

        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.pop("chain", None)
            st.cache_resource.clear()
            st.rerun()

        st.markdown("---")
        st.caption("Built with LangChain, OpenAI, and ChromaDB")

    # Main chat area
    st.title("💰 Finance Domain Support Assistant")
    st.caption("Ask questions about investments, mutual funds, stocks, tax planning, and more!")

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize the chain
    chain = initialize_chain()

    # Display existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("📄 Source Documents"):
                    st.markdown(message["sources"])

    # Chat input
    if user_input := st.chat_input("Ask a finance question..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                try:
                    result = chain.invoke({"question": user_input})
                    answer = result.get("answer", "I encountered an error.")
                    source_docs = result.get("source_documents", [])

                    st.markdown(answer)

                    # Format and display sources
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
                                st.markdown(f"**Chunk {i+1}** (from `{doc.metadata.get('source', 'Unknown')}`):")
                                st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                                st.markdown("---")

                    # Save to session state
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

"""
chatbot.py - RAG-based Finance Domain Chatbot

This module implements:
1. Loading the persisted ChromaDB vector store
2. Retrieval of top-k relevant chunks per user query
3. Context-aware conversational chain with conversation history
4. Grounded responses strictly based on retrieved documents
5. Safety mechanism to refuse when information is not available
"""

import os
import sys
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
TOP_K = 4  # Number of relevant chunks to retrieve
TEMPERATURE = 0.2  # Slight creativity while staying grounded


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# QA Prompt — used by the combine_docs chain.
# IMPORTANT: ConversationalRetrievalChain's combine_docs_chain only receives
# {context} and {question}. Do NOT include {chat_history} here — that is
# handled separately by the condense-question step.
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


# Condense Prompt — reformulates follow-up questions using chat history
CONDENSE_PROMPT_TEMPLATE = """Given the following conversation history and a new follow-up
question, rephrase the follow-up question to be a standalone question that captures the
full intent. If the question is already standalone or is a greeting, return it as-is.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""


def load_vector_store() -> Chroma:
    """
    Load the persisted ChromaDB vector store.

    Returns:
        ChromaDB vector store instance.
    """
    if not os.path.exists(CHROMA_PERSIST_DIR):
        print("Error: Vector store not found. Please run ingest.py first.")
        sys.exit(1)

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_BASE_URL")
    )

    vector_store = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name="finance_docs"
    )

    # Verify the store has documents
    collection_count = vector_store._collection.count()
    if collection_count == 0:
        print("Error: Vector store is empty. Please run ingest.py first.")
        sys.exit(1)

    print(f"Vector store loaded with {collection_count} chunks.")
    return vector_store


def create_rag_chain(vector_store: Chroma) -> ConversationalRetrievalChain:
    """
    Create the conversational RAG chain.

    Args:
        vector_store: ChromaDB vector store instance.

    Returns:
        ConversationalRetrievalChain ready for Q&A.
    """
    # Initialize the LLM
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Create retriever from vector store
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )

    # Set up conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    # QA prompt — only {context} and {question}
    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=QA_PROMPT_TEMPLATE
    )

    # Condense question prompt — uses {chat_history} and {question}
    condense_question_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template=CONDENSE_PROMPT_TEMPLATE
    )

    # Build the conversational retrieval chain
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


def format_source_documents(source_docs: list) -> str:
    """
    Format source documents for display.

    Args:
        source_docs: List of source Document objects.

    Returns:
        Formatted string of source references.
    """
    if not source_docs:
        return ""

    sources = set()
    for doc in source_docs:
        source_name = doc.metadata.get("source", "Unknown")
        sources.add(source_name)

    return "\n  Sources: " + ", ".join(sorted(sources))


def run_chatbot():
    """
    Run the interactive chatbot loop.
    """
    print("=" * 60)
    print("FINANCE RAG - Domain Support Assistant")
    print("=" * 60)

    # Load environment variables
    load_dotenv()

    # Validate API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please create a .env file with your OpenAI API key:")
        print('  OPENAI_API_KEY=your-api-key-here')
        sys.exit(1)

    # Load vector store
    print("\nLoading vector store...")
    vector_store = load_vector_store()

    # Create RAG chain
    print("Initializing RAG chain...")
    chain = create_rag_chain(vector_store)

    print("\n" + "-" * 60)
    print("Finance Assistant is ready! Ask me anything about:")
    print("  - Investment basics (stocks, bonds, mutual funds, ETFs)")
    print("  - Stock market trading and analysis")
    print("  - Personal finance planning and budgeting")
    print("  - Tax planning and deductions")
    print("  - Fixed deposits and savings instruments")
    print("\nType 'quit' or 'exit' to end the conversation.")
    print("-" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "q"]:
            print("\nThank you for using Finance RAG Assistant. Goodbye!")
            break

        try:
            # Get response from the RAG chain
            result = chain.invoke({"question": user_input})

            answer = result.get("answer", "I encountered an error processing your question.")
            source_docs = result.get("source_documents", [])

            # Display answer
            print(f"\nAssistant: {answer}")

            # Display sources
            sources_text = format_source_documents(source_docs)
            if sources_text:
                print(sources_text)

            print()  # Blank line for readability

        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again with a different question.\n")


if __name__ == "__main__":
    run_chatbot()

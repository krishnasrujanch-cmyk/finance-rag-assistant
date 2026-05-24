"""
chatbot.py - RAG-based Finance Domain Chatbot (terminal version)
Uses FAISS vector store instead of ChromaDB.
"""

import os
import sys
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FAISS_INDEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_index")
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


def load_vector_store() -> FAISS:
    """Load the persisted FAISS index."""
    if not os.path.exists(FAISS_INDEX_DIR):
        print("Error: FAISS index not found. Please run ingest.py first.")
        sys.exit(1)

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_BASE_URL")
    )

    vector_store = FAISS.load_local(
        FAISS_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("FAISS index loaded successfully.")
    return vector_store


def create_rag_chain(vector_store: FAISS) -> ConversationalRetrievalChain:
    """Create the conversational RAG chain."""
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_BASE_URL")
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=QA_PROMPT_TEMPLATE
    )

    condense_question_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template=CONDENSE_PROMPT_TEMPLATE
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


def run_chatbot():
    """Run the interactive chatbot loop."""
    print("=" * 60)
    print("FINANCE RAG - Domain Support Assistant")
    print("=" * 60)

    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set.")
        sys.exit(1)

    print("\nLoading FAISS index...")
    vector_store = load_vector_store()

    print("Initializing RAG chain...")
    chain = create_rag_chain(vector_store)

    print("\n" + "-" * 60)
    print("Finance Assistant is ready!")
    print("Type 'quit' or 'exit' to end.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit", "q"]:
            print("\nGoodbye!")
            break

        try:
            result = chain.invoke({"question": user_input})
            answer = result.get("answer", "Error processing your question.")
            source_docs = result.get("source_documents", [])

            print(f"\nAssistant: {answer}")

            if source_docs:
                sources = sorted({d.metadata.get("source", "Unknown") for d in source_docs})
                print(f"  Sources: {', '.join(sources)}")
            print()

        except Exception as e:
            print(f"\nError: {str(e)}\n")


if __name__ == "__main__":
    run_chatbot()
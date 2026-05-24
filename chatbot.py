"""
chatbot.py - RAG-based Finance Domain Chatbot (terminal version)
Uses LCEL (LangChain Expression Language) - works on Python 3.14.
"""

import os
import sys
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
TOP_K = 4
TEMPERATURE = 0.2

SYSTEM_PROMPT = """You are a helpful Finance Domain Support Assistant. Help users with
questions about investments, mutual funds, stock markets, personal finance, savings
instruments, tax planning, and related financial topics.

INSTRUCTIONS:
1. Answer using the context below. If the user sends a greeting or general request,
   welcome them and give a helpful overview from the context.
2. Give clear, detailed answers when the context is relevant.
3. Only say "I don't have enough information in the provided documents to answer this."
   when the context has NO relevant information.
4. Cite the source document when possible.

Context:
{context}"""


def load_vector_store() -> FAISS:
    if not os.path.exists(FAISS_INDEX_DIR):
        print("Error: FAISS index not found. Run ingest.py first.")
        sys.exit(1)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=os.getenv("OPENAI_API_KEY"))
    return FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)


def format_docs(docs) -> str:
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def run_chatbot():
    print("=" * 60)
    print("FINANCE RAG - Domain Support Assistant")
    print("=" * 60)

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set.")
        sys.exit(1)

    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})
    llm = ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE, openai_api_key=os.getenv("OPENAI_API_KEY"))

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    chain = (
        RunnablePassthrough.assign(context=lambda x: format_docs(retriever.invoke(x["question"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    chat_history = []
    print("\nFinance Assistant is ready! Type 'quit' to exit.\n")

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
            answer = chain.invoke({"question": user_input, "chat_history": chat_history})
            print(f"\nAssistant: {answer}\n")
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=answer))
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    run_chatbot()
# Finance RAG - Domain Support Assistant

## Domain: Finance

This project implements a **Retrieval-Augmented Generation (RAG)** conversational assistant for the **Finance domain**. The chatbot answers user questions about investments, mutual funds, stock markets, personal finance planning, tax planning, and savings instruments — strictly grounded in the provided documents.

---

## Public Data Sources

The knowledge base consists of 5 publicly sourced finance documents covering:

| Document | Content | Source Type |
|---|---|---|
| `investment_basics_faq.txt` | Investment fundamentals, diversification, compound interest, SIPs, asset allocation, risk types | Public investment education material |
| `mutual_fund_guide.txt` | Types of mutual funds, key metrics (NAV, CAGR, Sharpe ratio), taxation, how to invest | SEBI/AMFI public investor education |
| `stock_market_basics.txt` | BSE/NSE overview, order types, fundamental & technical analysis, settlement, SEBI regulations | Stock exchange public education guides |
| `personal_finance_planning.txt` | 50-30-20 rule, emergency funds, insurance, tax planning (80C/80D), retirement planning, loans | Public financial literacy resources |
| `fixed_deposits_savings.txt` | FD types, PPF, SSY, NSC, RD, SCSS, Sovereign Gold Bonds, comparison of instruments | Government scheme public documents (India Post, RBI) |

These documents are based on publicly available information from SEBI, AMFI, RBI, India Post, and government financial literacy portals.

---

## Architecture

```
User Query
    │
    ▼
┌──────────────────────┐
│  Query Reformulation  │  ← Uses conversation history to create standalone query
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  ChromaDB Retriever   │  ← Retrieves top-4 relevant chunks via similarity search
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Context Injection    │  ← Retrieved chunks + chat history injected into prompt
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  GPT-4o-mini LLM      │  ← Generates grounded answer from context only
└──────────┬───────────┘
           │
           ▼
     Answer + Sources
```

**Tech Stack:**
- **Python 3.11**
- **LangChain** — orchestration framework
- **OpenAI GPT-4o-mini** — language model
- **OpenAI text-embedding-3-small** — embedding model
- **ChromaDB** — vector store
- **Streamlit** — web UI (bonus)

---

## Setup & Installation

### Prerequisites
- Python 3.11+
- OpenAI API key

### Step 1: Clone / Extract the project
```bash
cd finance_rag_project
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure API key
```bash
# Create .env file from the template
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-actual-key-here
```

### Step 4: Run document ingestion
```bash
python ingest.py
```
This will:
- Load all 5 finance documents from the `documents/` folder
- Split them into semantic chunks (1000 chars, 200 overlap)
- Generate embeddings using `text-embedding-3-small`
- Store embeddings in ChromaDB at `chroma_db/`

### Step 5: Run the chatbot

**Option A — Terminal chatbot:**
```bash
python chatbot.py
```

**Option B — Streamlit web UI (bonus):**
```bash
streamlit run app.py
```

---

## Usage Examples

### Basic Question
```
You: What is a mutual fund?
Assistant: A mutual fund is a pooled investment vehicle managed by a professional fund 
manager. It collects money from multiple investors and invests it in a diversified portfolio 
of stocks, bonds, or other securities...
  Sources: mutual_fund_guide.txt, investment_basics_faq.txt
```

### Follow-up Question
```
You: What are the different types?
Assistant: Based on the Mutual Fund Investment Guide, there are several types of mutual 
funds: 1) Equity Mutual Funds (Large-Cap, Mid-Cap, Small-Cap, Multi-Cap, Flexi-Cap), 
2) Debt Mutual Funds (Liquid, Short Duration, Corporate Bond, Gilt, Dynamic Bond), 
3) Hybrid Mutual Funds (Conservative, Balanced Advantage, Aggressive), 4) Index Funds, 
and 5) ELSS (tax-saving funds)...
  Sources: mutual_fund_guide.txt
```

### Out-of-scope Question
```
You: What is the current price of Bitcoin?
Assistant: I don't have enough information in the provided documents to answer this 
question.
```

---

## Project Structure

```
finance_rag_project/
├── documents/                        # Finance domain documents
│   ├── investment_basics_faq.txt
│   ├── mutual_fund_guide.txt
│   ├── stock_market_basics.txt
│   ├── personal_finance_planning.txt
│   └── fixed_deposits_savings.txt
├── chroma_db/                        # ChromaDB vector store (generated)
├── ingest.py                         # Document ingestion & indexing
├── chatbot.py                        # Terminal-based RAG chatbot
├── app.py                            # Streamlit web UI (bonus)
├── requirements.txt                  # Python dependencies
├── .env.example                      # Environment variable template
└── README.md                         # This file
```

---

## Key Design Decisions

1. **Chunk Size (1000) & Overlap (200):** Balances capturing enough context per chunk while keeping retrieval precise. Overlap ensures no information is lost at chunk boundaries.

2. **Top-K = 4:** Retrieves 4 chunks to provide sufficient context without overwhelming the LLM's context window.

3. **Temperature = 0.0:** Ensures deterministic, consistent responses — critical for financial information accuracy.

4. **ConversationalRetrievalChain:** Reformulates follow-up questions into standalone queries using conversation history, enabling natural multi-turn conversations.

5. **Grounded System Prompt:** Explicitly instructs the LLM to answer only from retrieved context and refuse when information is missing, preventing hallucination.

---

## Disclaimer

This chatbot is built for educational purposes as part of a graded project. The financial information in the documents is sourced from publicly available materials and should not be treated as professional financial advice. Always consult a certified financial planner for personalized guidance.

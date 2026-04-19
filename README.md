# Intelligent Property Price Prediction & Agentic Advisor

An AI-driven real estate ecosystem that combines **Classical Machine Learning** for price prediction with **Agentic AI** for expert investment advisory.

> **Milestone 2 Update** — Agentic AI Real Estate Advisory Assistant (End-Sem)

---

## 🚀 New in Milestone 2: Agentic AI Advisor

The system has been extended from a simple predictor into a reasoning agent that analyzes market trends, regulatory data, and user preferences to generate structured advisory reports.

### 🧠 Agentic Workflow (LangGraph)
We implemented an **explicit state management** workflow using **LangGraph**. The agent processes requests through the following nodes:
1.  **Analyze**: Evaluates property features and ML-predicted prices for internal consistency.
2.  **Retrieve (RAG)**: Queries a local vector database for legal (RERA) and market context.
3.  **Compare**: Performs a horizontal scan of the historical dataset to define price brackets.
4.  **Generate**: Synthesizes all data into a professional Markdown report using **Groq (Llama 3.3)**.

### 📚 Retrieval-Augmented Generation (RAG)
*   **Knowledge Base**: Scraped and indexed Wikipedia data on the **RERA Act 2016** and **Indian Real Estate Trends**.
*   **Vector Engine**: Powered by **FAISS** and **HuggingFace Embeddings** (`all-MiniLM-L6-v2`) for fast, local, and free semantic retrieval.

### 📊 Comparative Analysis (Extension)
The agent automatically filters the `Housing.csv` dataset to find properties with similar area and bedroom counts, providing:
*   Average historical price for the segment.
*   Segment price range (Min/Max).
*   Contextual validation of the AI's prediction.

### 📄 PDF Report Export (Extension)
Users can now export the full AI Advisory Report as a professionally formatted **PDF document** directly from the UI, including:
*   Property Summary
*   Market Trend Insights
*   Investment Recommendations
*   Legal Disclaimers

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT WEB UI                          │
│  ┌──────────────┐ ┌───────────────┐ ┌────────────────────┐  │
│  │ Predict Price│ │  AI Advisor   │ │   Batch Analytics  │  │
│  └──────┬───────┘ └───────┬───────┘ └────────┬───────────┘  │
│         │                 │                   │              │
├─────────┼─────────────────┼───────────────────┼──────────────┤
│         ▼                 ▼                   ▼              │
│  ┌────────────────┐ ┌────────────────┐ ┌───────────────────┐ │
│  │   ML ENGINE    │ │  LANGGRAPH     │ │    RAG ENGINE     │ │
│  │ (Scikit-Learn) │ │ (Llama 3.3/Groq) │ │ (FAISS + HF)    │ │
│  └────────────────┘ └────────────────┘ └───────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Updated Tech Stack

| Category | Tool |
|----------|------|
| **LLM Inference** | Groq (Llama-3.3-70b-versatile) |
| **Agent Framework** | LangGraph, LangChain |
| **Vector Database** | FAISS |
| **Embeddings** | HuggingFace (Local) |
| **PDF Generation** | fpdf2 |
| **Classic ML** | Scikit-Learn |
| **Dashboard** | Streamlit |

---

## ⚡ Quick Start (Updated)

### 1. Installation
```bash
git clone https://github.com/Sahil5H4RD4/property-price-prediction.git
cd property-price-prediction
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Build Knowledge Base (RAG)
Fetch market data and build the vector index:
```bash
python src/build_rag.py
```

### 4. Run the App
```bash
streamlit run app.py
```

---

## 📂 Project Structure (New Modules)

*   `src/agent.py`: LangGraph state management and LLM reasoning logic.
*   `src/build_rag.py`: RAG pipeline for fetching and indexing market documents.
*   `src/pdf_exporter.py`: Utility to convert AI reports to PDF.
*   `data/vectorstore/`: Local FAISS index files.

---

## License
This project is for educational purposes as part of the AI/ML End-Sem course project.

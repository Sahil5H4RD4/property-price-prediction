"""
LangGraph Advisory Agent
========================
Multi-step real estate advisory agent:
  analyze → retrieve market data → compare properties → generate report
"""

import logging
import os

import pandas as pd
from typing import Dict, Any, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# Module-level vectorstore cache — loaded once, reused on every request.
_vectorstore_cache: FAISS | None = None


class AgentState(TypedDict):
    property_details: Dict[str, Any]
    predicted_price: float
    model_metrics: Dict[str, Any]
    market_context: str
    comparable_summary: str
    analysis: str
    report: str


def _load_vectorstore() -> FAISS | None:
    """Load FAISS vectorstore from disk once and cache at module level."""
    global _vectorstore_cache
    if _vectorstore_cache is not None:
        return _vectorstore_cache

    base_dir = os.path.dirname(os.path.dirname(__file__))
    vectorstore_path = os.path.join(base_dir, 'data', 'vectorstore')

    if not os.path.exists(vectorstore_path):
        logger.warning("Vectorstore not found at %s", vectorstore_path)
        return None

    logger.info("Loading FAISS vectorstore from %s", vectorstore_path)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # allow_dangerous_deserialization is required by LangChain for FAISS.load_local().
    # This is safe here because the vectorstore was built by build_rag.py and stored
    # locally — it is never loaded from an untrusted external source.
    _vectorstore_cache = FAISS.load_local(
        vectorstore_path, embeddings, allow_dangerous_deserialization=True
    )
    logger.info("Vectorstore loaded and cached")
    return _vectorstore_cache


def analyze_input(state: AgentState) -> dict:
    """Summarise the property and predicted price."""
    price = state['predicted_price']
    details = state['property_details']
    analysis = (
        f"Property: {details.get('area')} sqft, "
        f"{details.get('bedrooms')} BHK, "
        f"{details.get('bathrooms')} baths. "
        f"AI predicted price: \u20b9{price:,.0f}."
    )
    logger.debug("Analysis: %s", analysis)
    return {"analysis": analysis}


def retrieve_market_data(state: AgentState) -> dict:
    """Retrieve relevant market context from the FAISS vectorstore."""
    vs = _load_vectorstore()
    if vs is None:
        return {"market_context": "Market vector database unavailable. Using general knowledge."}

    query = (
        f"Real estate market trends pricing RERA regulations housing supply "
        f"{state['property_details'].get('area', '')} sqft property"
    )
    docs = vs.similarity_search(query, k=3)
    context = "\n\n".join(doc.page_content for doc in docs)
    logger.debug("Retrieved %d market context chunks", len(docs))
    return {"market_context": context}


def compare_properties(state: AgentState) -> dict:
    """Find comparable properties in Housing.csv with progressive fallback."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, 'data', 'Housing.csv')

    if not os.path.exists(data_path):
        return {"comparable_summary": "Historical dataset unavailable for comparison."}

    df = pd.read_csv(data_path)
    details = state['property_details']
    area = float(details.get('area', 5000))
    beds = int(details.get('bedrooms', 3))

    # Start with tight filter (±20% area, exact bedrooms), then progressively relax.
    for area_margin, require_beds in [(0.20, True), (0.35, True), (0.50, False)]:
        mask = (df['area'] >= area * (1 - area_margin)) & (df['area'] <= area * (1 + area_margin))
        if require_beds:
            mask &= df['bedrooms'] == beds
        similar = df[mask]
        if len(similar) >= 3:
            break

    if len(similar) == 0:
        return {
            "comparable_summary": (
                f"No comparable properties found for {beds} beds, ~{area:.0f} sqft."
            )
        }

    avg_price = similar['price'].mean()
    min_price = similar['price'].min()
    max_price = similar['price'].max()
    label = f"{beds} beds" if require_beds else "any bedrooms"
    comp_summary = (
        f"Found {len(similar)} comparable properties "
        f"({label}, {area * (1 - area_margin):.0f}\u2013{area * (1 + area_margin):.0f} sqft). "
        f"Average historic price: \u20b9{avg_price:,.0f}. "
        f"Range: \u20b9{min_price:,.0f} to \u20b9{max_price:,.0f}."
    )
    logger.debug("Comparables: %s", comp_summary)
    return {"comparable_summary": comp_summary}


def generate_recommendations(state: AgentState) -> dict:
    """Call Groq LLM to produce a structured Markdown advisory report."""
    llm = ChatGroq(
        model_name=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=float(os.environ.get("GROQ_TEMPERATURE", "0.2")),
    )

    system_prompt = (
        "You are an expert Real Estate AI Advisor. Produce a structured, professional "
        "advisory report in Markdown.\n\n"
        "The report MUST include:\n"
        "- Property Summary\n"
        "- Price Prediction Interpretation (compare predicted vs comparables)\n"
        "- Market Trend Insights (based on provided market context)\n"
        "- Recommended Actions (for buyers or investors)\n"
        "- Supporting Sources or References\n"
        "- Legal and financial disclaimers\n\n"
        "Be analytical and objective. Do NOT make unsupported financial claims."
    )

    prompt = (
        f"Property Details: {state['property_details']}\n"
        f"Predicted Price: \u20b9{state['predicted_price']:,.0f}\n"
        f"Model R\u00b2: {state['model_metrics'].get('model_r2', 'N/A')}\n\n"
        f"Market Context:\n{state['market_context']}\n\n"
        f"Comparable Properties:\n{state['comparable_summary']}"
    )

    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=prompt)])
    logger.info("Advisory report generated (%d chars)", len(response.content))
    return {"report": response.content}


# Build and compile the LangGraph workflow once at import time.
_workflow = StateGraph(AgentState)
_workflow.add_node("analyze", analyze_input)
_workflow.add_node("retrieve", retrieve_market_data)
_workflow.add_node("compare", compare_properties)
_workflow.add_node("generate", generate_recommendations)
_workflow.add_edge(START, "analyze")
_workflow.add_edge("analyze", "retrieve")
_workflow.add_edge("retrieve", "compare")
_workflow.add_edge("compare", "generate")
_workflow.add_edge("generate", END)
_app = _workflow.compile()


def run_advisory_agent(
    property_details: dict,
    predicted_price: float,
    model_metrics: dict,
) -> str:
    """Run the full advisory pipeline and return a Markdown report.

    Raises:
        EnvironmentError: if GROQ_API_KEY is not set.
    """
    from dotenv import load_dotenv
    load_dotenv()

    # Priority: 1. Environment Variable / .env  2. Streamlit Secrets (for cloud hosting)
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            pass

    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Add it to your .env file locally or "
            "to Streamlit Secrets in the cloud dashboard."
        )

    # Ensure it's in the environment for LangChain components
    os.environ["GROQ_API_KEY"] = api_key

    logger.info(
        "Running advisory agent for %.0f sqft, %s BHK, predicted \u20b9%,.0f",
        property_details.get('area', 0),
        property_details.get('bedrooms', '?'),
        predicted_price,
    )

    initial_state: AgentState = {
        "property_details": property_details,
        "predicted_price": predicted_price,
        "model_metrics": model_metrics,
        "market_context": "",
        "comparable_summary": "",
        "analysis": "",
        "report": "",
    }

    final_state = _app.invoke(initial_state)
    return final_state["report"]

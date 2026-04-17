import os
import pandas as pd
from typing import Dict, Any, List, TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Define State
class AgentState(TypedDict):
    property_details: Dict[str, Any]
    predicted_price: float
    model_metrics: Dict[str, Any]
    market_context: str
    comparable_summary: str
    analysis: str
    report: str

def load_vectorstore():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    vectorstore_path = os.path.join(base_dir, 'data', 'vectorstore')
    if os.path.exists(vectorstore_path):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # Enable dangerous deserialization because we created the index locally and trust it
        return FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
    return None

def analyze_input(state: AgentState):
    """Analyze the property inputs and price."""
    price = state['predicted_price']
    details = state['property_details']
    
    analysis = f"Property is a {details.get('area')} sqft {details.get('bedrooms')}BHK with {details.get('bathrooms')} baths."
    analysis += f" The AI predicted price is ₹{price:,.0f}."
    
    return {"analysis": analysis}

def retrieve_market_data(state: AgentState):
    """Retrieve market data from FAISS vector store based on property details."""
    vs = load_vectorstore()
    if vs is None:
        return {"market_context": "No market vector database found. Operating with general knowledge."}
    
    query = "Real estate market trends, pricing, RERA regulations, housing supply"
    docs = vs.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    return {"market_context": context}

def compare_properties(state: AgentState):
    """Find similar properties directly from the training dataset (Housing.csv) to compute comparable metrics."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, 'data', 'Housing.csv')
    
    if not os.path.exists(data_path):
        return {"comparable_summary": "No historical data available for comparison."}
    
    df = pd.read_csv(data_path)
    details = state['property_details']
    area = details.get('area', 5000)
    beds = details.get('bedrooms', 3)
    
    # Filter for comparable
    similar = df[
        (df['bedrooms'] == beds) & 
        (df['area'] >= area * 0.8) & 
        (df['area'] <= area * 1.2)
    ]
    
    if len(similar) == 0:
        return {"comparable_summary": f"Could not find any properties with {beds} beds and around {area} sqft in the historical database."}
    
    avg_price = similar['price'].mean()
    min_price = similar['price'].min()
    max_price = similar['price'].max()
    count = len(similar)
    
    comp_summary = (
        f"Found {count} comparable properties in the historical dataset "
        f"({beds} beds, {area * 0.8:.0f} - {area * 1.2:.0f} sqft). "
        f"Average historic price: ₹{avg_price:,.0f}. "
        f"Range: ₹{min_price:,.0f} to ₹{max_price:,.0f}."
    )
    return {"comparable_summary": comp_summary}

def generate_recommendations(state: AgentState):
    """Use an LLM to formulate the final structured Markdown report."""
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.2)
    
    system_prompt = (
        "You are an expert Real Estate AI Advisor. Your task is to produce a structured, professional "
        "Advisory Report in Markdown format.\n\n"
        "The report MUST include:\n"
        "- Property Summary\n"
        "- Price Prediction Interpretation (evaluate if the predicted price makes sense compared to comparables)\n"
        "- Market Trend Insights (injecting context from provided market data)\n"
        "- Recommended Actions (for buyers or investors)\n"
        "- Supporting Sources or References\n"
        "- Legal and financial disclaimers (e.g. 'Not financial advice, always consult a professional.')\n\n"
        "Be professional, analytical, and objective. AVOID unsupported financial claims!"
    )
    
    prompt = (
        f"Property Details: {state['property_details']}\n"
        f"Predicted Price: ₹{state['predicted_price']:,.0f}\n"
        f"Model R²: {state['model_metrics'].get('model_r2', 'N/A')}\n\n"
        f"Market Context:\n{state['market_context']}\n\n"
        f"Comparable Properties:\n{state['comparable_summary']}"
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    return {"report": response.content}

# Build LangGraph
workflow = StateGraph(AgentState)

workflow.add_node("analyze", analyze_input)
workflow.add_node("retrieve", retrieve_market_data)
workflow.add_node("compare", compare_properties)
workflow.add_node("generate", generate_recommendations)

workflow.add_edge(START, "analyze")
workflow.add_edge("analyze", "retrieve")
workflow.add_edge("retrieve", "compare")
workflow.add_edge("compare", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

def run_advisory_agent(property_details: dict, predicted_price: float, model_metrics: dict) -> str:
    """Entry point to run the agent from Streamlit."""
    from dotenv import load_dotenv
    load_dotenv()
    
    initial_state = {
        "property_details": property_details,
        "predicted_price": predicted_price,
        "model_metrics": model_metrics,
        "market_context": "",
        "comparable_summary": "",
        "analysis": "",
        "report": ""
    }
    
    final_state = app.invoke(initial_state)
    return final_state["report"]

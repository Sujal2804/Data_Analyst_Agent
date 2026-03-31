import os
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph


load_dotenv()  
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# ---------------------------
# LLM Setup
llm = ChatGroq(model="llama-3.1-8b-instant")

# ---------------------------
# Nodes

def clean_data(state):
    df = state["data"]
    df = df.dropna()
    return {"data": df}

def analyze_data(state):
    df = state["data"]
    summary = df.describe().to_string()
    return {"analysis": summary, "data": df}

def generate_insights(state):
    analysis = state.get("analysis", "")

    prompt = f"Give key insights from this data:\n{analysis}"

    try:
        response = llm.invoke(prompt)
        insights = response.content
    except Exception as e:
        insights = f"Error: {str(e)}"

    return {
        "data": state.get("data"),
        "analysis": analysis,
        "insights": insights
    }

def suggest_model(state):
    analysis = state.get("analysis", "")

    prompt = f"""
    Based on this dataset summary, suggest the best machine learning approach:
    {analysis}
    """

    response = llm.invoke(prompt)

    return {
        "analysis": analysis,
        "insights": state.get("insights"),
        "model": response.content
    }
# ---------------------------
# Build Graph

builder = StateGraph(dict)

builder.add_node("clean", clean_data)
builder.add_node("analyze", analyze_data)
builder.add_node("insight", generate_insights)
builder.add_node("model", suggest_model)

builder.set_entry_point("clean")

builder.add_edge("clean", "analyze")
builder.add_edge("analyze", "insight")
builder.add_edge("insight", "model")

graph = builder.compile()

# ---------------------------
# Main function (IMPORTANT)

def run_agent(df):
    result = graph.invoke({"data": df})
    return result
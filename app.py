import streamlit as st
import pandas as pd

from backend import run_agent

# ---------------------------
# Page Config
st.set_page_config(page_title="Data Analyst Agent", layout="wide")

st.title("📊 Data Analyst Agent (LangGraph + Groq)")

# ---------------------------
# File Upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    # ---------------------------
    # Analyze Button
    if st.button("Analyze Data"):
        with st.spinner("Analyzing data..."):
            result = run_agent(df)

        # ---------------------------
        # DEBUG (optional - remove later)
        # st.write("DEBUG RESULT:", result)

        # ---------------------------
        # Insights Section
        st.subheader("📊 Insights")
        st.write(result.get("insights", "No insights generated"))

        # ---------------------------
        # Model Suggestion
        st.subheader("🤖 Model Suggestion")
        st.write(result.get("model", "No model suggestion available"))

else:
    st.info("👆 Please upload a CSV file to begin.")
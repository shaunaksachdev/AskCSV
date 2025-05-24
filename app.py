import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Page setup
st.set_page_config(page_title="AskCSV - LLM-Powered CSV Assistant", layout="wide")

# Title section
st.markdown("<h1 style='text-align: center;'>AskCSV</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Your AI-powered assistant for exploring CSV files with natural language.</h4>", unsafe_allow_html=True)
st.markdown("---")

# 📘 About the app
with st.expander("What is AskCSV?"):
    st.write("""
    **AskCSV** is an AI-powered CSV data assistant that lets you explore and analyze datasets using natural language.

    Instead of writing code or formulas, you can ask questions like:
    - “What are the top-selling products in Q2?”
    - “Show me a bar chart of sales by region.”

    Under the hood, AskCSV uses OpenAI's GPT model with PandasAI to:
    - Parse your CSV structure
    - Translate questions into Python code using `pandas`
    - Generate charts using `matplotlib` or `plotly`
    - Return clear insights from your data

    All within this simple, clean Streamlit interface.
    """)

# File upload
uploaded_file = st.file_uploader("Upload your CSV file here:", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Layout: Two columns
    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

    with right_col:
        st.subheader("AskCSV Chat")
        user_input = st.text_input("Type your question:")

        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Display previous chat history (last 5)
        for i, (user_q, bot_a) in enumerate(reversed(st.session_state.chat_history[-5:])):
            with st.chat_message("user"):
                st.markdown(f"**You:** {user_q}")
            with st.chat_message("assistant"):
                st.markdown(f"**Assistant:** {bot_a}")

        if user_input:
            # Initialize LLM and SmartDataframe
            llm = OpenAI(api_token=openai_api_key)
            sdf = SmartDataframe(df, config={"llm": llm, "verbose": True})

            try:
                bot_response = sdf.chat(user_input)
            except Exception as e:
                bot_response = f"⚠️ Error: {str(e)}"

            # Save and display chat
            st.session_state.chat_history.append((user_input, bot_response))

            with st.chat_message("user"):
                st.markdown(f"**You:** {user_input}")
            with st.chat_message("assistant"):
                st.markdown(f"**Assistant:** {bot_response}")

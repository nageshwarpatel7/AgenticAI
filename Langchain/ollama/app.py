import os
from dotenv import load_dotenv
from langchain_perplexity import ChatPerplexity
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()
os.environ['PPLX_API_KEY'] = os.getenv('PPLX_API_KEY')

# Define prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the question asked."),
    ("user", "Question:{question}")
])

# Streamlit page setup
st.set_page_config(page_title="Langchain Demo with Perplexity", page_icon="🦙", layout="centered")
st.title("🦙 Langchain Demo with Perplexity")

# Example questions (short prompts)
examples = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Summarize the latest news in AI.",
    "How does photosynthesis work?",
]

# Dropdown for examples
selected_example = st.selectbox("Choose an example question (optional):", ["", *examples])

# User input (overrides example if both provided)
user_input = st.text_input("Or type your own question:")

# Final input: user > example
final_question = user_input if user_input.strip() else selected_example

try:
    # Initialize LLM
    llm = ChatPerplexity(model='sonar-pro')
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    # Generate response if input exists
    if final_question:
        with st.spinner("Thinking..."):
            response = chain.invoke({"question": final_question})
            st.subheader("🤖 Answer")
            st.write(response)

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.warning("Please ensure your Perplexity API key is valid and the model is accessible.")

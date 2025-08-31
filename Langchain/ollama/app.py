import os
from dotenv import load_dotenv

from langchain_ollama import OllamaLLM
import streamlit as st 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser 
from langchain_perplexity import ChatPerplexity

# Load environment variables from .env file
load_dotenv()

# Langsmith Tracking (optional, but good practice)
# Ensure these variables are set in your .env file
os.environ['PPLX_API_KEY'] = os.getenv('PPLX_API_KEY')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')

## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the question asked."),
        ("user", "Question:{question}") 
    ]
)

## Streamlit framework setup
st.set_page_config(page_title="Langchain Demo with Perplexity", page_icon="🦙", layout="centered")
st.title("🦙 Langchain Demo with Perplexity")
input_text = st.text_input("What question do you have in mind?")


## Ollama Llama3 model initialization
try:
    llm = ChatPerplexity(model='sonar-pro')
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser

    # Display the response if user provides input
    if input_text:
        with st.spinner("Thinking..."):
            response = chain.invoke({"question": input_text})
            st.write(response)

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.warning("Please ensure the Ollama application is running and the 'llama3' model is installed. You can run 'ollama run llama3' in your terminal.")
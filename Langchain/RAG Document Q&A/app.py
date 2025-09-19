import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# Note: OllamaEmbeddings is imported but not used in this script.
# from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()
import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

## load the GROQ API key
os.environ['GROQ_API_KEY'] =os.getenv("GROQ_API_KEY")

groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(groq_api_key=groq_api_key, model_name = 'Gemma2-9b-It')

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Question:{input}
    """
    
)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        # Make sure you have a GEMINI_API_KEY in your .env file
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004",google_api_key=os.environ['GEMINI_API_KEY'])
        st.session_state.loader = PyPDFDirectoryLoader('research_papers') ## Data Ingestion step
        st.session_state.docs = st.session_state.loader.load()  ## Document loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:100])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)
        
st.title("RAG Document Q&A With Groq and Gemini")
user_prompt = st.text_input("Enter your query from the research papers")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector Database is ready")
    
if user_prompt:
    if "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm,prompt)
        retriever =st.session_state.vectors.as_retriever()
        retriever_chain =create_retrieval_chain(retriever,document_chain)
        
        start = time.process_time()
        response= retriever_chain.invoke({'input':user_prompt})
        print(f'Response time: {time.process_time()-start}')
        
        st.write(response['answer'])
        
        ## With a streamlit expander
        with st.expander("Document Similarity search"):
            for i,doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write('---------------------------------------')
    else:
        st.warning("Please create the document embeddings first.")


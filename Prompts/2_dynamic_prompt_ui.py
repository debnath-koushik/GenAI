from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B",
    task="text-generation",
    max_new_tokens=100
)

st.header('Research Tool')
user_input = st.text_input('Enter your prompt')

if st.button('Submit'):
    result = llm.invoke(user_input)
    st.write(result)

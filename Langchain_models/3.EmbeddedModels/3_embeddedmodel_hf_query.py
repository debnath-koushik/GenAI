from langchain_huggingface import HuggingFaceEmbeddings
import os

# used this code to change default download location for store hugging face model data
os.environ['HF_HOME'] = 'D:/Koushik/huggingface_cache'

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = "What is the capital of India"

vector = embedding.embed_query(text)

print(vector)
from transformers import pipeline
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

# used this code to change default download location for store hugging face model data
os.environ['HF_HOME'] = 'D:/Koushik/huggingface_cache'

pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=100,
    temperature=0.5
)

llm = HuggingFacePipeline(pipeline=pipe)

prompt = "<|user|>\nWhat is the capital of India?\n<|assistant|>"

result = llm.invoke(prompt)

print(result)

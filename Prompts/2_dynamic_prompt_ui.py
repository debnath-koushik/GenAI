from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt
from transformers import pipeline

load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="google/gemma-7b",
#     task="text-generation",
#     max_new_tokens=100
# )

llm = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    # max_new_tokens=150
)

st.header('Research Tool')
paper_input = st.selectbox("Select Research Paper Name",
                           ["Select...", "Attention Is All You Need",
                            "BERT: Pre-training of Deep Bidirectional Transformers",
                            "GPT-3: Language Models are Few-Shot Learners",
                            "Diffusion Models Beat GANs on Image Synthesis"])

style_input = st.selectbox("Select Explanation Style",
                           ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"])

length_input = st.selectbox("Select Explanation Length",
                            ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)",
                             "Long (detailed explanation)"])

# fills the placehoders in the template with the user input
# prompt = template.invoke({
#     'paper_input': paper_input,
#     'style_input': style_input,
#     'length_input': length_input
# })

template = load_prompt('prompt_template.json')

prompt = template.format(
    paper_input=paper_input,
    style_input=style_input,
    length_input=length_input
)

if st.button('Submit'):
    result = llm(prompt)
    # print(result)
    # st.write(result)
    st.write(result[0]['generated_text'].replace(prompt, ""))
    print(result)

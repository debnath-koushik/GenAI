from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

dict = ["Delhi is the capital of India",
        "Kolkata is the capital of West bengal"]

vector = embedding.embed_documents(dict)

print(vector)

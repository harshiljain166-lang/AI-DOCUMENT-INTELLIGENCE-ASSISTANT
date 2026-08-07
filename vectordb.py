from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv

import pickle


# Load API key from .env
load_dotenv()
load_dotenv(override=True)


# Groq LLM
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7
)


# Load chunks
with open("chunks.pkl", "rb") as file:
    data = pickle.load(file)

chunks = data["chunks"]


# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)


# Create Vector Database
vector_db = Chroma.from_texts(
    texts=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)


print("Vector DB created successfully")


# Query loop
while True:

    query = input("\nEnter your query (type 'break' to exit): ")

    if query.lower() == "break":
        print("Exiting...")
        break


    # Similarity Search
    results = vector_db.similarity_search(
        query,
        k=3
    )


    # Combine retrieved chunks
    context = "\n\n".join(
        [doc.page_content for doc in results]
    )


    # Prompt
    prompt = f"""
You are a helpful AI assistant.

Answer using only the context below.

Context:
{context}

Question:
{query}

Answer:
"""


    # LLM generation
    response = model.invoke(prompt)


    print("\n========== AI ANSWER ==========")
    print(response.content)


    print("\n========== SOURCES ==========")

    for i, doc in enumerate(results, start=1):
        print(f"\nSource {i}")
        print(doc.page_content[:300])



# # Load chunks
# with open("chunks.pkl", "rb") as file:
#     data = pickle.load(file)

# chunks = data["chunks"]


# # Load embedding model
# embedding_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-mpnet-base-v2"
# )


# # Create Vector Database
# vector_db = Chroma.from_texts(
#     texts=chunks,
#     embedding=embedding_model,
#     persist_directory="./chroma_db"
# )


# print("Vector DB created successfully")


# # Search
# query = "What is machine learning?"

# results = vector_db.similarity_search(
#     query,
#     k=3
# )


# for i, doc in enumerate(results):
#     print("\nResult", i+1)
#     print(doc.page_content)
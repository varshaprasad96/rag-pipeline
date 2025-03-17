from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np
import faiss

# Retriever: Pre-trained SentenceTransform
retriever_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
documents = [
    "The Eiffel Tower is in Paris.",
    "The Great Wall of China is a historic fortification.",
    "Python is a programming language.",
    "SpaceX is a private aerospace company."
]
doc_embeddings = retriever_model.encode(documents)
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(np.array(doc_embeddings))

# Generator: Pre-trained GPT-2
generator = pipeline("text-generation", model="gpt2", tokenizer="gpt2")


def rag_pipeline_execute(query, top_k=1, temperature=1.0):
    # Retrieve top k documents
    query_embedding = retriever_model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    retrieved_docs = [documents[i] for i in indices[0]]

    # Generate response using retrieved documents
    context = " ".join(retrieved_docs)
    generated = generator(context, max_length=50, temperature=temperature, num_return_sequences=1)
    return generated[0]['generated_text']

# Example query
# query = "Tell me about the Eiffel Tower."
# print(query)
# response = rag_pipeline(query, top_k=2, temperature=0.7)
# print(f"Query: {query}")
# print(f"Response: {response}")


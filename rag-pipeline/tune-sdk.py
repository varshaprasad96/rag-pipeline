from nltk.translate.bleu_score import SmoothingFunction

from training import evaluate


def objective(parameters):
    # Import all dependencies inside the function
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    # Define the RAG pipeline within the function
    def rag_pipeline_execute(query, top_k, temperature):
        """Retrieves relevant documents and generates a response using GPT-2."""

        # Initialize retriever
        retriever_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

        # Sample documents
        documents = [
            "The Eiffel Tower is in Paris.",
            "The Great Wall of China is a historic fortification.",
            "Python is a programming language.",
            "SpaceX is a private aerospace company."
        ]

        # Encode documents
        doc_embeddings = retriever_model.encode(documents)
        index = faiss.IndexFlatL2(doc_embeddings.shape[1])
        index.add(np.array(doc_embeddings))

        # Encode query and retrieve top-k documents
        query_embedding = retriever_model.encode([query])
        distances, indices = index.search(query_embedding, top_k)
        retrieved_docs = [documents[i] for i in indices[0]]

        # Generate response using GPT-2
        generator = pipeline("text-generation", model="gpt2", tokenizer="gpt2")
        context = " ".join(retrieved_docs)
        generated = generator(context, max_length=50, temperature=temperature, num_return_sequences=1)

        return generated[0]["generated_text"]

    # Define query and ground truth
    query = "Tell me about the Eiffel Tower."
    ground_truth = "The Eiffel Tower is a famous landmark in Paris."

    # Extract hyperparameters
    top_k = int(parameters["top_k"])
    temperature = float(parameters["temperature"])

    # Generate response
    response = rag_pipeline_execute(query, top_k, temperature)

    # Compute BLEU score
    reference = [ground_truth.split()]  # Tokenized reference
    candidate = response.split()  # Tokenized candidate response
    smoothie = SmoothingFunction().method1
    bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothie)

    # Return BLEU score
    print(f"BLEU={bleu_score}")


import kubeflow.katib as katib

# [2] Create hyperparameter search space.
parameters = {
    "top_k": katib.search.int(min=10, max=20),
    "temperature": katib.search.double(min=0.5, max=1.0, step=0.1)
}

# [3] Create Katib Experiment with 12 Trials and 2 CPUs per Trial.
katib_client = katib.KatibClient(namespace="kubeflow")

name = "rag-tuning-experiment"
katib_client.tune(
    name="rag-tuning-experiment",
    objective=objective,
    parameters=parameters,
    algorithm_name="grid",
    objective_metric_name="BLEU",
    objective_type="maximize",
    objective_goal=0.8,
    max_trial_count=10,
    parallel_trial_count=2,
    resources_per_trial={"cpu": "1", "memory": "2Gi"},
    base_image="python:3.10-slim",
    packages_to_install=["transformers==4.36.0", "sentence-transformers==2.2.2", "faiss-cpu==1.7.4", "numpy==1.23.5",
                         "huggingface_hub==0.20.0", "nltk==3.9.1"]
)

# [4] Wait until Katib Experiment is complete
katib_client.wait_for_experiment_condition(name=name)

# [5] Get the best hyperparameters.
print(katib_client.get_optimal_hyperparameters(name))

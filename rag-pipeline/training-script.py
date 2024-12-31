import argparse

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rag import rag_pipeline_execute


# Simulated RAG pipeline (simplified for example)
def rag_pipeline(query, top_k, temperature):
    return rag_pipeline_execute(query, top_k=top_k, temperature=temperature)


# Evaluate BLEU score
def evaluate(query, ground_truth, top_k, temperature):
    # Get the RAG pipeline response
    response = rag_pipeline(query, top_k, temperature)

    # Tokenize the response and ground truth
    reference = [ground_truth.split()]  # Reference should be a list of tokens
    candidate = response.split()  # Candidate is the generated response tokens

    # Apply smoothing to the BLEU score
    smoothie = SmoothingFunction().method1  # Use method1 for smoothing
    bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothie)

    return bleu_score


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate BLEU score for a query using RAG pipeline")
    parser.add_argument("--top_k", type=int, required=True, help="Number of top documents to retrieve")
    parser.add_argument("--temperature", type=float, required=True, help="Temperature for the generator")
    args = parser.parse_args()

    # Simulate a single query
    query = "Tell me about the Eiffel Tower."
    ground_truth = "The Eiffel Tower is a famous landmark in Paris."

    # Call evaluate with arguments from the command line
    bleu_score = evaluate(query, ground_truth, args.top_k, args.temperature)
    print(f"BLEU score: {bleu_score}")

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from vector_storage import store_embeddings_in_faiss
from rag_agent import retrieve_documents

# Function to generate a response based on the query and retrieved documents
def generate_response(query, retrieved_docs):
    # Load GPT-Neo model for generation
    model_name = "EleutherAI/gpt-neo-2.7B"  # Example GPT-Neo model
    llm_model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Combine the query and retrieved documents as context for the model
    context = f"Query: {query}\n"
    context += "\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(retrieved_docs)])

    # Tokenize the combined input
    inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=1024)

    # Generate the response
    output = llm_model.generate(inputs['input_ids'], max_length=200, num_return_sequences=1)

    # Decode the output
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

# Main function to handle document retrieval and response generation
def main():
    query = "What is deep learning?"  # Example query

    # Retrieve documents based on the query using FAISS and SentenceTransformer
    retrieved_docs = retrieve_documents(query)

    # Generate a response based on the query and retrieved documents
    answer = generate_response(query, retrieved_docs)

    print(f"Query: {query}")
    print(f"Answer: {answer}")

if __name__ == '__main__':
    main()

from transformers import AutoModelForCausalLM, AutoTokenizer
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Function to retrieve documents based on a query
def retrieve_documents(query):
    # Load the FAISS index
    index = faiss.read_index("faiss_index.index")

    # Load the stored documents
    with open("documents.txt", "r") as f:
        documents = f.readlines()

    # Load SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Encode the query into an embedding
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')

    # Perform the query to get the top 2 results
    distances, indices = index.search(query_embedding, k=2)

    # Retrieve the documents corresponding to the indices
    retrieved_docs = []
    for i, idx in enumerate(indices[0]):
        retrieved_docs.append(documents[idx].strip())

    print(f"Query: {query}")
    print("Retrieved Documents:")
    for doc in retrieved_docs:
        print(f"Document: {doc}")

    return retrieved_docs

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
    print(f"Answer: {answer}")

# Main function
def main():
    query = "What is deep learning?"

    # Retrieve documents based on the query
    retrieved_docs = retrieve_documents(query)

    # Generate a response based on the query and retrieved documents
    generate_response(query, retrieved_docs)

if __name__ == '__main__':
    main()

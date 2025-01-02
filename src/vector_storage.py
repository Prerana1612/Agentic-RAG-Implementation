import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Function to store embeddings in FAISS index
def store_embeddings_in_faiss(embeddings, documents):
    # Convert embeddings to numpy array (FAISS requires this format)
    embeddings = np.array(embeddings).astype('float32')

    # Create a FAISS index for cosine similarity
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance (Euclidean)

    # Add embeddings to the index
    index.add(embeddings)
    print(f"Stored {len(documents)} documents in FAISS index.")

    # Save the FAISS index and document references for later retrieval
    faiss.write_index(index, "faiss_index.index")
    with open("documents.txt", "w") as f:
        for doc in documents:
            f.write(f"{doc}\n")

# Main function
def main():
    documents = [
        "Machine learning is a field of artificial intelligence.",
        "Deep learning is a subset of machine learning that uses neural networks."
    ]

    # Load SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Encode the documents into embeddings
    embeddings = model.encode(documents)

    # Store embeddings in FAISS
    store_embeddings_in_faiss(embeddings, documents)

if __name__ == '__main__':
    main()

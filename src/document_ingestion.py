import os
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load documents from the data folder
def load_documents(data_path):
    documents = []
    for filename in os.listdir(data_path):
        if filename.endswith('.txt'):
            with open(os.path.join(data_path, filename), 'r', encoding='utf-8') as file:
                documents.append(file.read())
    return documents

# Preprocess documents: Tokenize and clean
def preprocess_documents(documents):
    # Example of basic preprocessing: you can extend it based on your requirements
    cleaned_documents = [doc.replace('\n', ' ').strip() for doc in documents]
    return cleaned_documents

# Embed documents using a pre-trained model
def embed_documents(documents):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(documents, convert_to_tensor=True)
    return embeddings

# Main function to load, preprocess, and embed documents
def main():
    data_path = 'D:/Take-Home-Assessment(Cloud_Ambassadors)/data'  # Change to the correct path for your data
    documents = load_documents(data_path)
    cleaned_documents = preprocess_documents(documents)
    embeddings = embed_documents(cleaned_documents)
    return cleaned_documents, embeddings

if __name__ == '__main__':
    cleaned_documents, embeddings = main()
    print(f"Processed {len(cleaned_documents)} documents.")

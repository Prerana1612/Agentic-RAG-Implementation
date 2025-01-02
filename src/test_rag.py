from reasoning_agent import chain_of_thought  # Assuming you have a chain_of_thought function
import pytest

def test_agent():
    # Sample queries
    queries = [
        "What is deep learning?",
        "Explain the concept of neural networks",
        "How do transformers work in NLP?"
    ]
    
    for query in queries:
        print(f"Query: {query}")
        response = chain_of_thought(query)
        print(f"Response: {response}")
        assert len(response) > 0  # Ensure that the response is not empty

if __name__ == "__main__":
    test_agent()

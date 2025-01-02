from transformers import pipeline
import re

# Initialize the text generation pipeline with GPT-2 model and tokenizer
generator = pipeline('text-generation', model='gpt2', tokenizer='gpt2')

# Define the query for which the model will generate the answer
query = "What is deep learning?"

# Function to post-process the generated text
def post_process_answer(answer):
    """
    Post-processes the generated answer by removing excess repetition and fixing common issues.
    """
    # Remove excessive repetition (like the same sentence being repeated)
    answer = re.sub(r"(\b\w+\b)(?=.*\1)", "", answer)

    # Remove multiple spaces and ensure single spaces between words
    answer = re.sub(r'\s+', ' ', answer)

    # Optionally, trim any trailing spaces or punctuation that seems out of place
    answer = answer.strip()

    # Capitalize the first letter of the answer for proper sentence structure
    if answer:
        answer = answer[0].upper() + answer[1:]

    return answer

# Define the chain_of_thought function that can be imported in the test file
def chain_of_thought(query):
    """
    Wrapper function to generate a response using the pipeline and return the processed result.
    """
    result = generator(query, 
                       max_length=100,  # Limit output length to 100 tokens
                       num_return_sequences=1,  # Generate only one answer
                       temperature=0.6,  # Set temperature to 0.6 for more deterministic output
                       top_p=0.9,  # Use nucleus sampling (top_p) for diversity
                       top_k=50,  # Set top_k to 50 to restrict sampling from top 50 tokens
                       truncation=True,  # Ensure long responses are truncated
                       no_repeat_ngram_size=2)  # Prevent repeating n-grams

    # Extract and clean the generated answer
    raw_answer = result[0]['generated_text']
    cleaned_answer = post_process_answer(raw_answer)

    return cleaned_answer

from openai import OpenAI
import os

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---- Embeddings ----
# Generate embeddings for a single text input
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="The food was delicious and the waiter..."
)

print(f"Embedding dimension: {len(response.data[0].embedding)}")
print(f"First 5 values: {response.data[0].embedding[:5]}")

# Generate embeddings for multiple inputs
response_batch = client.embeddings.create(
    model="text-embedding-3-small",
    input=[
        "The food was delicious and the waiter...",
        "The quick brown fox jumped over the lazy dog."
    ]
)

print(f"\nBatch embeddings count: {len(response_batch.data)}")
print(f"First embedding dimension: {len(response_batch.data[0].embedding)}")
print(f"Second embedding dimension: {len(response_batch.data[1].embedding)}")

# Using a larger model
response_large = client.embeddings.create(
    model="text-embedding-3-large",
    input="The food was delicious and the waiter..."
)
print(f"\nLarge model embedding dimension: {len(response_large.data[0].embedding)}")

import json
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()

Settings.embed_model = OpenAIEmbedding(model='text-embedding-3-small')
Settings.llm = OpenAI(model='gpt-4o-mini')

# Load data
with open('../../data/synthetic_tickets.json', 'r', encoding='utf-8') as f:
    tickets = json.load(f)

documents = [
    Document(
        text=f"Title: {t['title']}\nDescription: {t['description']}",
        metadata={'ticket_id': t['ticket_id'], 'category': t['category']}
    )
    for t in tickets
]

# Step 1: Build and save
print("Building index...")
vector_index = VectorStoreIndex.from_documents(documents)
vector_index.storage_context.persist(persist_dir="./my_saved_index")
print("✓ Saved to ./my_saved_index")

# Step 2: Load from disk
print("\nLoading index...")
storage_context = StorageContext.from_defaults(persist_dir="./my_saved_index")
loaded_index = load_index_from_storage(storage_context)
print("✓ Loaded from disk")

# Step 3: Test it works
query = "login problem"
response = loaded_index.as_query_engine().query(query)
print(f"\nQuery: '{query}'")
print(f"Result: {response}")
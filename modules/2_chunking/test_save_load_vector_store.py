## Exercise 8: Save and Load Vector Store (Medium)

# **Task**: Persist a vector store and reload it.

# **Copy and run this complete code**:

import json
import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# Load data
with open('../../data/synthetic_tickets.json', 'r') as f:
    tickets = json.load(f)

documents = [
    Document(
        page_content=f"{t['title']}. {t['description']}",
        metadata={'ticket_id': t['ticket_id'], 'category': t['category']}
    )
    for t in tickets
]

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

# Step 1: Build and save (Chroma persists with persist_directory)
print("Building vector store...")
store = Chroma.from_documents(
    documents, 
    embeddings, 
    collection_name="my_collection",
    persist_directory="./my_chroma_db"
)
print("✓ Saved to ./my_chroma_db")

# Step 2: Load it back
print("\nLoading vector store...")
loaded_store = Chroma(
    persist_directory="./my_chroma_db",
    embedding_function=embeddings,
    collection_name="my_collection"
)
print("✓ Loaded from disk")

# Step 3: Verify it works
query = "login problem"
results = loaded_store.similarity_search(query, k=3)
print(f"\nSearch results for '{query}':")
for doc in results:
    print(f"  {doc.metadata['ticket_id']}: {doc.page_content[:50]}...")
import json
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, KeywordTableIndex, Document, Settings
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

# Build both indexes
vector_index = VectorStoreIndex.from_documents(documents)
keyword_index = KeywordTableIndex.from_documents(documents)

query = "authentication timeout error"

# Get retrievers
vector_retriever = vector_index.as_retriever(similarity_top_k=5)
keyword_retriever = keyword_index.as_retriever()

# Retrieve from both
print(f"Query: '{query}'\n")

print("Vector Results:")
vector_nodes = vector_retriever.retrieve(query)
for i, node in enumerate(vector_nodes[:3], 1):
    print(f"  {i}. {node.node.metadata.get('ticket_id', 'N/A')}")

print("\nKeyword Results:")
keyword_nodes = keyword_retriever.retrieve(query)
for i, node in enumerate(keyword_nodes[:3], 1):
    print(f"  {i}. {node.node.metadata.get('ticket_id', 'N/A')}")

# Simple hybrid: combine and deduplicate
seen = set()
hybrid_results = []
for node in vector_nodes + keyword_nodes:
    ticket_id = node.node.metadata.get('ticket_id')
    if ticket_id and ticket_id not in seen:
        seen.add(ticket_id)
        hybrid_results.append(ticket_id)

print(f"\nHybrid Results (combined): {hybrid_results[:5]}")
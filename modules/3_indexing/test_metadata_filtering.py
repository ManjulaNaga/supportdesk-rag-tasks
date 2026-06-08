# **Task**: Filter search results by category.

# **Copy and complete this code** (the filter is already filled in):

import json
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
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
        metadata={'ticket_id': t['ticket_id'], 'category': t['category'], 'priority': t['priority']}
    )
    for t in tickets
]

# Build index
vector_index = VectorStoreIndex.from_documents(documents)

# Query WITHOUT filter
print("Without filter:")
response = vector_index.as_query_engine(similarity_top_k=3).query("system problem")
print(f"  {response}\n")

# Query WITH category filter
print("With 'Authentication' filter:")
filters = MetadataFilters(filters=[
    ExactMatchFilter(key="category", value="Authentication")
])
filtered_engine = vector_index.as_query_engine(similarity_top_k=3, filters=filters)
filtered_response = filtered_engine.query("system problem")
print(f"  {filtered_response}")
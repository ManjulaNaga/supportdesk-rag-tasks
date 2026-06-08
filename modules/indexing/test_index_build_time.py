import json
import time
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SummaryIndex, KeywordTableIndex, Document, Settings
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

print(f"Building indexes for {len(documents)} documents...\n")

# Vector Index
start = time.time()
vector_index = VectorStoreIndex.from_documents(documents)
vector_time = time.time() - start
print(f"Vector Index: {vector_time:.2f}s")

# Keyword Index
start = time.time()
keyword_index = KeywordTableIndex.from_documents(documents)
keyword_time = time.time() - start
print(f"Keyword Index: {keyword_time:.2f}s")

# Summary Index (note: doesn't pre-build, so fast to create)
start = time.time()
summary_index = SummaryIndex.from_documents(documents)
summary_time = time.time() - start
print(f"Summary Index: {summary_time:.2f}s")

print(f"\n→ Vector Index takes longer because it generates embeddings for all documents")
print(f"→ Keyword Index uses LLM to extract keywords from each document")
print(f"→ Summary Index is just storing documents (work happens at query time)")
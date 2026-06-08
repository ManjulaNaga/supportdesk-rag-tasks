import json
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SummaryIndex, TreeIndex, KeywordTableIndex, Document, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()

# Configure LlamaIndex
Settings.embed_model = OpenAIEmbedding(model='text-embedding-3-small')
Settings.llm = OpenAI(model='gpt-4o-mini')

# Load data
with open('../../data/synthetic_tickets.json', 'r', encoding='utf-8') as f:
    tickets = json.load(f)

documents = [
    Document(
        text=f"Title: {t['title']}\nDescription: {t['description']}\nResolution: {t['resolution']}",
        metadata={'ticket_id': t['ticket_id'], 'category': t['category']}
    )
    for t in tickets
]

print("Building indexes...")
vector_idx = VectorStoreIndex.from_documents(documents)
keyword_idx = KeywordTableIndex.from_documents(documents)
print("✓ Indexes built\n")

# Compare on 3 queries
test_queries = [
    "authentication login problem",
    "database timeout error",
    "TICK-005"
]

for query in test_queries:
    print("=" * 60)
    print(f"Query: '{query}'")
    print("=" * 60)
    
    # Vector Index
    vec_response = vector_idx.as_query_engine(similarity_top_k=3).query(query)
    print(f"\nVector Index:")
    print(f"  {str(vec_response)[:150]}...")
    
    # Keyword Index
    kw_response = keyword_idx.as_query_engine().query(query)
    print(f"\nKeyword Index:")
    print(f"  {str(kw_response)[:150]}...")
    
    print()
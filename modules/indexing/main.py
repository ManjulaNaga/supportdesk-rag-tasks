# =============================================================================
# IMPORTS
# =============================================================================
import json
import os
import httpx
from dotenv import load_dotenv

# LlamaIndex core components
from llama_index.core import (
    VectorStoreIndex,    # Standard embedding-based index
    SummaryIndex,        # Full document storage, LLM-based relevance
    TreeIndex,           # Hierarchical summarization tree
    KeywordTableIndex,   # Inverted keyword index
    Document,            # Document wrapper with text + metadata
    Settings             # Global configuration
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# =============================================================================
# SETUP: Load Environment Variables
# =============================================================================
load_dotenv()

# Set longer timeout for httpx (used by OpenAI client)
# Some index types make MANY LLM calls and need more time
os.environ["HTTPX_TIMEOUT"] = "300"  # 5 minutes

# =============================================================================
# CONFIGURE LLAMAINDEX SETTINGS
# =============================================================================
#
# LlamaIndex uses a Settings singleton to configure:
#   - embed_model: Which embedding model to use
#   - llm: Which LLM to use for queries and index building
#
# These settings apply globally to all indexes we create.
# =============================================================================
Settings.embed_model = OpenAIEmbedding(
    model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small'),
    api_key=os.getenv('OPENAI_API_KEY'),
    timeout=120,      # 2 min timeout for embedding calls
    max_retries=5     # Retry on failure
)
Settings.llm = OpenAI(
    model=os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini'),
    api_key=os.getenv('OPENAI_API_KEY'),
    timeout=300,      # 5 min timeout (Tree/Keyword indexes are slow!)
    max_retries=5
)

# =============================================================================
# INTRODUCTION
# =============================================================================
print("="*80)
print("MODULE 3: INDEXING STRATEGIES FOR RAG")
print("="*80)
print("\nThis demo compares 5 different indexing approaches:")
print("1. Vector Index - Semantic similarity search (MOST COMMON)")
print("2. Summary Index - Search through full documents with LLM")
print("3. Tree Index - Hierarchical retrieval (summaries → details)")
print("4. Keyword Table Index - Traditional keyword matching")
print("5. Hybrid Retrieval - Combine multiple strategies (PRODUCTION)")

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n" + "="*80)
print("Loading Support Tickets")
print("="*80)

with open('../../data/synthetic_tickets.json', 'r', encoding='utf-8') as f:
    tickets = json.load(f)

# -----------------------------------------------------------------------------
# Convert to LlamaIndex Documents
# -----------------------------------------------------------------------------
# LlamaIndex uses Document objects (similar to LangChain's Document)
# Each Document has:
#   - text: The content to index
#   - metadata: Associated data for filtering/display
# -----------------------------------------------------------------------------
documents = []
for ticket in tickets:
    # Combine all fields into content (rich context for embedding)
    content = f"""Title: {ticket['title']}
Description: {ticket['description']}
Resolution: {ticket['resolution']}
Category: {ticket['category']}
Priority: {ticket['priority']}"""
    
    doc = Document(
        text=content,
        metadata={
            'ticket_id': ticket['ticket_id'],
            'category': ticket['category'],
            'priority': ticket['priority'],
            'title': ticket['title']
        }
    )
    documents.append(doc)

print(f"✓ Loaded {len(documents)} support tickets")

# Test query - we'll use this across all index types
# query = "Database connection is timing out"
# query = "Mobile app crashes on startup"
# query= "Payment processing fails for international cards"
query = "Email notifications not being delivered"

print(f"\nTest Query: '{query}'")

# ============================================================================
# PART 1: Vector Index (Flat Index)
# ============================================================================
vector_index = VectorStoreIndex.from_documents(documents)
vector_query_engine = vector_index.as_query_engine(similarity_top_k=5)
print("✓ Created vector index")
print(f"\nQuery: '{query}'")
vector_response = vector_query_engine.query(query)

print("\nVector Index Results:")
print(f"Answer: {vector_response.response}\n")
print("Source Documents:")
for i, node in enumerate(vector_response.source_nodes, 1):
    print(f"\n{i}. {node.metadata.get('ticket_id', 'Unknown')}")
    print(f"   Score: {node.score:.4f}")  # Similarity score (higher = more similar)
    print(f"   {node.text[:150]}...")

# ============================================================================
# PART 2: Summary Index
# ============================================================================

summary_index = SummaryIndex.from_documents(documents)
summary_query_engine = summary_index.as_query_engine(response_mode="tree_summarize")
print("✓ Created summary index")
print(f"\nQuery: '{query}'")
summary_response = summary_query_engine.query(query)
for i, node in enumerate(summary_response.source_nodes[:3], 1):
    print(f"\n{i}. {node.metadata.get('ticket_id', 'Unknown')}")
    print(f"   {node.text[:150]}...")
# ============================================================================
# PART 3: Tree Index (Hierarchical Retrieval)
# ============================================================================
tree_documents = documents
tree_index = TreeIndex.from_documents(tree_documents)
tree_query_engine = tree_index.as_query_engine(child_branch_factor=4)
print("✓ Created Tree index")
print(f"\nQuery: '{query}'")
tree_response = tree_query_engine.query(query)
print("Source Documents:")
for i, node in enumerate(tree_response.source_nodes[:3], 1):
    print(f"\n{i}. {node.metadata.get('ticket_id', 'Unknown')}")
    print(f"   {node.text[:150]}...")

# ============================================================================
# PART 4: Keyword Table Index
# ============================================================================
keyword_documents = documents
print(f"Building Keyword Index with {len(keyword_documents)} documents...")
keyword_index = KeywordTableIndex.from_documents(keyword_documents)

keyword_query_engine = keyword_index.as_query_engine()
print("✓ Created keyword table index")

keyword_query = "TICK-001"
print(f"\nKeyword-specific query: '{keyword_query}'")
keyword_response = keyword_query_engine.query(keyword_query)
print(f"Result: {keyword_response.response}")

# print(f"\nQuery: '{query}'")
# keyword_response = keyword_query_engine.query(query)

# print("\nKeyword Index Results:")
# print(f"Answer: {keyword_response.response}\n")
# print("Source Documents:")
# for i, node in enumerate(keyword_response.source_nodes[:3], 1):
#     print(f"\n{i}. {node.metadata.get('ticket_id', 'Unknown')}")
#     print(f"   {node.text[:150]}...")



# # ============================================================================
# # PART 5: Hybrid Retrieval
# # ============================================================================
# vector_nodes = vector_index.as_retriever(similarity_top_k=5).retrieve(query)
# keyword_nodes = keyword_index.as_retriever().retrieve(query)
# seen_ids = set()
# hybrid_nodes = []

# for node in vector_nodes + keyword_nodes:
#     node_id = node.metadata.get('ticket_id', node.node_id)
#     if node_id not in seen_ids:
#         seen_ids.add(node_id)
#         hybrid_nodes.append(node)

# # Documents found by BOTH methods are likely most relevant!

# print("\nHybrid Retrieval Results (Combined):")
# for i, node in enumerate(hybrid_nodes[:3], 1):
#     print(f"\n{i}. {node.metadata.get('ticket_id', 'Unknown')}")
#     if hasattr(node, 'score') and node.score:
#         print(f"   Score: {node.score:.4f}")
#     print(f"   {node.text[:150]}...")
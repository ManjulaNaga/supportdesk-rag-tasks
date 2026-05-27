"""
Module 2: Chunking & Vector Stores Demo

"""

import json
import os
from langchain_text_splitters import (  # Various splitting strategies
    RecursiveCharacterTextSplitter,  # Best general-purpose splitter
    CharacterTextSplitter,  # Simple split by character count
    MarkdownHeaderTextSplitter,  # Splits based on markdown headers
    HTMLHeaderTextSplitter  # Splits based on HTML tags
)
from langchain_experimental.text_splitter import SemanticChunker  # AI-powered semantic chunking
from langchain_community.vectorstores import Chroma  # Vector database
from langchain_openai import OpenAIEmbeddings  # OpenAI embedding function
from langchain_core.documents import Document  # Document abstraction
from dotenv import load_dotenv

# ============================================================================
# SETUP: Load environment and data
# ============================================================================

# Load API keys from .env file (never hardcode API keys!)
load_dotenv()

print("="*80)
print("MODULE 2: CHUNKING & VECTOR STORES")
print("="*80)

# Load our support ticket dataset

with open('../../data/synthetic_tickets.json', 'r') as f:
    tickets = json.load(f)
print(f"\nLoaded {len(tickets)} support tickets")

# ============================================================================
# PART 1: Chunking Strategies
# ============================================================================

print("\n" + "="*80)
print("PART 1: Chunking Strategies")
print("="*80)

# -----------------------------------------------------------------------------
# First, convert our ticket data into LangChain Document objects
# Document = page_content (text) + metadata (structured info for filtering)
# -----------------------------------------------------------------------------
documents = []
for ticket in tickets:
    # Combine all ticket fields into a single text block
    # TIP: Include all relevant context that helps understand the document
    full_text = f"""
Ticket ID: {ticket['ticket_id']}
Title: {ticket['title']}
Category: {ticket['category']}
Priority: {ticket['priority']}
Description: {ticket['description']}
Resolution: {ticket['resolution']}
    """.strip()
    
    # Create Document object with metadata
    # Metadata is CRUCIAL - it enables filtering later!
    # Example: "Find similar tickets, but only in the 'Authentication' category"
    doc = Document(
        page_content=full_text,  # The actual text content
        metadata={
            'ticket_id': ticket['ticket_id'],   # For identifying results
            'category': ticket['category'],      # For category filtering
            'priority': ticket['priority']       # For priority filtering
        }
    )
    documents.append(doc)

print(f"Created {len(documents)} documents")
print(f"\nSample document length: {len(documents[0].page_content)} characters")

# =============================================================================
# STRATEGY 1: Fixed-Size Chunking
# =============================================================================
print("\n--- Strategy 1: Fixed-Size Chunking ---")

fixed_splitter = CharacterTextSplitter(
    chunk_size=500,      # Maximum characters per chunk
    chunk_overlap=20,    # Characters to repeat between chunks (10% overlap)
    separator="\n"       # Prefer splitting on newlines when possible
)
fixed_chunks = fixed_splitter.split_documents(documents)

print(f"✓ Created {len(fixed_chunks)} chunks")
print(f"  Chunk size: 200 chars, Overlap: 20 chars")
print(f"  Sample chunk: {fixed_chunks[0].page_content[:100]}...")

# =============================================================================
# STRATEGY 2: Recursive Character Splitting (RECOMMENDED DEFAULT)
# =============================================================================
print("\n--- Strategy 2: Recursive Character Splitting ---")

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=20,      # Max characters per chunk
    chunk_overlap=10,    # 50 char overlap (~17%)
    # Separators tried in ORDER - most specific first!
    separators=[
        "\n\n",  # 1st: Paragraph breaks (best split point)
        "\n",    # 2nd: Line breaks
        ". ",    # 3rd: Sentence boundaries
        " "      # 4th: Word boundaries (last resort for text)
    ]
)
recursive_chunks = recursive_splitter.split_documents(documents)

print(f"✓ Created {len(recursive_chunks)} chunks")
print(f"  Tries to split on paragraph/sentence boundaries")
print(f"  Sample chunk: {recursive_chunks[0].page_content[:100]}...")

# =============================================================================
# STRATEGY 3: Semantic Chunking (Embedding-Based)
# =============================================================================
print("\n--- Strategy 3: Semantic Chunking ---")
print("  Note: Semantic chunking uses embeddings to find natural break points")

# Initialize OpenAI embeddings for semantic chunker
# IMPORTANT: This costs money! Each sentence needs an embedding API call
embeddings_model = OpenAIEmbeddings(
    model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
)

# Demo with a paragraph that has CLEAR topic shifts
# This makes it obvious where semantic chunking will split
demo_paragraph = """
Database performance is critical for application speed. Slow queries can cause timeouts and frustrated users. Adding proper indexes to frequently queried columns dramatically improves response times. Query optimization should be a top priority for any development team.

The weather forecast shows rain expected throughout the weekend. Temperatures will drop to the mid-40s by Sunday evening. Residents should prepare for possible flooding in low-lying areas. Don't forget to bring an umbrella if you're heading out.

Authentication security requires multiple layers of protection. Passwords should be hashed using bcrypt or Argon2. Two-factor authentication adds an essential second layer of defense. Session tokens must be rotated regularly to prevent hijacking. API keys should never be exposed in client-side code.
Temparatures are high next week
"""

print("\n  📝 Demo Text (3 distinct topics):")
print("  " + "-"*70)
print("  Topic 1: Database performance (sentences 1-4)")
print("  Topic 2: Weather forecast (sentences 5-8)")
print("  Topic 3: Authentication security (sentences 9-13)")
print("  " + "-"*70)

semantic_splitter = SemanticChunker(
    embeddings=embeddings_model,
    # How to detect "topic change":
    # - "percentile": Split where similarity is in bottom X percentile
    # - "standard_deviation": Split where similarity is X std devs below mean
    # - "interquartile": Split where similarity is below Q1 - 1.5*IQR
    breakpoint_threshold_type="percentile"
)

demo_doc = Document(page_content=demo_paragraph.strip())
semantic_chunks = semantic_splitter.split_documents([demo_doc])
print(f"\n✓ Created {len(semantic_chunks)} chunks (expected: ~3 for 3 topics)")

# Show each semantic chunk
print("\n  📊 Resulting Semantic Chunks:")
print("  " + "-"*70)
for i, chunk in enumerate(semantic_chunks):
    print(f"\n  Chunk {i+1} ({len(chunk.page_content)} chars):")
    print("  " + "~"*60)
    # Show full content for clarity
    for line in chunk.page_content.strip().split('\n'):
        if line.strip():
            print(f"    {line.strip()}")
    print("  " + "~"*60)

print("\n  ✨ Notice how each chunk contains semantically related sentences!")
print("  The chunker detected topic shifts between database → weather → auth")

# =============================================================================
# STRATEGY 4: Markdown Structure-Aware Splitting
# =============================================================================
print("\n--- Strategy 4: Markdown Header Splitting ---")

# Sample markdown documentation (simulating a knowledge base article)
markdown_doc = """
# Database Troubleshooting Guide

## Connection Issues

### Timeout Errors
If you encounter timeout errors, check the connection string and ensure the database server is reachable.
Increase the connection timeout value in your configuration.

### Authentication Failures
Verify your credentials are correct. Check for expired passwords or locked accounts.
Ensure the user has proper permissions on the database.

## Performance Problems

### Slow Queries
Analyze query execution plans using EXPLAIN.
Consider adding indexes on frequently queried columns.
Review and optimize JOIN operations.

### High CPU Usage
Monitor long-running queries.
Check for missing indexes causing table scans.
"""

# Define which headers to split on
# Format: (header_marker, metadata_key)
headers_to_split_on = [
    ("#", "Header 1"),    # H1 tags
    ("##", "Header 2"),   # H2 tags  
    ("###", "Header 3"),  # H3 tags
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False  # Keep headers in the chunk content (usually want True)
)
md_chunks = markdown_splitter.split_text(markdown_doc)

print(f"✓ Created {len(md_chunks)} chunks from markdown")
print(f"  Preserves document structure and header context")
if md_chunks:
    print(f"  Sample chunk with metadata:")
    print(f"    Content: '/n'{md_chunks[0].page_content[:80]}...")
    print(f"    Metadata: '/n'{md_chunks[0].metadata}")  # Shows header hierarchy!

# =============================================================================
# STRATEGY 5: HTML Structure-Aware Splitting
# =============================================================================

print("\n--- Strategy 5: HTML Header Splitting ---")

# Sample HTML documentation (simulating a scraped help page)
html_doc = """
<!DOCTYPE html>
<html>
<body>
    <h1>Email Configuration Guide</h1>
    
    <h2>SMTP Settings</h2>
    <p>Configure your SMTP server settings in the admin panel. Use port 587 for TLS or port 465 for SSL.</p>
    
    <h3>Common SMTP Servers</h3>
    <p>Gmail: smtp.gmail.com, Outlook: smtp.office365.com, Yahoo: smtp.mail.yahoo.com</p>
    
    <h2>IMAP Configuration</h2>
    <p>Set up IMAP to sync your emails across devices. Use port 993 for secure connections.</p>
    
    <h3>Folder Mapping</h3>
    <p>Map your email folders to the appropriate IMAP folders for proper synchronization.</p>
</body>
</html>
"""

# Map HTML tags to metadata keys
headers_to_split_on_html = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]

html_splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on_html
)
html_chunks = html_splitter.split_text(html_doc)

print(f"✓ Created {len(html_chunks)} chunks from HTML")
print(f"  Respects HTML semantic structure")
if html_chunks:
    print(f"  Sample chunk with metadata:")
    print(f"    Content: {html_chunks[0].page_content[:80]}...")
    print(f"    Metadata: {html_chunks[0].metadata}")

# =============================================================================
# STRATEGY 6: No Chunking (Whole Documents)
# =============================================================================

print("\n--- Strategy 6: Whole Documents (No Chunking) ---")
print(f"✓ Using {len(documents)} whole documents")
print(f"  Good for small documents like our tickets")


# ============================================================================
# PART 2: Chroma Vector Store
# ============================================================================
#
print("\n" + "="*80)
print("PART 2: Chroma Vector Store")
print("="*80)

# Use LangChain's embedding wrapper (handles API calls internally)
embeddings_model = OpenAIEmbeddings(
    model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
)

query = "Mobile app crashes"

print("\nDatabase is timing out frequently")

chroma_store = Chroma.from_documents(
    documents=documents,              # Our LangChain Document objects
    embedding=embeddings_model,       # OpenAI embeddings
    collection_name="support_tickets",# Like a "table" in a database
    persist_directory="./chroma_db"   # Save to disk for persistence
)
print("✓ Chroma store created and persisted")

# -----------------------------------------------------------------------------
# Basic Similarity Search
# -----------------------------------------------------------------------------
print(f"\nSearching in Chroma: '{query}'")
# chroma_results = chroma_store.similarity_search(query, k=3)

# print(f"\nTop {len(chroma_results)} results:")
# for i, doc in enumerate(chroma_results, 1):
#     print(f"\n#{i}")
#     print(f"Ticket: {doc.metadata['ticket_id']}")
#     print(f"Category: {doc.metadata['category']}")

# Use similarity_search_with_score instead
results_with_scores = chroma_store.similarity_search_with_score(query, k=3)

print(f"\nTop {len(results_with_scores)} results:")
for i, (doc, score) in enumerate(results_with_scores, 1):
    print(f"\n#{i} - Distance: {score:.4f}")
    print(f"Ticket: {doc.metadata['ticket_id']}")
    print(f"Category: {doc.metadata['category']}")


# -----------------------------------------------------------------------------
# MMR Search (Maximal Marginal Relevance)
# -----------------------------------------------------------------------------
print("\n--- Using MMR for Diverse Results ---")
mmr_results = chroma_store.max_marginal_relevance_search(query, k=3)

print(f"\nMMR Results (more diverse):")
for i, doc in enumerate(mmr_results, 1):
    print(f"\n#{i}")
    print(f"Ticket: {doc.metadata['ticket_id']}")
    print(f"Title: {tickets[int(doc.metadata['ticket_id'].split('-')[1]) - 1]['title']}")

# ============================================================================
# PART 3: Metadata Filtering
# ============================================================================
#
print("\n" + "="*80)
print("PART 3: Metadata Filtering")
print("="*80)
# -----------------------------------------------------------------------------
# Example 1: Filter by category
# -----------------------------------------------------------------------------
print("\nSearching only in 'Authentication' category:")
filtered_results = chroma_store.similarity_search(
    query,
    k=3,
    filter={"category": "Authentication"}  # Only match this category
)

print(f"\nFiltered results ({len(filtered_results)}):")
for i, doc in enumerate(filtered_results, 1):
    print(f"\n#{i}")
    print(f"Ticket: {doc.metadata['ticket_id']}")
    print(f"Category: {doc.metadata['category']}")
    print(f"Content: {doc.page_content[:100]}...")

# -----------------------------------------------------------------------------
# Example 2: Filter by priority
# -----------------------------------------------------------------------------

print("\n\nSearching only 'High' priority tickets:")
high_priority_results = chroma_store.similarity_search(
    "Database performance issues",
    k=3,
    filter={"priority": "High"}  # Only high priority
)

print(f"\nHigh priority results ({len(high_priority_results)}):")
for i, doc in enumerate(high_priority_results, 1):
    print(f"\n#{i}")
    print(f"Ticket: {doc.metadata['ticket_id']}")
    print(f"Priority: {doc.metadata['priority']}")
# ============================================================================
# PART 4: Comparing Chunking Strategies
# ============================================================================
print("\n" + "="*80)
print("PART 4: Evaluating Chunking Strategies")
print("="*80)

# Build stores with different chunking
print("\nBuilding vector stores with different chunking strategies...")

# Store 1: Whole documents (no chunking)
store_whole = Chroma.from_documents(
    documents=documents,
    embedding=embeddings_model,
    collection_name="whole_docs"
)

# Store 2: Fixed-size chunks (may split mid-sentence)
store_fixed = Chroma.from_documents(
    documents=fixed_chunks,
    embedding=embeddings_model,
    collection_name="fixed_chunks"
)

# Store 3: Recursive chunks (splits at natural boundaries)
store_recursive = Chroma.from_documents(
    documents=recursive_chunks,
    embedding=embeddings_model,
    collection_name="recursive_chunks"
)

test_query = "Database connection failures"
print(f"\nTest query: '{test_query}'")
# Compare results from each strategy
stores = [
    ("Whole Documents", store_whole),
    ("Fixed Chunks", store_fixed),
    ("Recursive Chunks", store_recursive)
]

for name, store in stores:
    results = store.similarity_search(test_query, k=1)
    print(f"\n{name}:")
    if results:
        print(f"  Top result: {results[0].page_content[:100]}...")
        print(f"  Length: {len(results[0].page_content)} chars")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("DEMO COMPLETE!")
print("="*80)
# ============================================================================
# Cleanup
# ============================================================================
print("\n" + "=" * 80)
print("CLEANUP")
print("=" * 80)

import shutil
if os.path.exists("./solution_chroma_db"):
    shutil.rmtree("./solution_chroma_db")
    print("✓ Cleaned up solution_chroma_db")
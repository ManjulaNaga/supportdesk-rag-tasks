import json
import time
import os
from langchain_core.documents import Document 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # OpenAI integrations
from langchain_community.vectorstores import Chroma  # Vector database for similarity search
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Smart text chunking

# ============================================================================
# PART 1: Data Ingestion & Vector Store Setup
# ============================================================================
print("\n" + "="*80)
print("PART 1: Data Ingestion Pipeline")
print("="*80)

# Load tickets
with open('../../data/synthetic_tickets.json', 'r') as f:
    tickets = json.load(f)
print(f"✓ Loaded {len(tickets)} support tickets")

# Convert to LangChain Document objects

documents = []
for ticket in tickets:
    # Create rich document with all context
    # TIP: Structure your content logically - LLMs understand formatted text better
    content = f"""
Ticket ID: {ticket['ticket_id']}
Title: {ticket['title']}
Category: {ticket['category']}
Priority: {ticket['priority']}
Date: {ticket['created_date']} to {ticket['resolved_date']}
Problem Description:
{ticket['description']}
Resolution:
{ticket['resolution']}
    """.strip()
    
    # Create Document with metadata
    doc = Document(
        page_content=content,  # The actual text content
        metadata={  # Structured data about the document
            'ticket_id': ticket['ticket_id'],
            'title': ticket['title'],
            'category': ticket['category'],
            'priority': ticket['priority'],
            'source': f"Ticket {ticket['ticket_id']}"
        }
    )
    documents.append(doc)

print(f"✓ Created {len(documents)} documents with metadata")

# Initialize OpenAI embeddings
print("\nInitializing OpenAI embedding model...")
embeddings = OpenAIEmbeddings(
    model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
)
print("✓ OpenAI embedding model ready")

# Build vector store using Chroma
print("\nBuilding Chroma vector store...")
vector_store = Chroma.from_documents(
    documents=documents,  # Our support ticket documents
    embedding=embeddings,  # Embedding function to use
    collection_name="supportdesk_rag",  # Name for this collection
    persist_directory="./rag_vectorstore"  # Where to save the database
)
print("✓ Vector store created and persisted")

# ============================================================================
#  Create Retriever
# ============================================================================
print("\n" + "="*80)
print("PART 2: Setting Up Retriever")
print("="*80)

# Create a retriever from the vector store
retriever = vector_store.as_retriever(
    search_type="mmr",  # Use cosine similarity for ranking
    search_kwargs={"k": 3, "fetch_k": 10}  # Retrieve top-3 most similar documents
)

print("✓ Retriever configured:")
print(f"  - Search type: similarity")
print(f"  - Top-K results: 3")
query="How do I fix database timeouts?"
# 1. "stuff" strategy - All docs in one prompt (default LCEL pattern)
stuff_prompt = ChatPromptTemplate.from_template(
    "Answer using context:\n\nContext: {context}\n\nQuestion: {question}"
)
def format_docs(docs):
    return "\n\n---\n\n".join([doc.page_content for doc in docs])
if os.getenv("OPENAI_API_KEY"):
    print("✓ OpenAI API key found")
    # Initialize ChatOpenAI for generation
    # Reference: https://python.langchain.com/docs/integrations/chat/openai
    llm = ChatOpenAI(
        model=os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini'),
        temperature=0,  # Temperature controls randomness (0 = deterministic, 2 = very creative)
        timeout=120,  # Increase timeout for slower connections
        max_retries=3,  # Retry on transient failures
    )
    print(f"✓ Using {os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini')}")
else:
    print("⚠ OpenAI API key not found!")
    print("  Please set OPENAI_API_KEY environment variable")
    print("  Or use Ollama: ollama pull llama2")
    print("\nFor this demo, we'll show the prompt without generating answers.")
    llm = None

stuff_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | stuff_prompt | llm | StrOutputParser()
)

# 2. "map_reduce" strategy - Process each doc, then combine
docs = retriever.invoke(query)
individual_answers = []
for doc in docs:
    single_chain = single_doc_prompt | llm | StrOutputParser()
    individual_answers.append(single_chain.invoke({"doc": doc.page_content}))
combined_result = combine_chain.invoke({"summaries": "\n".join(individual_answers)})
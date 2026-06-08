## Bonus Exercise: Semantic vs Fixed Chunking (Challenge)

# **Task**: Compare how semantic chunking differs from fixed-size chunking.

# **Note**: This is optional and takes longer to run.

import json
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# Load a longer document for better comparison
long_text = """
User Authentication System Overview

The authentication system handles user login and session management. Users can 
authenticate using username/password or single sign-on (SSO). The system supports
OAuth 2.0 and SAML protocols for enterprise integration.

Password Security

All passwords are hashed using bcrypt with a cost factor of 12. Password policies
require a minimum of 8 characters with at least one uppercase, lowercase, number,
and special character. Passwords expire after 90 days for compliance.

Session Management

User sessions are stored in Redis with a 24-hour expiration. Each session includes
the user ID, roles, and creation timestamp. Sessions can be invalidated through
the admin panel or via API.

Two-Factor Authentication

Users can enable 2FA using TOTP (Google Authenticator) or SMS verification. 
Backup codes are generated for account recovery. Enterprise accounts require
2FA for all users.
"""

doc = Document(page_content=long_text)

# Fixed chunking
fixed_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
fixed_chunks = fixed_splitter.split_documents([doc])

# Semantic chunking
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
semantic_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
semantic_chunks = semantic_splitter.split_documents([doc])

print("Fixed Chunking Results:")
print(f"Number of chunks: {len(fixed_chunks)}")
for i, chunk in enumerate(fixed_chunks, 1):
    print(f"\nChunk {i} ({len(chunk.page_content)} chars):")
    print(f"  {chunk.page_content[:80]}...")

print("\n" + "="*60)
print("\nSemantic Chunking Results:")
print(f"Number of chunks: {len(semantic_chunks)}")
for i, chunk in enumerate(semantic_chunks, 1):
    print(f"\nChunk {i} ({len(chunk.page_content)} chars):")
    print(f"  {chunk.page_content[:80]}...")

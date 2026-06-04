"""Migration script: ingest local files and push chunks to Chroma Cloud.

Usage:
    python migrate_to_chroma.py

Ensure CHROMA_API_KEY and CHROMA_DATABASE are set in your environment or .env.
"""
import os
from dotenv import load_dotenv

load_dotenv(override=True)

import rag_service


def main():
    try:
        num_chunks = rag_service.build_vector_database()
        print(f"Migration completed successfully! Uploaded {num_chunks} chunks.")
    except Exception as e:
        print(f"Error during migration: {e}")


if __name__ == "__main__":
    main()


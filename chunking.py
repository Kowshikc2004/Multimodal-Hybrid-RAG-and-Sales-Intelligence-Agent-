from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List

def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

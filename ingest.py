#!/usr/bin/env python3
"""
Ingest personal data from ./data/ into a local vector database.
Supports .txt, .md, and .pdf files.
"""
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DATA_PATH = "data"
DB_PATH = "db"

def main():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Data directory '{DATA_PATH}' not found!")
        print("👉 Create it and add .txt or .md files.")
        return

    # Load .md and .txt files
    try:
        loader = DirectoryLoader(DATA_PATH, glob="**/*.md", loader_cls=TextLoader)
        documents = loader.load()
    except Exception:
        documents = []

    try:
        loader_txt = DirectoryLoader(DATA_PATH, glob="**/*.txt", loader_cls=TextLoader)
        documents += loader_txt.load()
    except Exception:
        pass

    try:
        loader_pdf = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents += loader_pdf.load()
    except Exception:
        pass

    if not documents:
        print("⚠️ No .md, .txt, or .pdf files found in 'data/' directory!")
        return

    print(f"📄 Loaded {len(documents)} file(s). Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    print("🧠 Generating embeddings (this may take a minute on first run)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    vectorstore.persist()
    print(f"✅ Knowledge base saved to '{DB_PATH}' ({len(chunks)} chunks)")

if __name__ == "__main__":
    main()

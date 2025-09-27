#!/usr/bin/env python3
"""
Ingest personal data from ./data/ into a local vector database.
Supports .txt, .md, and .pdf files.
"""
import os
import logging
from pathlib import Path
from typing import List
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import tqdm

from config import config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_data_directory(data_path: str) -> bool:
    """Validate that the data directory exists and contains supported files."""
    data_dir = Path(data_path)
    if not data_dir.exists():
        print(f"❌ Data directory '{data_path}' not found!")
        print(f"👉 Create it and add {', '.join(config.get('supported_extensions'))} files.")
        return False

    if not data_dir.is_dir():
        print(f"❌ '{data_path}' is not a directory!")
        return False

    # Check for supported files
    supported_files = []
    for ext in config.get('supported_extensions'):
        supported_files.extend(list(data_dir.rglob(f"**/*{ext}")))

    if not supported_files:
        print(f"⚠️ No supported files found in '{data_path}' directory!")
        print(f"Supported formats: {', '.join(config.get('supported_extensions'))}")
        return False

    return True

def load_documents(data_path: str) -> List[Document]:
    """Load documents from the data directory with proper error handling."""
    documents = []
    errors = []

    print(f"🔍 Scanning '{data_path}' for documents...")

    # Load .md files
    try:
        loader_md = DirectoryLoader(data_path, glob="**/*.md", loader_cls=TextLoader)
        md_docs = loader_md.load()
        documents.extend(md_docs)
        print(f"✅ Loaded {len(md_docs)} Markdown file(s)")
    except Exception as e:
        error_msg = f"Failed to load Markdown files: {e}"
        logger.error(error_msg)
        errors.append(error_msg)

    # Load .txt files
    try:
        loader_txt = DirectoryLoader(data_path, glob="**/*.txt", loader_cls=TextLoader)
        txt_docs = loader_txt.load()
        documents.extend(txt_docs)
        print(f"✅ Loaded {len(txt_docs)} text file(s)")
    except Exception as e:
        error_msg = f"Failed to load text files: {e}"
        logger.error(error_msg)
        errors.append(error_msg)

    # Load .pdf files
    try:
        loader_pdf = DirectoryLoader(data_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
        pdf_docs = loader_pdf.load()
        documents.extend(pdf_docs)
        print(f"✅ Loaded {len(pdf_docs)} PDF file(s)")
    except Exception as e:
        error_msg = f"Failed to load PDF files: {e}"
        logger.error(error_msg)
        errors.append(error_msg)

    if errors:
        print(f"⚠️ Encountered {len(errors)} error(s) during loading:")
        for error in errors:
            print(f"   - {error}")

    return documents

def create_chunks(documents: List[Document]) -> List[Document]:
    """Split documents into chunks with progress indication."""
    if not documents:
        return []

    print(f"📄 Splitting {len(documents)} document(s) into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.get('chunk_size'),
        chunk_overlap=config.get('chunk_overlap')
    )

    # Use tqdm for progress indication
    chunks = []
    with tqdm.tqdm(total=len(documents), desc="Processing documents") as pbar:
        for doc in documents:
            doc_chunks = text_splitter.split_documents([doc])
            chunks.extend(doc_chunks)
            pbar.update(1)

    print(f"✅ Created {len(chunks)} chunks")
    return chunks

def create_vectorstore(chunks: List[Document], db_path: str):
    """Create and persist the vector store with progress indication."""
    if not chunks:
        print("⚠️ No chunks to process!")
        return

    print("🧠 Generating embeddings (this may take a while on first run)...")

    embeddings = HuggingFaceEmbeddings(model_name=config.get('embedding_model'))

    # Create vector store with progress
    with tqdm.tqdm(total=1, desc="Creating vector database") as pbar:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=db_path
        )
        vectorstore.persist()
        pbar.update(1)

    print(f"✅ Knowledge base saved to '{db_path}' ({len(chunks)} chunks)")

def main():
    """Main ingestion function."""
    # Create default config if it doesn't exist
    config.save_default_config()

    data_path = config.get('data_path')
    db_path = config.get('db_path')

    print("🚀 Starting data ingestion...")
    print(f"📁 Data directory: {data_path}")
    print(f"🗄️  Database path: {db_path}")

    # Validate data directory
    if not validate_data_directory(data_path):
        return

    # Load documents
    documents = load_documents(data_path)
    if not documents:
        print("❌ No documents could be loaded. Exiting.")
        return

    # Create chunks
    chunks = create_chunks(documents)

    # Create vector store
    create_vectorstore(chunks, db_path)

    print("🎉 Ingestion complete!")

if __name__ == "__main__":
    main()

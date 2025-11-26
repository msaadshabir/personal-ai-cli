#!/usr/bin/env python3
"""
Ingest personal data from ./data/ into a local vector database.
Supports .txt, .md, .pdf, .docx, .html, .json, .csv, and .epub files.
Features: incremental ingestion (only re-ingests changed files), rich UI.
"""
import os
import json
import hashlib
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Set

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel

from langchain_community.document_loaders import (
    DirectoryLoader, 
    TextLoader, 
    PyPDFLoader,
    UnstructuredHTMLLoader,
    CSVLoader,
    JSONLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from config import config

# Rich console
console = Console()

# Setup logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hash file location
HASH_FILE = "file_hashes.json"


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest personal data into the vector database"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-ingestion of all files (ignore hashes)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with full stack traces"
    )
    return parser.parse_args()


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def load_file_hashes() -> Dict[str, str]:
    """Load previously computed file hashes."""
    if os.path.exists(HASH_FILE):
        try:
            with open(HASH_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load hash file: {e}")
    return {}


def save_file_hashes(hashes: Dict[str, str]):
    """Save file hashes to disk."""
    try:
        with open(HASH_FILE, 'w') as f:
            json.dump(hashes, f, indent=2)
    except Exception as e:
        logger.error(f"Could not save hash file: {e}")


def get_changed_files(data_path: str, force: bool = False) -> tuple[List[Path], Dict[str, str]]:
    """
    Determine which files have changed since last ingestion.
    Returns tuple of (changed_files, new_hashes).
    """
    data_dir = Path(data_path)
    old_hashes = {} if force else load_file_hashes()
    new_hashes = {}
    changed_files = []
    
    # Get all supported files
    supported_extensions = config.get('supported_extensions')
    all_files = []
    for ext in supported_extensions:
        all_files.extend(list(data_dir.rglob(f"*{ext}")))
    
    for file_path in all_files:
        file_str = str(file_path)
        current_hash = compute_file_hash(file_path)
        new_hashes[file_str] = current_hash
        
        if file_str not in old_hashes or old_hashes[file_str] != current_hash:
            changed_files.append(file_path)
    
    return changed_files, new_hashes


def validate_data_directory(data_path: str) -> bool:
    """Validate that the data directory exists and contains supported files."""
    data_dir = Path(data_path)
    if not data_dir.exists():
        console.print(f"[bold red]Error:[/bold red] Data directory '{data_path}' not found!")
        console.print(f"[cyan]Hint:[/cyan] Create it and add {', '.join(config.get('supported_extensions'))} files.")
        return False

    if not data_dir.is_dir():
        console.print(f"[bold red]Error:[/bold red] '{data_path}' is not a directory!")
        return False

    # Check for supported files
    supported_files = []
    for ext in config.get('supported_extensions'):
        supported_files.extend(list(data_dir.rglob(f"*{ext}")))

    if not supported_files:
        console.print(f"[yellow]Warning:[/yellow] No supported files found in '{data_path}' directory!")
        console.print(f"Supported formats: {', '.join(config.get('supported_extensions'))}")
        return False

    return True


def load_single_file(file_path: Path, debug: bool = False) -> List[Document]:
    """Load a single file based on its extension."""
    documents = []
    ext = file_path.suffix.lower()
    
    try:
        if ext in ['.md', '.txt']:
            loader = TextLoader(str(file_path))
            documents = loader.load()
        elif ext == '.pdf':
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
        elif ext == '.html':
            loader = UnstructuredHTMLLoader(str(file_path))
            documents = loader.load()
        elif ext == '.csv':
            loader = CSVLoader(str(file_path))
            documents = loader.load()
        elif ext == '.json':
            # JSONLoader requires a jq_schema, we'll use a simple text extraction
            loader = JSONLoader(
                file_path=str(file_path),
                jq_schema='.',
                text_content=False
            )
            documents = loader.load()
        elif ext == '.docx':
            try:
                from langchain_community.document_loaders import Docx2txtLoader
                loader = Docx2txtLoader(str(file_path))
                documents = loader.load()
            except ImportError:
                console.print(f"[yellow]Warning:[/yellow] Install 'docx2txt' to load .docx files: pip install docx2txt")
        elif ext == '.epub':
            try:
                from langchain_community.document_loaders import UnstructuredEPubLoader
                loader = UnstructuredEPubLoader(str(file_path))
                documents = loader.load()
            except ImportError:
                console.print(f"[yellow]Warning:[/yellow] Install 'ebooklib' to load .epub files: pip install ebooklib")
    except Exception as e:
        if debug:
            console.print_exception()
        logger.error(f"Failed to load {file_path}: {e}")
        console.print(f"[yellow]Warning:[/yellow] Failed to load {file_path.name}: {e}")
    
    return documents


def load_documents(data_path: str, files_to_load: Optional[List[Path]] = None, debug: bool = False) -> List[Document]:
    """Load documents from the data directory."""
    documents = []
    errors = []
    data_dir = Path(data_path)
    
    # If specific files provided, load only those
    if files_to_load:
        files = files_to_load
    else:
        # Load all supported files
        files = []
        for ext in config.get('supported_extensions'):
            files.extend(list(data_dir.rglob(f"*{ext}")))
    
    if not files:
        return documents
    
    console.print(f"\n[cyan]Loading {len(files)} file(s)...[/cyan]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Loading files...", total=len(files))
        
        for file_path in files:
            progress.update(task, description=f"Loading {file_path.name}...")
            docs = load_single_file(file_path, debug)
            documents.extend(docs)
            progress.advance(task)
    
    # Summary table
    ext_counts: Dict[str, int] = {}
    for file in files:
        ext = file.suffix.lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
    
    table = Table(title="Files Loaded", show_header=True, header_style="bold cyan")
    table.add_column("Extension")
    table.add_column("Count", justify="right")
    
    for ext, count in sorted(ext_counts.items()):
        table.add_row(ext, str(count))
    
    table.add_row("[bold]Total[/bold]", f"[bold]{len(files)}[/bold]")
    console.print(table)
    
    if errors:
        console.print(f"[yellow]Encountered {len(errors)} error(s) during loading[/yellow]")
    
    return documents


def create_chunks(documents: List[Document]) -> List[Document]:
    """Split documents into chunks with progress indication."""
    if not documents:
        return []

    console.print(f"\n[cyan]Splitting {len(documents)} document(s) into chunks...[/cyan]")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.get('chunk_size'),
        chunk_overlap=config.get('chunk_overlap')
    )

    chunks = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Processing...", total=len(documents))
        for doc in documents:
            doc_chunks = text_splitter.split_documents([doc])
            chunks.extend(doc_chunks)
            progress.advance(task)

    console.print(f"[green]Created {len(chunks)} chunks[/green]")
    return chunks


def create_vectorstore(chunks: List[Document], db_path: str, incremental: bool = False):
    """Create and persist the vector store."""
    if not chunks:
        console.print("[yellow]No chunks to process![/yellow]")
        return

    console.print("\n[cyan]Generating embeddings (this may take a while on first run)...[/cyan]")

    embeddings = HuggingFaceEmbeddings(model_name=config.get('embedding_model'))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        progress.add_task("Creating vector database...", total=None)
        
        if incremental and os.path.exists(db_path):
            # Add to existing vectorstore
            vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
            vectorstore.add_documents(chunks)
        else:
            # Create new vectorstore
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=db_path
            )

    console.print(f"[green]Knowledge base saved to '{db_path}' ({len(chunks)} chunks)[/green]")


def main():
    """Main ingestion function."""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create default config if it doesn't exist
    config.save_default_config()

    data_path = config.get('data_path')
    db_path = config.get('db_path')

    # Welcome banner
    console.print(Panel.fit(
        "[bold]Personal AI CLI - Data Ingestion[/bold]",
        title="Ingestion",
        border_style="cyan"
    ))
    
    console.print(f"Data directory: [cyan]{data_path}[/cyan]")
    console.print(f"Database path: [cyan]{db_path}[/cyan]")

    # Validate data directory
    if not validate_data_directory(data_path):
        return

    # Check for changed files (incremental ingestion)
    if args.force:
        console.print("\n[yellow]Force mode: re-ingesting all files[/yellow]")
        changed_files, new_hashes = get_changed_files(data_path, force=True)
        incremental = False
    else:
        changed_files, new_hashes = get_changed_files(data_path, force=False)
        incremental = os.path.exists(db_path)
    
    if not changed_files and incremental:
        console.print("\n[green]No files have changed. Database is up to date![/green]")
        return
    
    if incremental and changed_files:
        console.print(f"\n[cyan]Found {len(changed_files)} changed/new file(s)[/cyan]")
        for f in changed_files[:5]:  # Show first 5
            console.print(f"   • {f.name}")
        if len(changed_files) > 5:
            console.print(f"   ... and {len(changed_files) - 5} more")

    # Load documents (only changed files if incremental)
    if incremental and changed_files:
        documents = load_documents(data_path, files_to_load=changed_files, debug=args.debug)
    else:
        documents = load_documents(data_path, debug=args.debug)
    
    if not documents:
        console.print("[bold red]No documents could be loaded. Exiting.[/bold red]")
        return

    # Create chunks
    chunks = create_chunks(documents)

    # Create vector store
    create_vectorstore(chunks, db_path, incremental=incremental)
    
    # Save file hashes for incremental updates
    save_file_hashes(new_hashes)

    console.print("\n[bold green]Ingestion complete![/bold green]")


if __name__ == "__main__":
    main()

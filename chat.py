#!/usr/bin/env python3
"""
Terminal-based RAG chatbot using your personal data and a local LLM (via Ollama).
Features: streaming responses, conversation memory, source citation, rich UI, multi-model support.
"""
import os
import sys
import argparse
import logging
import re
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table
from rich.theme import Theme

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.callbacks import BaseCallbackHandler

from config import config

# Rich console with custom theme
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "ai": "bold magenta",
    "user": "bold blue",
    "source": "dim italic",
})
console = Console(theme=custom_theme)

# Setup logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming LLM responses."""
    
    def __init__(self, console: Console):
        self.console = console
        self.text = ""
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Handle new token from LLM."""
        self.text += token
        self.console.print(token, end="", style="ai")
    
    def on_llm_end(self, response, **kwargs) -> None:
        """Handle end of LLM response."""
        self.console.print()  # New line after response


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Personal AI CLI - Chat with your personal data using a local LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands during chat:
  /model <name>    Switch to a different Ollama model
  /models          List available models
  /export          Export chat history to file
  /clear           Clear conversation history
  /sources         Toggle source citation display
  /help            Show available commands
  exit, quit, q    Exit the chat
        """
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Ollama model to use (overrides config)"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose/debug output"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with full stack traces"
    )
    parser.add_argument(
        "--clear-db",
        action="store_true",
        help="Clear the vector database and exit"
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming responses"
    )
    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="Disable source citation display"
    )
    return parser.parse_args()


def validate_input(text: str, max_length: int = 1000) -> bool:
    """Validate user input for safety and reasonableness."""
    if not text or not text.strip():
        return False

    # Check length
    if len(text.strip()) > max_length:
        console.print(f"[error]Input too long! Maximum {max_length} characters allowed.[/error]")
        return False

    # Check for potentially harmful patterns (basic check)
    dangerous_patterns = [
        r'<script',
        r'javascript:',
        r'data:',
        r'vbscript:',
    ]

    text_lower = text.lower()
    for pattern in dangerous_patterns:
        if re.search(pattern, text_lower):
            console.print("[error]Potentially unsafe input detected.[/error]")
            return False

    return True


def check_db(db_path: str) -> bool:
    """Check if vector database exists."""
    if not os.path.exists(db_path):
        console.print("[error]Vector database not found![/error]")
        console.print("[info]Run 'python ingest.py' first to load your data.[/info]")
        return False
    return True


def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple = (Exception,)
) -> Any:
    """Execute a function with exponential backoff retry logic."""
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return func()
        except exceptions as e:
            last_exception = e
            if attempt < max_retries - 1:
                delay = min(base_delay * (2 ** attempt), max_delay)
                console.print(f"[warning]Attempt {attempt + 1} failed. Retrying in {delay:.1f}s...[/warning]")
                time.sleep(delay)
            else:
                raise last_exception


def test_ollama(model_name: str, streaming: bool = True, debug: bool = False) -> Optional[Ollama]:
    """Test Ollama connection and return LLM instance with retry logic."""
    def _connect():
        callbacks = []
        if streaming:
            callbacks.append(StreamingCallbackHandler(console))
        
        llm = Ollama(
            model=model_name,
            temperature=config.get('temperature'),
            callbacks=callbacks if streaming else None,
        )
        # Test connection
        llm.invoke("hello")
        return llm
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task(f"Connecting to Ollama ({model_name})...", total=None)
            return retry_with_backoff(_connect, max_retries=3)
    except Exception as e:
        if debug:
            console.print_exception()
        logger.error(f"Failed to connect to Ollama with model '{model_name}': {e}")
        console.print(f"[error]Failed to connect to Ollama with model '{model_name}'[/error]")
        console.print("\n[info]Please install Ollama: https://ollama.com/[/info]")
        console.print(f"[info]Then run: ollama pull {model_name}[/info]")
        console.print("\n[info]Alternatively, check your Ollama service is running with: ollama serve[/info]")
        return None


def create_qa_chain(llm: Ollama, vectorstore: Chroma, memory: ConversationBufferMemory):
    """Create the RAG QA chain with conversation memory and source documents."""
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": config.get('top_k')}
        ),
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )

    return qa_chain


def display_sources(source_documents: List[Any], show_sources: bool = True):
    """Display source documents used for the answer."""
    if not show_sources or not source_documents:
        return
    
    console.print()
    sources_table = Table(title="Sources", show_header=True, header_style="bold cyan")
    sources_table.add_column("File", style="source")
    sources_table.add_column("Preview", style="source", max_width=60)
    
    seen_sources = set()
    for doc in source_documents:
        source = doc.metadata.get('source', 'Unknown')
        if source not in seen_sources:
            seen_sources.add(source)
            preview = doc.page_content[:100].replace('\n', ' ') + "..."
            sources_table.add_row(Path(source).name, preview)
    
    console.print(sources_table)


class ChatSession:
    """Manages the chat session with history export and model switching."""
    
    def __init__(self, model: str, vectorstore: Chroma, streaming: bool = True, 
                 show_sources: bool = True, debug: bool = False):
        self.model = model
        self.vectorstore = vectorstore
        self.streaming = streaming
        self.show_sources = show_sources
        self.debug = debug
        self.history: List[Dict[str, Any]] = []
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        self.llm = None
        self.qa_chain = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize or reinitialize the LLM."""
        self.llm = test_ollama(self.model, streaming=self.streaming, debug=self.debug)
        if self.llm:
            self.qa_chain = create_qa_chain(self.llm, self.vectorstore, self.memory)
    
    def switch_model(self, new_model: str) -> bool:
        """Switch to a different model."""
        console.print(f"[info]Switching to model: {new_model}...[/info]")
        old_model = self.model
        self.model = new_model
        
        try:
            self._initialize_llm()
            if self.llm:
                console.print(f"[success]Successfully switched to {new_model}[/success]")
                return True
            else:
                self.model = old_model
                self._initialize_llm()
                return False
        except Exception as e:
            console.print(f"[error]Failed to switch model: {e}[/error]")
            self.model = old_model
            self._initialize_llm()
            return False
    
    def clear_history(self):
        """Clear conversation history."""
        self.history = []
        self.memory.clear()
        console.print("[success]Conversation history cleared.[/success]")
    
    def export_history(self, format: str = "json") -> Optional[str]:
        """Export chat history to a file."""
        if not self.history:
            console.print("[warning]No chat history to export.[/warning]")
            return None
        
        # Create chat_history directory if it doesn't exist
        export_dir = Path("chat_history")
        export_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "json":
            filename = export_dir / f"chat_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump({
                    "model": self.model,
                    "timestamp": timestamp,
                    "messages": self.history
                }, f, indent=2)
        else:  # markdown
            filename = export_dir / f"chat_{timestamp}.md"
            with open(filename, 'w') as f:
                f.write(f"# Chat Export - {timestamp}\n\n")
                f.write(f"**Model:** {self.model}\n\n---\n\n")
                for msg in self.history:
                    if msg["role"] == "user":
                        f.write(f"## You\n{msg['content']}\n\n")
                    else:
                        f.write(f"## AI\n{msg['content']}\n\n")
                        if msg.get("sources"):
                            f.write("**Sources:**\n")
                            for src in msg["sources"]:
                                f.write(f"- {src}\n")
                            f.write("\n")
        
        console.print(f"[success]Chat history exported to: {filename}[/success]")
        return str(filename)
    
    def ask(self, question: str) -> Optional[str]:
        """Ask a question and get a response."""
        if not self.qa_chain:
            console.print("[error]Chat system not initialized. Please check your connection.[/error]")
            return None
        
        # Record user message
        self.history.append({
            "role": "user",
            "content": question,
            "timestamp": datetime.now().isoformat()
        })
        
        try:
            if not self.streaming:
                console.print("[ai]AI:[/ai] ", end="")
            else:
                console.print("[ai]AI:[/ai] ", end="")
            
            response = self.qa_chain.invoke({"question": question})
            answer = response["answer"].strip()
            
            if not self.streaming:
                console.print(answer, style="ai")
            
            # Get source documents
            sources = []
            if response.get("source_documents"):
                sources = list(set(
                    doc.metadata.get('source', 'Unknown') 
                    for doc in response["source_documents"]
                ))
                display_sources(response["source_documents"], self.show_sources)
            
            # Record AI message
            self.history.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "timestamp": datetime.now().isoformat()
            })
            
            return answer
            
        except Exception as e:
            if self.debug:
                console.print_exception()
            logger.error(f"Error during query processing: {e}")
            console.print(f"\n[error]Sorry, I encountered an error: {e}[/error]")
            console.print("[info]Please try rephrasing your question.[/info]")
            return None


def show_help():
    """Display help information."""
    help_table = Table(title="Available Commands", show_header=True, header_style="bold cyan")
    help_table.add_column("Command", style="bold")
    help_table.add_column("Description")
    
    commands = [
        ("/model <name>", "Switch to a different Ollama model"),
        ("/models", "List suggested models"),
        ("/export [json|md]", "Export chat history to file"),
        ("/clear", "Clear conversation history"),
        ("/sources", "Toggle source citation display"),
        ("/help", "Show this help message"),
        ("exit, quit, q", "Exit the chat"),
    ]
    
    for cmd, desc in commands:
        help_table.add_row(cmd, desc)
    
    console.print(help_table)


def show_models():
    """Display suggested models."""
    models_table = Table(title="Suggested Ollama Models", show_header=True, header_style="bold cyan")
    models_table.add_column("Model", style="bold")
    models_table.add_column("Size")
    models_table.add_column("Description")
    
    models = [
        ("llama3", "4.7GB", "Meta's Llama 3 - Great general purpose model"),
        ("llama3:70b", "40GB", "Llama 3 70B - More capable, needs more RAM"),
        ("mistral", "4.1GB", "Mistral 7B - Fast and efficient"),
        ("mixtral", "26GB", "Mixtral 8x7B - Mixture of experts"),
        ("phi3", "2.2GB", "Microsoft Phi-3 - Small but capable"),
        ("gemma2", "5.4GB", "Google Gemma 2 - Good for conversations"),
        ("qwen2", "4.4GB", "Alibaba Qwen 2 - Multilingual"),
    ]
    
    for model, size, desc in models:
        models_table.add_row(model, size, desc)
    
    console.print(models_table)
    console.print("\n[info]Install with: ollama pull <model_name>[/info]")


def clear_database(db_path: str):
    """Clear the vector database."""
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
        console.print(f"[success]Vector database at '{db_path}' has been cleared.[/success]")
    else:
        console.print(f"[warning]No database found at '{db_path}'.[/warning]")


def main():
    """Main chat function."""
    args = parse_args()
    
    # Set debug logging if requested
    if args.verbose or args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Reload config with custom path if specified
    if args.config != "config.yaml":
        from config import Config
        global config
        config = Config(args.config)
    
    # Create default config if it doesn't exist
    config.save_default_config()
    
    db_path = config.get('db_path')
    model = args.model or config.get('default_model')
    
    # Handle --clear-db
    if args.clear_db:
        clear_database(db_path)
        sys.exit(0)
    
    # Check database
    if not check_db(db_path):
        sys.exit(1)
    
    # Welcome banner
    console.print(Panel.fit(
        f"[bold]Personal AI CLI[/bold]\n"
        f"Model: [cyan]{model}[/cyan] | Type [bold]/help[/bold] for commands",
        title="Welcome",
        border_style="cyan"
    ))
    
    # Load vector DB
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Loading knowledge base...", total=None)
            embeddings = HuggingFaceEmbeddings(model_name=config.get('embedding_model'))
            vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
    except Exception as e:
        if args.debug:
            console.print_exception()
        logger.error(f"Failed to load vector database: {e}")
        console.print(f"[error]Failed to load vector database: {e}[/error]")
        console.print("[info]Try re-running 'python ingest.py' to rebuild the database.[/info]")
        sys.exit(1)
    
    # Create chat session
    session = ChatSession(
        model=model,
        vectorstore=vectorstore,
        streaming=not args.no_stream,
        show_sources=not args.no_sources,
        debug=args.debug
    )
    
    if not session.llm:
        sys.exit(1)
    
    console.print("[info]Ask questions about your data. Type 'exit' to quit.[/info]\n")
    
    # Chat loop
    while True:
        try:
            user_input = Prompt.ask("[user]You[/user]").strip()
            
            # Check for exit commands
            if user_input.lower() in {"exit", "quit", "q"}:
                console.print("[success]Goodbye![/success]")
                break
            
            # Handle slash commands
            if user_input.startswith("/"):
                parts = user_input[1:].split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else None
                
                if cmd == "help":
                    show_help()
                elif cmd == "models":
                    show_models()
                elif cmd == "model" and arg:
                    session.switch_model(arg)
                elif cmd == "model":
                    console.print(f"[info]Current model: {session.model}[/info]")
                elif cmd == "export":
                    fmt = arg if arg in ["json", "md"] else "json"
                    session.export_history(fmt)
                elif cmd == "clear":
                    session.clear_history()
                elif cmd == "sources":
                    session.show_sources = not session.show_sources
                    status = "enabled" if session.show_sources else "disabled"
                    console.print(f"[info]Source citations {status}.[/info]")
                else:
                    console.print(f"[warning]Unknown command: /{cmd}. Type /help for available commands.[/warning]")
                continue
            
            # Validate input
            if not validate_input(user_input):
                continue
            
            # Get response
            session.ask(user_input)
            console.print()

        except KeyboardInterrupt:
            console.print("\n\n[success]Goodbye![/success]")
            break
        except EOFError:
            console.print("\n\n[success]Goodbye![/success]")
            break
        except Exception as e:
            if args.debug:
                console.print_exception()
            logger.error(f"Unexpected error in chat loop: {e}")
            console.print(f"\n[warning]Unexpected error: {e}[/warning]\n")


if __name__ == "__main__":
    main()

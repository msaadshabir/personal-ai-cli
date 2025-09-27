#!/usr/bin/env python3
"""
Terminal-based RAG chatbot using your personal data and a local LLM (via Ollama).
"""
import os
import sys
import logging
import re
from typing import Optional
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from config import config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_input(text: str, max_length: int = 1000) -> bool:
    """Validate user input for safety and reasonableness."""
    if not text or not text.strip():
        return False

    # Check length
    if len(text.strip()) > max_length:
        print(f"❌ Input too long! Maximum {max_length} characters allowed.")
        return False

    # Check for potentially harmful patterns (basic check)
    dangerous_patterns = [
        r'<script',  # Script injection
        r'javascript:',  # JavaScript URLs
        r'data:',  # Data URLs
        r'vbscript:',  # VBScript
    ]

    text_lower = text.lower()
    for pattern in dangerous_patterns:
        if re.search(pattern, text_lower):
            print("❌ Potentially unsafe input detected.")
            return False

    return True

def check_db(db_path: str) -> bool:
    """Check if vector database exists."""
    if not os.path.exists(db_path):
        print("❌ Vector database not found!")
        print("👉 Run 'python ingest.py' first to load your data.")
        return False
    return True

def test_ollama(model_name: str) -> Optional[Ollama]:
    """Test Ollama connection and return LLM instance."""
    try:
        llm = Ollama(
            model=model_name,
            temperature=config.get('temperature')
            # Note: num_predict/max_tokens may not be supported by Ollama class
            # num_predict=config.get('max_tokens')
        )
        # Test connection
        llm.invoke("hello")
        return llm
    except Exception as e:
        logger.error(f"Failed to connect to Ollama with model '{model_name}': {e}")
        print(f"❌ Failed to connect to Ollama with model '{model_name}': {e}")
        print("\n💡 Please install Ollama: https://ollama.com/")
        print(f"Then run: ollama pull {model_name}")
        print("\nAlternatively, check your Ollama service is running with: ollama serve")
        return None

def create_qa_chain(llm: Ollama, vectorstore: Chroma):
    """Create the RAG QA chain with improved prompt."""
    prompt = PromptTemplate.from_template(
        "You are a helpful assistant answering questions based on the user's personal data. "
        "Answer based ONLY on the following context. If the context doesn't contain enough "
        "information to answer the question, say 'I don't have enough information about that in your data.'\n\n"
        "Context: {context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": config.get('top_k')}
        ),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )

    return qa_chain

def main():
    """Main chat function."""
    # Create default config if it doesn't exist
    config.save_default_config()

    db_path = config.get('db_path')
    model = config.get('default_model')

    # Check database
    if not check_db(db_path):
        sys.exit(1)

    print(f"💬 Starting Personal AI CLI (model: {model})")
    print("📘 Ask questions about yourself. Type 'exit', 'quit', or 'q' to quit.\n")

    # Load vector DB
    try:
        embeddings = HuggingFaceEmbeddings(model_name=config.get('embedding_model'))
        vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
    except Exception as e:
        logger.error(f"Failed to load vector database: {e}")
        print(f"❌ Failed to load vector database: {e}")
        print("Try re-running 'python ingest.py' to rebuild the database.")
        sys.exit(1)

    # Connect to LLM
    llm = test_ollama(model)
    if not llm:
        sys.exit(1)

    # Create QA chain
    try:
        qa_chain = create_qa_chain(llm, vectorstore)
    except Exception as e:
        logger.error(f"Failed to create QA chain: {e}")
        print(f"❌ Failed to initialize chat system: {e}")
        sys.exit(1)

    # Chat loop
    while True:
        try:
            user_input = input("You: ").strip()

            # Check for exit commands
            if user_input.lower() in {"exit", "quit", "q"}:
                print("👋 Goodbye!")
                break

            # Validate input
            if not validate_input(user_input):
                continue

            print("AI: ", end="", flush=True)

            # Get response
            try:
                response = qa_chain.invoke({"query": user_input})
                print(response["result"].strip())
            except Exception as e:
                logger.error(f"Error during query processing: {e}")
                print(f"❌ Sorry, I encountered an error processing your question: {e}")
                print("Please try rephrasing your question.")

            print()

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except EOFError:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"Unexpected error in chat loop: {e}")
            print(f"\n⚠️ Unexpected error: {e}\n")

if __name__ == "__main__":
    main()

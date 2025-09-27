#!/usr/bin/env python3
"""
Terminal-based RAG chatbot using your personal data and a local LLM (via Ollama).
"""
import os
import sys
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

DB_PATH = "db"
DEFAULT_MODEL = "llama3"

def check_db():
    if not os.path.exists(DB_PATH):
        print("❌ Vector database not found!")
        print("👉 Run 'python ingest.py' first to load your data.")
        sys.exit(1)

def test_ollama(model_name):
    try:
        llm = Ollama(model=model_name)
        llm.invoke("hello")
        return llm
    except Exception as e:
        print(f"❌ Failed to connect to Ollama with model '{model_name}': {e}")
        print("\n💡 Please install Ollama: https://ollama.com/")
        print(f"Then run: ollama pull {model_name}")
        sys.exit(1)

def main():
    check_db()
    model = os.getenv("AI_MODEL", DEFAULT_MODEL)
    print(f"💬 Starting Personal AI CLI (model: {model})")
    print("📘 Ask questions about yourself. Type 'exit' to quit.\n")

    # Load vector DB
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    # Connect to LLM
    llm = test_ollama(model)

    # RAG prompt
    prompt = PromptTemplate.from_template(
        "Answer based ONLY on the following context. If the context doesn't contain the answer, say 'I don't know.'\n"
        "Context: {context}\n"
        "Question: {question}"
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )

    # Chat loop
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in {"exit", "quit", "q"}:
                print("👋 Goodbye!")
                break
            if not user_input:
                continue

            print("AI: ", end="", flush=True)
            response = qa_chain.invoke({"query": user_input})
            print(response["result"].strip())
            print()

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n⚠️ Unexpected error: {e}\n")

if __name__ == "__main__":
    main()

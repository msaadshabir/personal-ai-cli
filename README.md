# personal-ai-cli

Terminal based, open-source, self-hosted personal AI chatbot that anyone can use with their own data. No UI, no cloud, no tracking.

## Features

- Supports .txt, .md, and .pdf files
- Local vector database (ChromaDB)
- Local embeddings (sentence-transformers)
- Local LLM via Ollama
- Retrieval-Augmented Generation (RAG)

## Installation

1. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Install Ollama: https://ollama.com/
   ```bash
   # Pull a model (default is llama3)
   ollama pull llama3
   ```

## Usage

1. Add your personal documents (.txt, .md, .pdf) to the `data/` directory.

2. Ingest your data into the vector database:

   ```bash
   python ingest.py
   ```

3. Start chatting:

   ```bash
   python chat.py
   ```

   You can specify a different model with:

   ```bash
   AI_MODEL=llama2 python chat.py
   ```

## License

MIT

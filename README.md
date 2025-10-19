# personal-ai-cli

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-green.svg)](https://github.com/langchain-ai/langchain)
[![Ollama](https://img.shields.io/badge/Ollama-Compatible-orange.svg)](https://ollama.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-purple.svg)](https://www.trychroma.com/)

Terminal based, open-source, self-hosted personal AI chatbot that anyone can use with their own data. No UI, no cloud, no tracking.

## Features

- Supports .txt, .md, and .pdf files
- Local vector database (ChromaDB)
- Local embeddings (sentence-transformers)
- Local LLM via Ollama
- Retrieval-Augmented Generation (RAG)
- Configurable settings via YAML or environment variables
- Robust error handling and progress indicators
- Input validation and security checks

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

## Configuration

The application uses a `config.yaml` file for settings. A default configuration will be created automatically on first run. You can customize:

- **Data paths**: Where to find documents and store the database
- **LLM settings**: Model, temperature, max tokens
- **Retrieval settings**: Chunk size, overlap, number of results
- **Embedding model**: Which sentence transformer to use

### Environment Variable Overrides

You can override any config setting using environment variables prefixed with `AI_`:

```bash
# Use a different model
AI_DEFAULT_MODEL=llama2 python chat.py

# Custom chunk size
AI_CHUNK_SIZE=1000 python ingest.py

# Multiple overrides
AI_DEFAULT_MODEL=mistral AI_TEMPERATURE=0.3 python chat.py
```

## Usage

1. Add your personal documents (.txt, .md, .pdf) to the `data/` directory.

2. Ingest your data into the vector database:

   ```bash
   python ingest.py
   ```

   This will show progress indicators and detailed status messages.

3. Start chatting:

   ```bash
   python chat.py
   ```

   The chat interface includes input validation and error handling.

## Configuration Options

| Setting           | Default              | Description                           |
| ----------------- | -------------------- | ------------------------------------- |
| `data_path`       | `"data"`             | Directory containing documents        |
| `db_path`         | `"db"`               | Vector database storage location      |
| `default_model`   | `"llama3"`           | Ollama model to use                   |
| `max_tokens`      | `512`                | Maximum response length               |
| `temperature`     | `0.1`                | Response creativity (0.0-1.0)         |
| `top_k`           | `3`                  | Number of document chunks to retrieve |
| `chunk_size`      | `500`                | Text chunk size in characters         |
| `chunk_overlap`   | `50`                 | Overlap between chunks                |
| `embedding_model` | `"all-MiniLM-L6-v2"` | Embedding model name                  |

**"Vector database not found"**

- Run `python ingest.py` first to create the database

**"Failed to connect to Ollama"**

- Ensure Ollama is installed: `ollama --version`
- Start Ollama service: `ollama serve`
- Pull the required model: `ollama pull llama3`

**"No supported files found"**

- Check that files have the correct extensions (.txt, .md, .pdf)
- Ensure files are in the `data/` directory

**Import errors**

- Install dependencies: `pip install -r requirements.txt`

### Logs

The application logs detailed information to help with troubleshooting. Check the console output for error messages and warnings.

## License

MIT

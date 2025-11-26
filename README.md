# personal-ai-cli

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-green.svg)](https://github.com/langchain-ai/langchain)
[![Ollama](https://img.shields.io/badge/Ollama-Compatible-orange.svg)](https://ollama.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-purple.svg)](https://www.trychroma.com/)

Terminal based, open-source, self-hosted personal AI chatbot that anyone can use with their own data. No UI, no cloud, no tracking.

## Features

- **Streaming Responses** - Token-by-token streaming for real-time output
- **Conversation Memory** - Maintains context across your chat session
- **Source Citations** - See which documents were used to generate answers
- **Rich Terminal UI** - Beautiful colored output with progress indicators
- **Multi-Model Support** - Switch between Ollama models mid-conversation
- **Many File Formats** - Supports .txt, .md, .pdf, .docx, .html, .json, .csv, .epub
- **Incremental Ingestion** - Only re-processes changed files
- **Chat History Export** - Export conversations to JSON or Markdown
- **CLI Arguments** - Full command-line interface with flags
- **Retry Logic** - Automatic retry with exponential backoff
- **Input Validation** - Security checks on user input
- **Configurable** - YAML config with environment variable overrides
- **Unit Tests** - Comprehensive test suite with pytest

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

3. (Optional) Install additional document loaders:

   ```bash
   # For .docx support
   pip install docx2txt

   # For .epub support
   pip install ebooklib
   ```

## Quick Start

1. Add your personal documents to the `data/` directory.

2. Ingest your data:

   ```bash
   python ingest.py
   ```

3. Start chatting:
   ```bash
   python chat.py
   ```

## Configuration

The application uses a `config.yaml` file for settings. A default configuration will be created automatically on first run. You can customize:

- **Data paths**: Where to find documents and store the database
- **LLM settings**: Model, temperature, max tokens
- **Retrieval settings**: Chunk size, overlap, number of results
- **Embedding model**: Which sentence transformer to use
- **Supported extensions**: File types to process

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

## Command Line Options

### Chat (chat.py)

```bash
python chat.py [OPTIONS]

Options:
  -m, --model MODEL    Ollama model to use (overrides config)
  -c, --config FILE    Path to configuration file
  -v, --verbose        Enable verbose output
  --debug              Enable debug mode with full stack traces
  --clear-db           Clear the vector database and exit
  --no-stream          Disable streaming responses
  --no-sources         Disable source citation display
```

### Ingest (ingest.py)

```bash
python ingest.py [OPTIONS]

Options:
  -f, --force          Force re-ingestion of all files (ignore hashes)
  -v, --verbose        Enable verbose output
  --debug              Enable debug mode with full stack traces
```

## Chat Commands

During a chat session, you can use these commands:

| Command              | Description                        |
| -------------------- | ---------------------------------- |
| `/model <name>`      | Switch to a different Ollama model |
| `/models`            | List suggested models              |
| `/export [json\|md]` | Export chat history to file        |
| `/clear`             | Clear conversation history         |
| `/sources`           | Toggle source citation display     |
| `/help`              | Show available commands            |
| `exit`, `quit`, `q`  | Exit the chat                      |

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

## Supported File Types

| Extension | Format     | Notes                       |
| --------- | ---------- | --------------------------- |
| `.txt`    | Plain text | Basic text files            |
| `.md`     | Markdown   | Markdown formatted files    |
| `.pdf`    | PDF        | Portable Document Format    |
| `.docx`   | Word       | Requires `docx2txt` package |
| `.html`   | HTML       | Web pages                   |
| `.json`   | JSON       | Structured data             |
| `.csv`    | CSV        | Comma-separated values      |
| `.epub`   | EPUB       | Ebooks, requires `ebooklib` |

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_config.py -v
```

## Troubleshooting

## Troubleshooting

**"Vector database not found"**

- Run `python ingest.py` first to create the database

**"Failed to connect to Ollama"**

- Ensure Ollama is installed: `ollama --version`
- Start Ollama service: `ollama serve`
- Pull the required model: `ollama pull llama3`

**"No supported files found"**

- Check that files have the correct extensions
- Ensure files are in the `data/` directory

**Import errors**

- Install dependencies: `pip install -r requirements.txt`

**Debug mode**

- Run with `--debug` flag for full stack traces: `python chat.py --debug`

### Chat History Export

Conversations are exported to the `chat_history/` directory:

```bash
# Export as JSON
/export json

# Export as Markdown
/export md
```

### Incremental Ingestion

The system tracks file changes using SHA256 hashes stored in `file_hashes.json`. Only modified or new files are re-ingested:

```bash
# Normal run (incremental)
python ingest.py

# Force full re-ingestion
python ingest.py --force
```

### Logs

The application logs detailed information to help with troubleshooting. Use `--verbose` or `--debug` flags for more output.

## License

MIT

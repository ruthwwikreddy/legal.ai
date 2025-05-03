# Legal Assistant Lite

A lightweight AI assistant for Indian legal research using Ollama to process and analyze legal texts like IPC, Constitution, and case law.

## Prerequisites

1. Python 3.8 or higher
2. Ollama installed and running (https://ollama.com/)
3. Recommended models downloaded in Ollama:
   - `mistral:latest` (default)
   - `llama3:latest` (for higher quality)
   - `tinyllama:latest` (for lightweight use)

## Installation

1. Clone or download this repository
2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Mode
```bash
# Basic query
python legal_assistant_lite.py --query "Your legal question here"

# Use a high-quality model for complex constitutional analysis
python legal_assistant_lite.py --model-type high_quality ask "Analyze the doctrine of basic structure"

# Use a lightweight model for quicker responses on simple questions
python legal_assistant_lite.py --model-type lightweight ask "What is Section 302 of IPC?"
```

### Interactive Mode
```bash
python legal_assistant_lite.py --interactive
```

Available interactive commands:
- `models`: Show all recommended models for legal tasks
- `model [name]`: Switch to specified model (e.g. `model llama3:70b`)
- `load [document]`: Load a legal document for analysis
- `ask [question]`: Ask a legal question
- `exit`: Quit the interactive session

### Options
- `--model MODEL_NAME`: Specify which Ollama model to use (default: mistral)
- `--list-models`: Show available recommended models
- `--document DOC_PATH`: Load a specific legal document

## Recommended Models

The application works best with these Ollama models:
- High quality: llama3, mistral, deepseek-llm
- Balanced: phi3, llama2:7b, mistral:7b
- Lightweight: tinyllama, gemma:2b, phi

## Data Storage

Legal documents are stored in the `legal_data/` directory, organized by type (ipc, constitution, judgments, etc.)


# Call Transcription Processing Agent

A powerful AI-driven system for processing, storing, and querying call transcriptions using LangGraph, RAG (Retrieval-Augmented Generation), and vector databases.

## 🚀 Features

- 🎯 **Multi-format Support**: Process JSON, TXT files
- 🔄 **Recursive Chunking**: Intelligent text splitting using LangChain's RecursiveCharacterTextSplitter
- 🧠 **BGE-M3 Embeddings**: Generate high-quality dense embeddings using BAAI/bge-m3 model
- 🗄️ **Qdrant Integration**: Store and search embeddings in Qdrant vector database
- 🤖 **5-Tool Agent System**: Specialized tools for different transcription tasks
- 🔍 **RAG Question Answering**: Semantic search with AI-powered precise answers
- 🚫 **Duplicate Detection**: Prevents re-processing of existing files
- 📋 **File Listing**: Browse all available transcripts and call IDs
- 🎨 **Clean Output**: Minimal, professional responses
- 📝 **Template System**: Centralized prompt management
- 💬 **Natural Language Interface**: Chat-like interaction for all operations
- MCP server implementation is remaining in this commit(will complete in next commit)

## 📋 Requirements

- **Python**: 3.11.11 or higher
- **Memory**: 4GB+ RAM recommended
- **Storage**: 2GB+ free space
- **GPU**: Optional (CUDA support for faster embeddings)
- **API Keys**: OpenAI
- **Software**: Docker (for Qdrant)

## 🛠️ Quick Installation

### Option 1: Manual Installation

```bash
# 1. Create virtual environment (Python 3.11.11 required)
python3.11 -m venv venv
source venv/bin/activate

# 2. Upgrade pip
pip install --upgrade pip setuptools wheel

# 3. Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. Install requirements
pip install -r requirement.txt

# 5. Create .env file
cp .env.example .env  # Edit with your API keys

# 6. Start Qdrant (optional, if not using Docker)
docker run -p 6333:6333 -v $(pwd):/qdrant/storage qdrant/qdrant

# 7. Start CLI
python src/main.py

# 8. Add your call transcript like this one by one, give the command in CLI with absolute path
"Load and process the transcription file at /home/bhayabhai.sisodiya/Documents/retriever-chatbot/4_negotiation_call.txt"

# 9. Ask questions like this from your inserted transcripts
"what was the pricing discussed?"
```

#

## ⚙️ Configuration

### Environment Variables (.env file)

```env
# OpenAI API Configuration (Required)
OPENAI_API_KEY=your_openai_api_key_here

# Qdrant Configuration
QDRANT_URL="127.0.0.1"
QDRANT_PORT=6333

# Optional: Logging
LOG_LEVEL=INFO
```

### Start Qdrant Server

```bash
# Using Docker (recommended)
docker run -p 6333:6333 qdrant/qdrant
```

## Quick Start

```bash
"I have attached langgraph_demo.ipynb file for quick start and example input outputs, you can take reference from there. also you can run the demo_cli.py file to see the demo of the project after completing the installation and inserting your transcripts.

I have also added mcp_server.py file, and agent file mcp_langgraph_agent.py file for mcp server and client implementation. you can run the mcp_langgraph_agent.py file to see the demo of the project."
```







# RAGLab — A Framework for Multimodal Semantic Retrieval Experiments

## Overview
**RAGLab** is a modular and extensible framework designed to **benchmark and optimize** the semantic retrieval components of **Retrieval-Augmented Generation (RAG)** systems.  
It enables rapid experimentation across all stages of the retrieval pipeline — from **document loading and multimodal chunking**, to **embedding generation** and **vector database indexing**.

## Key Features

- **Flexible Document Ingestion** — Supports multiple file types (PDF, DOCX, images, tables, etc.) with customizable chunking strategies.  
- **Pluggable Embedding Models** — Swap between text and multimodal embedding models for comparative evaluation.  
- **Configurable Vector Stores** — Test across FAISS, Milvus, Chroma, and others with adjustable indexing strategies and hyperparameters.  
- **Evaluation & Logging** — Built-in tools for reproducible experiments, retrieval quality metrics, and performance tracking.  

## Goal
To systematically study how **chunking**, **embedding**, and **vector indexing** choices affect semantic retrieval quality and efficiency in RAG pipelines.

---

## Current Implementations

### 1. File Types

**PDF**
- Multimodal: Text, Images, Tables  
- Vision-Enhanced Chunking: Automatic vision processing for image-heavy pages (based on area threshold)  
- Table Extraction: Preserves tables in Markdown format  
- Chunking by page  

*Extensible: Easily add DOCX, code files, HTML, and more.*

---

### 2. Embedding Models

**OpenAI Embeddings**  
- `text-embedding-3-large` / `text-embedding-3-small`

**Sentence Transformers (HuggingFace)**  
- Compatible with a variety of transformer-based models  

*Extensible: Add code or domain-specific embeddings.*

---

### 3. Vector Database Support

**Milvus**  
- Local standalone setup via Docker container  

*Extensible: Add FAISS, Chroma, or other vector databases with configurable parameters.*

---

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone <your-repo>
cd rag-pipeline

# Install dependencies
uv sync

# Setup environment variables
cat > .env << EOF
OPENAI_API_KEY=your_openai_key_here
EOF
```

### 2. Start Milvus

```bash
# Using Docker Compose (recommended)
# Download docker-compose.yml from Milvus documentation
docker-compose up -d

# Verify Milvus is running
curl http://localhost:19530/healthz
```

### 3. Configure Pipeline

Edit `config.py`:

```python
# Set your input directory
INPUT_DIR = DATA_DIR / "your-documents"

# Choose embedding model
ACTIVE_EMBEDDING_PROVIDER = "openai"  # or "sentence_transformers"

# Configure chunking
CHUNKING_CONFIG = {
    "pdf": {
        "image_coverage_threshold": 0.15,  # Adjust for your needs
        "vision_model": "gpt-4o",
    }
}
```

### 4. Run Ingestion Pipeline

```bash
# Full pipeline: Chunk → Embed → Ingest (split up in case you want to test different vector DB/embedding model with existing chunks after run_chunking.py)
uv run scripts/run_chunking.py
uv run scripts/run_ingestion_milvus.py 
```

### 5. Query Your Data

```bash
# Run example queries
uv run scripts/run_query_milvus.py 
```

Or programmatically:

```python
from scripts.query_milvus import query_collection

results = query_collection(
    queries=["What is covered by insurance?"],
    top_k=5
)
```

## Project Structure

```
RAGLab/
├── config.py                   # Central configuration - edit for experimentation
├── chunking/                   # Document processing + chunking
|   ├── __init__.py            
│   ├── base.py                 # Base classes
│   └── pdf_chunker.py          # PDF implementation
├── embedding/                  # Vector embeddings
|   ├── __init__.py
│   └── embedding_manager.py
├── vector_db/                   # Database clients
|   ├── __init__.py
│   └── milvus_client.py         # [Milvus] connection + disconnect + ingest (schema creation, insert data, index creation) + query/search 
├── utils/                       # Utilities
|   ├── __init__.py
|   ├── logger.py                  
│   └── storage.py
└── scripts/                     # Executable scripts
    ├── run_chunking.py          # Entry 1 - Chunking Interface
    ├── run_ingestion_milvus.py  # Entry 2 - Embedding + VectorDB ingestion Interface
    └── run_query_milvus.py      # Entry 3 - Querying Interface
```

## Use Cases

### 1. Experiment with Different Embedding Models

1. Implement a new base embedder model in `embedding/embedding_manager.py`:

```python
class NewEmbedder(BaseEmbedder):
    
    def embed(self, texts: List[str]) -> np.ndarray:
        pass
    
    def get_dimension(self) -> int:
        """Return the embedding dimension."""
        pass
    
    def model_name(self) -> str:
        """Return the model name."""
        pass

```

2. Re-Embed and ingest without re-chunking

```bash
# 1. (DO NOT RE_RUN IF RAN ONCE BEFORE) Chunk documents once 
uv run scripts/run_chunking.py

# 2. Change the embedding model in config.py
# ACTIVE_EMBEDDING_PROVIDER = "new-embedding-model"

# 3. Embed and ingest without re-chunking
uv run scripts/run_ingestion_milvus.py 
```

### 2. Process New Document Types

1. Create a new chunker in `chunking/`:

```python
from chunking.base import BaseChunker

class MyChunker(BaseChunker):
    def chunk(self, file_path):
        # Your implementation
        pass
    
    def get_metadata_schema(self):
        return {"field1": str, "field2": int}
```

2. Add to config and use!
- Add in CHUNKING_CONFIG, chunker to use for new document type
- Add in EMBEDDING_CONFIG, the embedding model to use for the new document type

### 3. Switch Vector Databases

1. Create a new client in `vector_db/`:

```python
class PineconeClient:
    # Implement the same interface as MilvusClient
    pass
```

2. Update imports in scripts
3. No changes needed to chunking/embedding!

## Configuration Guide

### Chunking Parameters

```python
CHUNKING_CONFIG = {
    "pdf": {
        "image_coverage_threshold": 0.15,  # Trigger vision at 15% image coverage
        "vision_model": "gpt-4o",         # Vision model for image-heavy pages
        "log_level": "INFO"
    }
}
```

### Embedding Options

```python
# OpenAI embeddings
EMBEDDING_CONFIG = {
    "text": {
        "openai": {
            "model": "text-embedding-3-large",  # 3072 dim
            "batch_size": 64,
            "normalize": True
        }
    }
}

# Sentence Transformers
EMBEDDING_CONFIG = {
    "text": {
        "sentence_transformers": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "batch_size": 64,
            "normalize": True
        }
    }
}
```

### Vector Index Configuration

```python
MILVUS_CONFIG = {
    "index": {
        "type": "IVF_FLAT",        # Options: HNSW, IVF_FLAT, IVF_PQ
        "metric_type": "IP",        # Options: IP, L2, COSINE
        "params": {"nlist": 1024}  # Index-specific parameters
    }
}
```

## Extending the Pipeline

### Adding Hybrid Search

1. Create `vector_db/hybrid_search.py`
2. Implement BM25 + dense retrieval
3. Use existing components for dense vectors

### Adding Code Embeddings

1. Extend `embedding_manager.py`:

```python
class CodeEmbedder(BaseEmbedder):
    def embed(self, texts):
        # Use code-specific model
        pass
```

2. Update config:

```python
EMBEDDING_CONFIG = {
    "code": {
        "openai": {"model": "text-embedding-3-large"}
    }
}
```

### Custom Metadata Fields

Just extend the metadata class:

```python
@dataclass
class MyPDFMetadata(PDFChunkMetadata):
    custom_field: str = ""
    another_field: int = 0
```

Schema automatically generated in Milvus!

## Logging

Logs are written to:
- Console (INFO level)
- File: `logs/rag_pipeline.log` (DEBUG level)

Configure in `config.py`:

```python
LOGGING_CONFIG = {
    "handlers": {
        "console": {"level": "INFO"},
        "file": {"level": "DEBUG"}
    }
}
```

## Troubleshooting

### Milvus Connection Failed

```bash
# Check if Milvus is running
docker ps | grep milvus

# Restart Milvus
docker-compose restart
```

### Out of Memory During Embedding

Reduce batch size in `config.py`:

```python
EMBEDDING_CONFIG = {
    "text": {
        "openai": {
            "batch_size": 32  # Lower if needed
        }
    }
}
```

### Vision Processing Errors

Check OpenAI API key and rate limits:
- GPT-4o vision requires valid API access
- Consider reducing `image_coverage_threshold` to process fewer pages with vision

## Performance Tips

1. **Chunk Reuse**: Save chunks to avoid re-processing documents
2. **Batch Size**: Tune embedding batch size for your hardware
3. **Index Selection**: 
   - HNSW: Best for accuracy, slower build
   - IVF_FLAT: Balanced
   - IVF_PQ: Fastest, uses quantization
4. **Vision Processing**: Only use for truly image-heavy pages (adjust threshold)

## License

MIT License - see LICENSE file

## Contributing

Contributions welcome! Please:
1. Follow the modular architecture
2. Add tests for new features
3. Update documentation
4. Follow PEP 8 style guide

## Support

For issues or questions:
- Check [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed docs
- Open an issue on GitHub
- Review logs in `logs/rag_pipeline.log`

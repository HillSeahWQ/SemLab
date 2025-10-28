# SemantIQ — A Framework for Multimodal Semantic Retrieval Experiments

## Overview
**SemantIQ** is a modular and extensible framework designed to benchmark and optimize **semantic retrieval** of systems like RAG, Recommendation Systems, ... etc.

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
- Distributed vector database with production-ready features
- Supports multiple index types (HNSW, IVF_FLAT, IVF_PQ)
- Requires Docker setup (see below)
- Best for: Production deployments, large-scale datasets

**FAISS (Facebook AI Similarity Search)**  
- File-based local vector storage
- CPU and GPU support available
- Multiple index types (Flat, IVF, HNSW)
- Best for: Development, experimentation, small-to-medium datasets
- No server setup required

*Extensible: Add Chroma, Pinecone, Weaviate, or other vector databases with configurable parameters.*

---

## Quick Start

### 0. Prerequisites

Before getting started, ensure you have the following installed:

| Requirement | Version | Notes |
|--------------|----------|--------|
| **Python** | ≥ 3.12 | Recommended to use a virtual environment |
| **uv** | Latest | Fast Python package manager ([uv documentation](https://docs.astral.sh/uv/)) |
|**Docker** | Latest | Required for running local vector databases or external services |

### 1. Installation

```bash
# Clone repository
git clone <repo>
cd SemantIQ

# Install dependencies
uv sync

# Setup environment variables
cat > .env << EOF
OPENAI_API_KEY=your_openai_key_here
EOF
```

### 2. Start Milvus (If Vector DB choice = Milvus)

```bash
# Using Docker Compose (recommended)
# Download docker-compose.yml from Milvus documentation
docker-compose -f docker-compose-milvus-standalone.yml up -d

# Verify Milvus is running
curl http://localhost:19530/healthz
```

### 3. Configure Pipeline - See more under Configuration Guide

Edit `config.py`:

```python
# Set your input directory to documents folder
INPUT_DIR = DATA_DIR / "your-documents"

# Choose embedding model - provider to use
ACTIVE_EMBEDDING_PROVIDER = "openai"  # or "sentence_transformers"

# Choose desired supported vector database
ACTIVE_VECTOR_DB = "faiss"  # Options: "milvus", "faiss"
```

### 4. Run Ingestion Pipeline

```bash
# Full pipeline: Chunk → Embed → Ingest (split up in case you want to test different vector DB/embedding model with existing chunks after run_chunking.py)

# Step 1: Chunk documents (run once)
uv run scripts/run_chunking.py

# Step 2: Embed and ingest to your chosen vector database

# For FAISS:
uv run scripts/ingest_to_faiss.py

# For Milvus:
uv run scripts/ingest_to_milvus.py
```

### 5. Query Your Data

```bash
# Run example queries - (automatically uses ACTIVE_VECTOR_DB)
uv run scripts/query_vector_db.py
```

Or programmatically:

```python
from scripts.query_vector_db import query_vector_db

results = query_vector_db(
    queries=["What is covered by insurance?"],
    top_k=5
)
```

## Project Structure

```
SemantIQ/
├── config.py                          # Central configuration - edit for experimentation
├── docker-compose-milvus-standalone.yml  # Milvus Docker setup
├── chunking/                          # Document processing + chunking
│   ├── __init__.py            
│   ├── base.py                        # Base classes
│   └── pdf_chunker.py                 # PDF implementation
├── embedding/                         # Vector embeddings
│   ├── __init__.py
│   └── embedding_manager.py
├── vector_db/                         # Database clients
│   ├── __init__.py
│   ├── milvus_client.py               # Milvus implementation
│   └── faiss_client.py                # FAISS implementation
├── utils/                             # Utilities
│   ├── __init__.py
│   ├── logger.py                  
│   └── storage.py
├── scripts/                           # Executable scripts
│   ├── run_chunking.py                # Entry 1 - Chunking Interface
│   ├── ingest_to_milvus.py            # Entry 2a - Milvus ingestion
│   ├── ingest_to_faiss.py             # Entry 2b - FAISS ingestion
│   └── query_vector_db.py             # Entry 3 - Unified querying (Detects choosen active vectordb)
└── data/
    ├── your-documents-to-ingest-folder/ 
    ├── chunks/                        # Stored document chunks
    └── faiss_indices/                 # FAISS index files
```
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
        # OpenAI embeddings
        "openai": {
            "model": "text-embedding-3-large",  # or "text-embedding-3-small"
            "batch_size": 64,
            "normalize": True,
            "dimensions": 3072  # 3072 for large, 1536 for small
        },
        # Sentence Transformers
        "sentence_transformers": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "batch_size": 64,
            "normalize": True,
            "dimensions": 384
        }
    },
    # Future: code embeddings
    # "code": {
    #     "openai": {
    #         "model": "text-embedding-3-large",
    #         ...
    #     }
    # }
}
```

### Vector Database Selection

```python
# Choose your vector database
ACTIVE_VECTOR_DB = "faiss"  # Options: "faiss", "milvus"
```

### FAISS Configuration

```python
FAISS_CONFIG = {
    "index": {
        "index_dir": "data/faiss_indices",
        "name": "document_embeddings",
        "index_type": "Flat",      # Options: Flat, IVF, HNSW
        "metric_type": "IP",       # Options: IP (inner product), L2
        "normalize": True,         # True for cosine similarity with IP
        "use_gpu": False,          # Set True if using faiss-gpu
        "gpu_id": 0,              # GPU device ID
        "params": {
            # For IVF: {"nlist": 100}
            # For HNSW: {"M": 32}
        }
    },
    "search": {
        "top_k": 5,
        "params": {
            # For IVF: {"nprobe": 10}
        }
    }
}
```

**Index Type Selection:**

- **Flat**: Exact search, best accuracy, slower for large datasets
- **IVF**: Fast approximate search with clustering
- **HNSW**: Best speed/accuracy tradeoff for production

### Milvus Configuration

```python
MILVUS_CONFIG = {
    "connection": {
        "host": "localhost",
        "port": "19530",
        "alias": "default"
    },
    "collection": {
        "name": "document_embeddings",
        "description": "Document embeddings with metadata"
    },
    "index": {
        "index_type": "IVF_FLAT",  # Options: HNSW, IVF_FLAT, IVF_PQ
        "metric_type": "IP",        # Options: IP, L2, COSINE
        "params": {"nlist": 1024}  # Index-specific parameters
    },
    "search": {
        "top_k": 5,
        "params": {}  # e.g., {"nprobe": 10} for IVF
    }
}
```

## Use Cases

### 1. Experiment with Different Embedding Models

1. Implement a new base embedder model in `embedding/embedding_manager.py`:

```python
class NewEmbedder(BaseEmbedder):
    
    def embed(self, texts: List[str]) -> np.ndarray:
        # Your implementation
        pass

```

2. Re-Embed and ingest without re-chunking

```bash
# 1. (DO NOT RE_RUN IF RAN ONCE BEFORE) Chunk documents once 
uv run scripts/run_chunking.py

# 2. Change the embedding model in config.py
# ACTIVE_EMBEDDING_PROVIDER = "new-embedding-model"

# 3. Embed and ingest without re-chunking - e.g. with milvus
uv run scripts/ingest_to_milvus.py 
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
        # Your implementation
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

## Extending the Pipeline

### Adding a New Vector Database

1. Create a new client in `vector_db/`:

```python
class PineconeClient:
    def __init__(self, ...):
        pass
    
    def create_index(self, ...):
        pass
    
    def ingest_data(self, embeddings, contents, metadatas):
        pass
    
    def search(self, query_embeddings, top_k):
        pass
    
    def get_index_stats(self):
        pass
```

2. Add configuration to `config.py`:

```python
PINECONE_CONFIG = {
    "api_key": "...",
    "index_name": "...",
    ...
}
```

3. Create ingestion and query scripts in `scripts/`

4. Update `ACTIVE_VECTOR_DB` options


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
## Support

For issues or questions:
- Open an issue on GitHub
- Review logs in `logs/`

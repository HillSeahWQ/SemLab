"""
Configuration file for RAG pipeline experiments.
Modify these settings to experiment with different configurations.
"""
from pathlib import Path
from typing import Dict, Any

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "kyndryl-docs-test"
CHUNKS_DIR = DATA_DIR / "chunks"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "detailed": {
            "format": "%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "default",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": str(LOGS_DIR / "rag_pipeline.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5
        }
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console", "file"]
    }
}

# ============================================================================
# CHUNKING CONFIGURATION
# ============================================================================
CHUNKING_CONFIG = {
    "pdf": {
        "chunker_class": "MultimodalPDFChunker",
        "image_coverage_threshold": 0.15,  # 15% triggers vision processing
        "vision_model": "gpt-4o",
        "log_level": "INFO"
    },
    # Add more document types here as needed
    # "docx": {
    #     "chunker_class": "DocxChunker",
    #     ...
    # },
}

# ============================================================================
# EMBEDDING CONFIGURATION
# ============================================================================
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

# Current embedding provider to use
ACTIVE_EMBEDDING_PROVIDER = "openai"  # or "sentence_transformers"
ACTIVE_EMBEDDING_TYPE = "text"

# ============================================================================
# VECTOR DATABASE SELECTION
# ============================================================================
ACTIVE_VECTOR_DB = "faiss"  # Options: "milvus", "faiss"

# ============================================================================
# VECTOR DATABASE CONFIGURATION - MILVUS
# ============================================================================
MILVUS_CONFIG = {
    "connection": {
        "host": "localhost",
        "port": "19530",
        "alias": "default"
    },
    "collection": {
        "name": "kyndryl_document_embeddings",
        "description": "Document embeddings with full chunk metadata"
    },
    "index": {
        "index_type": "IVF_FLAT",  # Options: HNSW, IVF_FLAT, IVF_PQ, etc.
        "metric_type": "IP",  # Options: IP (inner product), L2, COSINE
        "params": {
            "nlist": 1024  # For IVF_FLAT
            # For HNSW: {"M": 16, "efConstruction": 200}
        }
    },
    "search": {
        "top_k": 5,
        "params": {}  # Index-specific search params, e.g., {"nprobe": 10} for IVF
    }
}

# ============================================================================
# VECTOR DATABASE CONFIGURATION - FAISS
# ============================================================================
FAISS_CONFIG = {
    "index": {
        "index_dir": str(DATA_DIR / "faiss_indices"),
        "name": "kyndryl_document_embeddings", # TODO - EDIT HERE
        "index_type": "Flat",  # Options: Flat, IVF, HNSW
        "metric_type": "IP",  # Options: IP (inner product), L2
        "normalize": True,  # True for cosine similarity with IP metric
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

# ============================================================================
# EXPERIMENT TRACKING
# ============================================================================
EXPERIMENT_CONFIG = {
    "name": "kyndryl_pdfs",
    "description": "Initial RAG pipeline with PDF multimodal chunking",
    "version": "1.0.0",
    "tags": ["pdf", "multimodal", "milvus", "openai"]
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_embedding_config() -> Dict[str, Any]:
    """Get active embedding configuration."""
    return EMBEDDING_CONFIG[ACTIVE_EMBEDDING_TYPE][ACTIVE_EMBEDDING_PROVIDER]

def get_chunk_output_path(experiment_name: str = None) -> Path:
    """Get output path for chunks."""
    if experiment_name:
        return CHUNKS_DIR / f"{experiment_name}_chunks.json"
    return CHUNKS_DIR / f"{EXPERIMENT_CONFIG['name']}_chunks.json"

def get_collection_name() -> str:
    """Get Milvus collection name."""
    return MILVUS_CONFIG["collection"]["name"]

def get_vector_db_config() -> Dict[str, Any]:
    """Get active vector database configuration."""
    if ACTIVE_VECTOR_DB == "milvus":
        return MILVUS_CONFIG
    elif ACTIVE_VECTOR_DB == "faiss":
        return FAISS_CONFIG
    else:
        raise ValueError(f"Unknown vector database: {ACTIVE_VECTOR_DB}")
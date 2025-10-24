"""
Configuration file for RAG pipeline.
Modify these settings for experimentation.
"""
from pathlib import Path

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT.parent / "data"
INPUT_DIR = DATA_DIR / "kyndryl-docs-test"
CHUNKS_DIR = DATA_DIR / "chunks"
CHUNKS_FILE = CHUNKS_DIR / "kyndryl_chunks.json"

# ============================================================================
# CHUNKING CONFIGURATION
# ============================================================================
CHUNKING = {
    "image_coverage_threshold": 0.15,  # 15% image coverage triggers vision processing
    "vision_model": "gpt-4o"
}

# ============================================================================
# EMBEDDING CONFIGURATION
# ============================================================================
EMBEDDING = {
    # Options: "openai" or "sentence_transformers"
    "provider": "openai",
    
    # OpenAI models: "text-embedding-3-small", "text-embedding-3-large"
    # Sentence Transformers: "sentence-transformers/all-MiniLM-L6-v2", etc.
    "model": "text-embedding-3-large",
    
    "batch_size": 64,
    "normalize_embeddings": True
}

# ============================================================================
# VECTOR DB CONFIGURATION (MILVUS)
# ============================================================================
MILVUS = {
    "host": "localhost",
    "port": "19530",
    "alias": "default",
    
    # Collection settings
    "collection_name": "kyndryl_document_embeddings",
    
    # Index settings
    # Options: "IVF_FLAT", "IVF_SQ8", "IVF_PQ", "HNSW", "ANNOY"
    "index_type": "IVF_FLAT",
    
    # Similarity metric options: "IP" (Inner Product), "L2", "COSINE"
    "similarity_metric": "IP",
    
    # Index hyperparameters (depends on index_type)
    # For IVF_FLAT: {"nlist": 1024}
    # For HNSW: {"M": 16, "efConstruction": 200}
    "index_params": {
        "nlist": 1024
    },
    
    # Search parameters (depends on index_type)
    # For IVF_FLAT: {"nprobe": 10}
    # For HNSW: {"ef": 200}
    "search_params": None,  # None uses defaults
    
    # Whether to drop existing collection on ingestion (True for testing)
    "drop_existing": True
}

# ============================================================================
# SEARCH CONFIGURATION
# ============================================================================
SEARCH = {
    "top_k": 5,
    "output_fields": [
        "source_file", 
        "chunk_id", 
        "preview", 
        "full",
        "page_number",
        "chunk_type"
    ]
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOGGING = {
    "level": "INFO",  # Options: "DEBUG", "INFO", "WARNING", "ERROR"
    "format": "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S"
}
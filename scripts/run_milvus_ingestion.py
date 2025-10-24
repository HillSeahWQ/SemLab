"""
Script to run embedding and ingestion into Milvus.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from chunking import load_chunks
from embedding import get_embedding_provider
from vector_db.milvus_ingestion import ingest_embeddings
from config import (
    CHUNKS_FILE,
    EMBEDDING,
    MILVUS
)
from utils.logger import setup_logger, LOGGING

logger = setup_logger(__name__, **LOGGING)


def main():
    """Run embedding and ingestion pipeline."""
    logger.info("Starting ingestion pipeline...")
    
    # Load environment variables (for API keys)
    load_dotenv()
    
    # Load chunks
    if not CHUNKS_FILE.exists():
        logger.error(f"Chunks file not found: {CHUNKS_FILE}")
        logger.error("Please run chunking first: python scripts/run_chunking.py")
        return
    
    chunks_dict = load_chunks(CHUNKS_FILE)
    chunks = [chunk["content"] for chunk in chunks_dict]
    metadatas = [chunk["metadata"] for chunk in chunks_dict]
    
    logger.info(f"Loaded {len(chunks)} chunks for embedding")
    
    # Get embedding provider
    embedding_provider = get_embedding_provider(
        provider=EMBEDDING["provider"],
        model=EMBEDDING["model"],
        batch_size=EMBEDDING["batch_size"],
        normalize_embeddings=EMBEDDING["normalize_embeddings"]
    )
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    embeddings = embedding_provider.embed(chunks)
    logger.info(f"Generated embeddings with shape: {embeddings.shape}")
    
    # Ingest into Milvus
    logger.info("Ingesting into Milvus...")
    ingest_embeddings(
        embeddings=embeddings,
        chunks=chunks,
        metadatas=metadatas,
        collection_name=MILVUS["collection_name"],
        similarity_metric=MILVUS["similarity_metric"],
        index_type=MILVUS["index_type"],
        index_params=MILVUS["index_params"],
        host=MILVUS["host"],
        port=MILVUS["port"],
        alias=MILVUS["alias"],
        drop_existing=MILVUS["drop_existing"]
    )
    
    logger.info("✅ Ingestion pipeline complete!")


if __name__ == "__main__":
    main()
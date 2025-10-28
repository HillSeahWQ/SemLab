"""
FAISS ingestion script - loads chunks, generates embeddings, and ingests to FAISS.
Run this script after chunking to ingest documents into the FAISS vector database.
"""
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    get_embedding_config,
    get_chunk_output_path,
    FAISS_CONFIG,
    ACTIVE_EMBEDDING_PROVIDER,
    ACTIVE_EMBEDDING_TYPE,
    EXPERIMENT_CONFIG
)
from embedding.embedding_manager import EmbeddingManager
from vector_db.faiss_client import FAISSClient
from utils.storage import load_chunks, print_chunk_statistics
from utils.logger import get_logger

logger = get_logger(__name__)


def main(
    chunk_file: Path = None,
    index_name: str = None,
    drop_existing: bool = True
):
    """
    Run FAISS ingestion pipeline.
    
    Args:
        chunk_file: Path to chunks JSON file (uses config default if None)
        index_name: Name of FAISS index (uses config default if None)
        drop_existing: If True, drop existing index before creating new one
    """
    logger.info("="*80)
    logger.info("FAISS INGESTION PIPELINE")
    logger.info("="*80)
    logger.info(f"Experiment: {EXPERIMENT_CONFIG['name']}")
    logger.info(f"Description: {EXPERIMENT_CONFIG['description']}")
    logger.info(f"Version: {EXPERIMENT_CONFIG['version']}")
    logger.info("")
    
    # Use defaults from config if not provided
    if chunk_file is None:
        chunk_file = get_chunk_output_path()
    if index_name is None:
        index_name = FAISS_CONFIG["index"]["name"]
    
    try:
        # ================================================================
        # STEP 1: LOAD CHUNKS
        # ================================================================
        logger.info("STEP 1: LOADING CHUNKS FROM JSON")
        logger.info("-"*80)
        logger.info(f"Loading from: {chunk_file}")
        
        if not chunk_file.exists():
            raise FileNotFoundError(
                f"Chunk file not found: {chunk_file}\n"
                f"Please run chunking first: python scripts/run_chunking.py"
            )
        
        contents, metadatas = load_chunks(chunk_file)
        
        logger.info(f"Loaded {len(contents)} chunks")
        logger.info("")
        print_chunk_statistics(metadatas)
        
        # ================================================================
        # STEP 2: GENERATE EMBEDDINGS
        # ================================================================
        logger.info("")
        logger.info("STEP 2: GENERATING EMBEDDINGS")
        logger.info("-"*80)
        
        # Load environment variables for API keys
        load_dotenv()
        
        # Get embedding configuration
        embed_config = get_embedding_config()
        
        logger.info(f"Embedding provider: {ACTIVE_EMBEDDING_PROVIDER}")
        logger.info(f"Embedding type: {ACTIVE_EMBEDDING_TYPE}")
        
        # Create embedder
        embedder = EmbeddingManager.create_embedder(
            provider=ACTIVE_EMBEDDING_PROVIDER,
            embedding_type=ACTIVE_EMBEDDING_TYPE,
            config=embed_config
        )
        
        logger.info(f"Model: {embedder.model_name}")
        logger.info(f"Embedding dimension: {embedder.get_dimension()}")
        logger.info(f"Batch size: {embed_config.get('batch_size', 'default')}")
        
        # Generate embeddings
        embeddings = embedder.embed(contents)
        
        logger.info(f"Embeddings generated successfully")
        logger.info(f"   Shape: {embeddings.shape}")
        logger.info(f"   Dtype: {embeddings.dtype}")
        
        # ================================================================
        # STEP 3: INITIALIZE FAISS CLIENT
        # ================================================================
        logger.info("")
        logger.info("STEP 3: INITIALIZING FAISS CLIENT")
        logger.info("-"*80)
        
        index_dir = FAISS_CONFIG["index"]["index_dir"]
        
        logger.info(f"Index directory: {index_dir}")
        logger.info(f"Index name: {index_name}")
        
        client = FAISSClient(
            index_dir=index_dir,
            index_name=index_name
        )
        
        # ================================================================
        # STEP 4: CREATE INDEX
        # ================================================================
        logger.info("")
        logger.info("STEP 4: CREATING FAISS INDEX")
        logger.info("-"*80)
        logger.info(f"Drop existing: {drop_existing}")
        
        index_config = FAISS_CONFIG["index"]
        
        logger.info(f"Index type: {index_config['index_type']}")
        logger.info(f"Metric type: {index_config['metric_type']}")
        logger.info(f"Index params: {index_config.get('params', {})}")
        
        # Determine if normalization is needed (for cosine similarity)
        normalize = index_config.get("normalize", False)
        if normalize:
            logger.info("Normalization enabled for cosine similarity")
        
        client.create_index(
            embedding_dim=embedder.get_dimension(),
            index_type=index_config["index_type"],
            metric_type=index_config["metric_type"],
            index_params=index_config.get("params", {}),
            drop_existing=drop_existing
        )
        
        logger.info(f"Index ready: {index_name}")
        
        # ================================================================
        # STEP 5: INGEST DATA
        # ================================================================
        logger.info("")
        logger.info("STEP 5: INGESTING DATA")
        logger.info("-"*80)
        
        client.ingest_data(
            embeddings=embeddings,
            contents=contents,
            metadatas=metadatas,
            normalize=normalize
        )
        
        # ================================================================
        # STEP 6: VERIFY INGESTION
        # ================================================================
        logger.info("")
        logger.info("STEP 6: VERIFYING INGESTION")
        logger.info("-"*80)
        
        stats = client.get_index_stats()
        
        logger.info(f"Index statistics:")
        logger.info(f"   • Name: {stats['name']}")
        logger.info(f"   • Total vectors: {stats['num_vectors']:,}")
        logger.info(f"   • Dimension: {stats['dimension']}")
        logger.info(f"   • Index type: {stats['index_type']}")
        logger.info(f"   • Metadata entries: {stats['num_metadata']:,}")
        logger.info(f"   • Index path: {stats['index_path']}")
        
        # ================================================================
        # FINAL SUMMARY
        # ================================================================
        logger.info("")
        logger.info("="*80)
        logger.info("[SUCCESS] - INGESTION PIPELINE COMPLETE")
        logger.info("="*80)
        logger.info(f"Chunks processed: {len(contents)}")
        logger.info(f"Embedding model: {embedder.model_name}")
        logger.info(f"Embedding dimension: {embedder.get_dimension()}")
        logger.info(f"Index name: {index_name}")
        logger.info(f"Total vectors: {stats['num_vectors']:,}")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  • Query your data: python scripts/query_faiss.py")
        logger.info("  • View logs: tail -f logs/rag_pipeline.log")
        logger.info("="*80)
        
        return {
            "index_name": index_name,
            "num_vectors": stats['num_vectors'],
            "embedding_model": embedder.model_name,
            "embedding_dim": embedder.get_dimension()
        }
        
    except Exception as e:
        logger.error(f"[ERROR] - Ingestion pipeline failed: {e}")
        logger.exception("Full error traceback:")
        raise


if __name__ == "__main__":
    # Run with defaults from config
    main()
    
    # Or customize:
    # main(
    #     chunk_file=Path("data/chunks/my_chunks.json"),
    #     index_name="my_custom_index",
    #     drop_existing=False  # Keep existing data
    # )
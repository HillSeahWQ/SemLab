"""
Milvus ingestion script - loads chunks, generates embeddings, and ingests to Milvus.
Run this script after chunking to ingest documents into the Milvus vector database.
"""
import logging
import logging.config
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    get_embedding_config,
    get_chunk_output_path,
    MILVUS_CONFIG,
    ACTIVE_EMBEDDING_PROVIDER,
    ACTIVE_EMBEDDING_TYPE,
    LOGGING_CONFIG,
    EXPERIMENT_CONFIG
)
from embedding.embedding_manager import EmbeddingManager
from vector_db.milvus_client import MilvusClient
from utils.storage import load_chunks, print_chunk_statistics
from utils.logger import get_logger

logger = get_logger(__name__)


def main(
    chunk_file: Path = None,
    collection_name: str = None,
    drop_existing: bool = True
):
    """
    Run Milvus ingestion pipeline.
    
    Args:
        chunk_file: Path to chunks JSON file (uses config default if None)
        collection_name: Name of Milvus collection (uses config default if None)
        drop_existing: If True, drop existing collection before creating new one
    """
    logger.info("="*80)
    logger.info("MILVUS INGESTION PIPELINE")
    logger.info("="*80)
    logger.info(f"Experiment: {EXPERIMENT_CONFIG['name']}")
    logger.info(f"Description: {EXPERIMENT_CONFIG['description']}")
    logger.info(f"Version: {EXPERIMENT_CONFIG['version']}")
    logger.info("")
    
    # Use defaults from config if not provided
    if chunk_file is None:
        chunk_file = get_chunk_output_path()
    if collection_name is None:
        collection_name = MILVUS_CONFIG["collection"]["name"]
    
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
        # STEP 3: CONNECT TO MILVUS
        # ================================================================
        logger.info("")
        logger.info("STEP 3: CONNECTING TO MILVUS")
        logger.info("-"*80)
        
        milvus_conn = MILVUS_CONFIG["connection"]
        
        logger.info(f"Host: {milvus_conn['host']}")
        logger.info(f"Port: {milvus_conn['port']}")
        
        client = MilvusClient(
            host=milvus_conn["host"],
            port=milvus_conn["port"],
            alias=milvus_conn["alias"]
        )
        
        client.connect()
        
        # ================================================================
        # STEP 4: CREATE COLLECTION WITH AUTO SCHEMA
        # ================================================================
        logger.info("")
        logger.info("STEP 4: CREATING/LOADING COLLECTION")
        logger.info("-"*80)
        logger.info(f"Collection name: {collection_name}")
        logger.info(f"Drop existing: {drop_existing}")
        
        # Infer schema from metadata
        if metadatas:
            schema_dict = {}
            for key, value in metadatas[0].items():
                if isinstance(value, int):
                    schema_dict[key] = int
                elif isinstance(value, float):
                    schema_dict[key] = float
                elif isinstance(value, bool):
                    schema_dict[key] = bool
                else:
                    schema_dict[key] = str
            
            logger.info(f"Auto-detected schema fields: {len(schema_dict)}")
            logger.debug(f"Schema: {schema_dict}")
        
        client.create_collection_from_schema(
            collection_name=collection_name,
            metadata_schema=schema_dict,
            embedding_dim=embedder.get_dimension(),
            description=MILVUS_CONFIG["collection"]["description"],
            drop_existing=drop_existing
        )
        
        logger.info(f"Collection ready: {collection_name}")
        
        # ================================================================
        # STEP 5: INGEST DATA WITH INDEXING
        # ================================================================
        logger.info("")
        logger.info("STEP 5: INGESTING DATA")
        logger.info("-"*80)
        
        index_config = MILVUS_CONFIG["index"]
        
        logger.info(f"Index type: {index_config['index_type']}")
        logger.info(f"Metric type: {index_config['metric_type']}")
        logger.info(f"Index params: {index_config['params']}")
        
        client.ingest_data(
            collection_name=collection_name,
            embeddings=embeddings,
            contents=contents,
            metadatas=metadatas,
            index_config=index_config
        )
        
        # ================================================================
        # STEP 6: VERIFY INGESTION
        # ================================================================
        logger.info("")
        logger.info("STEP 6: VERIFYING INGESTION")
        logger.info("-"*80)
        
        stats = client.get_collection_stats(collection_name)
        
        logger.info(f"Collection statistics:")
        logger.info(f"   • Name: {stats['name']}")
        logger.info(f"   • Total entities: {stats['num_entities']:,}")
        logger.info(f"   • Schema fields: {len(stats['schema'])}")
        
        client.disconnect()
        
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
        logger.info(f"Collection: {collection_name}")
        logger.info(f"Total entities: {stats['num_entities']:,}")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  • Query your data: python scripts/query_milvus.py")
        logger.info("  • View logs: tail -f logs/rag_pipeline.log")
        logger.info("="*80)
        
        return {
            "collection_name": collection_name,
            "num_entities": stats['num_entities'],
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
    #     collection_name="my_custom_collection",
    #     drop_existing=False  # Keep existing data
    # )
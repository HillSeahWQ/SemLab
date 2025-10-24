"""
Script for querying the Milvus vector database.
Run this script to test retrieval with different queries.
"""
import logging
import logging.config
import sys
from pathlib import Path
from dotenv import load_dotenv
from typing import List

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    get_embedding_config,
    MILVUS_CONFIG,
    ACTIVE_EMBEDDING_PROVIDER,
    ACTIVE_EMBEDDING_TYPE,
    LOGGING_CONFIG
)
from embedding.embedding_manager import EmbeddingManager
from vector_db.milvus_client import MilvusClient
from utils.logger import get_logger

logger = get_logger(__name__)

def query_collection(
    queries: List[str],
    top_k: int = None,
    output_fields: List[str] = None
):
    """
    Query the Milvus collection.
    
    Args:
        queries: List of query strings
        top_k: Number of results to return (uses config default if None)
        output_fields: Fields to return in results
    """
    logger.info("="*80)
    logger.info("QUERYING MILVUS VECTOR DATABASE")
    logger.info("="*80)
    
    # Load environment variables
    load_dotenv()
    
    # Get embedding config
    embed_config = get_embedding_config()
    
    # Create embedder (same as used during ingestion)
    embedder = EmbeddingManager.create_embedder(
        provider=ACTIVE_EMBEDDING_PROVIDER,
        embedding_type=ACTIVE_EMBEDDING_TYPE,
        config=embed_config
    )
    
    logger.info(f"Using embedder: {embedder.model_name}")
    
    # Embed queries
    logger.info(f"Embedding {len(queries)} queries...")
    query_embeddings = embedder.embed(queries)
    
    # Initialize Milvus client
    milvus_conn = MILVUS_CONFIG["connection"]
    client = MilvusClient(
        host=milvus_conn["host"],
        port=milvus_conn["port"],
        alias=milvus_conn["alias"]
    )
    
    # Connect and search
    client.connect()
    
    collection_name = MILVUS_CONFIG["collection"]["name"]
    search_config = MILVUS_CONFIG["search"]
    
    if top_k is None:
        top_k = search_config["top_k"]
    
    if output_fields is None:
        output_fields = [
            "source_file",
            "id",
            "page_number",
            "chunk_type",
            "preview",
            "content"
        ]
    
    logger.info(f"Searching collection: {collection_name}")
    logger.info(f"Top K: {top_k}")
    logger.info(f"Metric: {MILVUS_CONFIG['index']['metric_type']}")
    
    results = client.search(
        collection_name=collection_name,
        query_embeddings=query_embeddings,
        top_k=top_k,
        metric_type=MILVUS_CONFIG["index"]["metric_type"],
        search_params=search_config["params"],
        output_fields=output_fields
    )
    
    client.disconnect()
    
    # Display results
    logger.info("")
    logger.info("="*80)
    logger.info("SEARCH RESULTS")
    logger.info("="*80)
    
    for i, (query, query_results) in enumerate(zip(queries, results)):
        logger.info("")
        logger.info(f"Query {i+1}: {query}")
        logger.info("-"*80)
        
        for rank, hit in enumerate(query_results, start=1):
            logger.info(f"\n  Rank {rank} (Score: {hit['score']:.4f})")
            logger.info(f"  Source: {hit.get('source_file', 'N/A')}")
            logger.info(f"  Page: {hit.get('page_number', 'N/A')}")
            logger.info(f"  Chunk Type: {hit.get('chunk_type', 'N/A')}")
            logger.info(f"  Preview: {hit.get('preview', '')[:150]}...")
        
        logger.info("")
    
    logger.info("="*80)
    
    return results


def main():
    """Run example queries."""
    
    # Example queries - modify these for your use case
    queries = [
        "How much does Kyndryl cover for surgeries",
        "What are the hospitals covered?"
    ]
    
    logger.info(f"Running {len(queries)} example queries...")
    
    results = query_collection(
        queries=queries,
        top_k=5  # Can be customized
    )
    
    logger.info(f"\n[SUCCESS] - Query complete: retrieved {len(results)} result sets")


if __name__ == "__main__":
    main()
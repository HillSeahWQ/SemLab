"""
Unified script for querying vector databases (Milvus or FAISS).
Automatically uses the active vector database from config.
Supports saving results for evaluation.
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    get_embedding_config,
    ACTIVE_VECTOR_DB,
    MILVUS_CONFIG,
    FAISS_CONFIG,
    ACTIVE_EMBEDDING_PROVIDER,
    ACTIVE_EMBEDDING_TYPE,
    EXPERIMENT_CONFIG,
    QUERY_RESULTS_DIR
)
from embedding.embedding_manager import EmbeddingManager
from vector_db.milvus_client import MilvusClient
from vector_db.faiss_client import FAISSClient
from utils.logger import get_logger

logger = get_logger(__name__)


def query_vector_db(
    queries: List[str],
    top_k: int = None,
    output_fields: List[str] = None,
    vector_db: str = None,
    query_ids: Optional[List[str]] = None
):
    """
    Query the active vector database (Milvus or FAISS).
    
    Args:
        queries: List of query strings
        top_k: Number of results to return (uses config default if None)
        output_fields: Fields to return in results
        vector_db: Override active vector DB ("milvus" or "faiss")
        query_ids: Optional list of query IDs for result tracking
        
    Returns:
        Dictionary with results and metadata
    """
    # Use active vector DB from config if not specified
    if vector_db is None:
        vector_db = ACTIVE_VECTOR_DB
    
    logger.info("="*80)
    logger.info(f"QUERYING {vector_db.upper()} VECTOR DATABASE")
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
    
    # Default output fields
    if output_fields is None:
        output_fields = [
            "source_file",
            "id",
            "page_number",
            "chunk_type",
            "preview",
            "content"
        ]
    
    # Query based on vector DB type
    if vector_db == "milvus":
        results = _query_milvus(
            query_embeddings=query_embeddings,
            top_k=top_k,
            output_fields=output_fields
        )
        collection_name = MILVUS_CONFIG["collection"]["name"]
    elif vector_db == "faiss":
        results = _query_faiss(
            query_embeddings=query_embeddings,
            top_k=top_k,
            output_fields=output_fields
        )
        collection_name = FAISS_CONFIG["index"]["name"]
    else:
        raise ValueError(f"Unknown vector database: {vector_db}")
    
    # Display results
    _display_results(queries, results)
    
    # Build structured output
    if query_ids is None:
        query_ids = [f"q{i+1}" for i in range(len(queries))]
    
    structured_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "vector_db": vector_db,
            "embedding_provider": ACTIVE_EMBEDDING_PROVIDER,
            "embedding_type": ACTIVE_EMBEDDING_TYPE,
            "embedding_model": embedder.model_name,
            "collection_name": collection_name,
            "top_k": top_k,
            "num_queries": len(queries),
            "experiment_name": EXPERIMENT_CONFIG.get("name", "unknown")
        },
        "queries": []
    }
    
    for query_id, query_text, query_results in zip(query_ids, queries, results):
        structured_results["queries"].append({
            "query_id": query_id,
            "query_text": query_text,
            "results": query_results
        })
    
    return structured_results


def _query_milvus(query_embeddings, top_k, output_fields):
    """Query Milvus vector database."""
    milvus_conn = MILVUS_CONFIG["connection"]
    client = MilvusClient(
        host=milvus_conn["host"],
        port=milvus_conn["port"],
        alias=milvus_conn["alias"]
    )
    
    client.connect()
    
    collection_name = MILVUS_CONFIG["collection"]["name"]
    search_config = MILVUS_CONFIG["search"]
    
    if top_k is None:
        top_k = search_config["top_k"]
    
    logger.info(f"Searching collection: {collection_name}")
    logger.info(f"Top K: {top_k}")
    logger.info(f"Metric: {MILVUS_CONFIG['index']['metric_type']}")
    
    results = client.search(
        collection_name=collection_name,
        query_embeddings=query_embeddings,
        top_k=top_k,
        metric_type=MILVUS_CONFIG['index']['metric_type'],
        search_params=search_config["params"],
        output_fields=output_fields
    )
    
    client.disconnect()
    return results


def _query_faiss(query_embeddings, top_k, output_fields):
    """Query FAISS vector database."""
    index_name = FAISS_CONFIG["index"]["name"]
    index_dir = FAISS_CONFIG["index"]["index_dir"]
    
    client = FAISSClient(
        index_dir=index_dir,
        index_name=index_name
    )
    
    logger.info(f"Loading index: {index_name}")
    if not client.load_index():
        raise FileNotFoundError(
            f"Index '{index_name}' not found. "
            f"Please run ingestion first: python scripts/ingest_to_faiss.py"
        )
    
    search_config = FAISS_CONFIG["search"]
    
    if top_k is None:
        top_k = search_config["top_k"]
    
    logger.info(f"Searching index: {index_name}")
    logger.info(f"Top K: {top_k}")
    logger.info(f"Metric: {FAISS_CONFIG['index']['metric_type']}")
    
    normalize = FAISS_CONFIG["index"].get("normalize", False)
    
    results = client.search(
        query_embeddings=query_embeddings,
        top_k=top_k,
        normalize=normalize,
        search_params=search_config.get("params", {}),
        output_fields=output_fields
    )
    
    return results


def _display_results(queries, results):
    """Display search results in a formatted way."""
    logger.info("")
    logger.info("="*80)
    logger.info("SEARCH RESULTS")
    logger.info("="*80)
    
    for i, (query, query_results) in enumerate(zip(queries, results)):
        logger.info("")
        logger.info(f"Query {i+1}: {query}")
        logger.info("-"*80)
        
        if not query_results:
            logger.info("  No results found")
            continue
        
        for rank, hit in enumerate(query_results, start=1):
            logger.info(f"\n  Rank {rank} (Score: {hit['score']:.4f})")
            logger.info(f"  Source: {hit.get('source_file', 'N/A')}")
            logger.info(f"  Page: {hit.get('page_number', 'N/A')}")
            logger.info(f"  Chunk Type: {hit.get('chunk_type', 'N/A')}")
            logger.info(f"  Preview: {hit.get('preview', '')[:150]}...")
        
        logger.info("")
    
    logger.info("="*80)


def save_results(results: dict, output_path: Path) -> None:
    """
    Save query results to JSON file.
    
    Args:
        results: Results dictionary from query_vector_db
        output_path: Path to save results
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving results to: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info("Results saved successfully")


def main():
    """Run example queries with optional result saving."""
    parser = argparse.ArgumentParser(description="Query vector database")
    parser.add_argument(
        "--save-results",
        type=str,
        help="Path to save query results (for evaluation)"
    )
    parser.add_argument(
        "--queries-file",
        type=str,
        help="Path to JSON file with queries and query IDs"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of results to return"
    )
    
    args = parser.parse_args()
    
    # Load queries from file or use defaults
    if args.queries_file:
        with open(args.queries_file, 'r', encoding='utf-8') as f:
            queries_data = json.load(f)
        queries = [q["query_text"] for q in queries_data["queries"]]
        query_ids = [q["query_id"] for q in queries_data["queries"]]
    else:
        # Example queries - modify these for your use case
        queries = [
            "How much does Kyndryl cover for surgeries",
            "What are the hospitals covered?"
        ]
        query_ids = None
    
    logger.info(f"Running {len(queries)} queries...")
    logger.info(f"Active vector database: {ACTIVE_VECTOR_DB}")
    
    results = query_vector_db(
        queries=queries,
        top_k=args.top_k,
        query_ids=query_ids
    )
    
    # Save results if requested
    if args.save_results:
        output_path = Path(args.save_results)
        save_results(results, output_path)
    else:
        # Auto-save to default location with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"query_results_{ACTIVE_VECTOR_DB}_{timestamp}.json"
        default_path = QUERY_RESULTS_DIR / default_filename
        logger.info(f"\nTo save results for evaluation, run with: --save-results {default_path}")
    
    logger.info(f"\n[SUCCESS] - Query complete: retrieved {len(results['queries'])} result sets")


if __name__ == "__main__":
    main()
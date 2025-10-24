"""
Script to run queries against Milvus.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from embedding import get_embedding_provider
from vector_db.milvus_query import search, format_results
from config import EMBEDDING, MILVUS, SEARCH
from utils.logger import setup_logger, LOGGING

logger = setup_logger(__name__, **LOGGING)


def main():
    """Run query pipeline."""
    logger.info("Starting query pipeline...")
    
    # Load environment variables
    load_dotenv()
    
    # Define queries
    queries = [
        "How much does Kyndryl cover for surgeries",
        "What are the hospitals covered?"
    ]
    
    logger.info(f"Running {len(queries)} queries...")
    
    # Get embedding provider
    embedding_provider = get_embedding_provider(
        provider=EMBEDDING["provider"],
        model=EMBEDDING["model"],
        batch_size=EMBEDDING["batch_size"],
        normalize_embeddings=EMBEDDING["normalize_embeddings"]
    )
    
    # Perform search
    results = search(
        queries=queries,
        embedding_provider=embedding_provider,
        collection_name=MILVUS["collection_name"],
        similarity_metric=MILVUS["similarity_metric"],
        search_params=MILVUS["search_params"],
        top_k=SEARCH["top_k"],
        output_fields=SEARCH["output_fields"],
        host=MILVUS["host"],
        port=MILVUS["port"],
        alias=MILVUS["alias"]
    )
    
    # Display results
    formatted = format_results(results, include_full_text=False)
    print(formatted)
    
    logger.info("✅ Query pipeline complete!")


if __name__ == "__main__":
    main()
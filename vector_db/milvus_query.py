"""
Milvus query module for vector similarity search.
"""
import numpy as np
from typing import List, Dict, Optional
from pymilvus import Collection
from vector_db.milvus_connection import get_connection
from embedding import EmbeddingProvider
from utils.logger import setup_logger
from config import LOGGING

logger = setup_logger(__name__, **LOGGING)


def search(
    queries: List[str],
    embedding_provider: EmbeddingProvider,
    collection_name: str,
    similarity_metric: str = "COSINE",
    search_params: Optional[Dict] = None,
    top_k: int = 5,
    output_fields: Optional[List[str]] = None,
    host: str = "localhost",
    port: str = "19530",
    alias: str = "default"
) -> List[List[Dict]]:
    """
    Perform vector similarity search in Milvus.
    
    Parameters
    ----------
    queries : List[str]
        List of query strings
    embedding_provider : EmbeddingProvider
        Provider to generate query embeddings
    collection_name : str
        Name of the Milvus collection to search
    similarity_metric : str
        Similarity metric: "COSINE", "L2", or "IP"
    search_params : Dict, optional
        Index-specific search parameters
        - For IVF_FLAT: {"nprobe": 10}
        - For HNSW: {"ef": 200}
    top_k : int
        Number of top results to return per query
    output_fields : List[str], optional
        List of fields to return in results
    host : str
        Milvus host address
    port : str
        Milvus port
    alias : str
        Connection alias
        
    Returns
    -------
    List[List[Dict]]
        List of results per query, each as a list of dicts with scores and metadata
    """
    # Connect to Milvus
    get_connection(alias=alias, host=host, port=port)
    
    # Generate query embeddings
    logger.info(f"Encoding {len(queries)} queries...")
    query_embeddings = embedding_provider.embed(queries)
    
    # Load collection
    collection = Collection(collection_name)
    collection.load()
    logger.info(f"Loaded collection: {collection_name}")
    
    # Set default output fields
    if output_fields is None:
        output_fields = ["source_file", "chunk_id", "preview", "full"]
    
    # Set default search params
    if search_params is None:
        search_params = {}
    
    # Perform search
    logger.info(f"Searching for top-{top_k} results per query...")
    results = collection.search(
        data=query_embeddings.tolist(),
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        metric_type=similarity_metric,
        output_fields=output_fields
    )
    
    # Format results
    all_results = []
    for i, hits in enumerate(results):
        query_results = []
        for hit in hits:
            result = {
                "id": hit.id,
                "score": hit.score,
                "query": queries[i]
            }
            # Add all requested output fields
            for field in output_fields:
                result[field] = hit.entity.get(field)
            
            query_results.append(result)
        all_results.append(query_results)
    
    logger.info(f"✅ Search complete for {len(queries)} queries")
    return all_results


def format_results(
    all_results: List[List[Dict]],
    include_full_text: bool = False
) -> str:
    """
    Format search results for display.
    
    Parameters
    ----------
    all_results : List[List[Dict]]
        Search results from the search function
    include_full_text : bool
        Whether to include full chunk text in output
        
    Returns
    -------
    str
        Formatted results string
    """
    output = []
    
    for query_idx, results in enumerate(all_results):
        if not results:
            continue
            
        query = results[0].get("query", f"Query {query_idx + 1}")
        output.append(f"\n{'='*80}")
        output.append(f"Query: {query}")
        output.append(f"{'='*80}")
        
        for rank, hit in enumerate(results, start=1):
            output.append(f"\n[Rank {rank}] Score: {hit['score']:.4f}")
            output.append(f"Source: {hit.get('source_file', 'N/A')}")
            output.append(f"Chunk ID: {hit.get('chunk_id', 'N/A')}")
            
            if hit.get('page_number'):
                output.append(f"Page: {hit.get('page_number')}")
            if hit.get('chunk_type'):
                output.append(f"Type: {hit.get('chunk_type')}")
            
            # Show preview or full text
            if include_full_text and hit.get('full'):
                output.append(f"\nFull Text:\n{hit['full'][:1000]}...")
            elif hit.get('preview'):
                output.append(f"\nPreview:\n{hit['preview']}")
            
            output.append("-" * 80)
    
    return "\n".join(output)
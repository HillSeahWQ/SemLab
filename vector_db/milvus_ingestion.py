"""
Milvus ingestion module for storing embeddings and metadata.
"""
import json
import numpy as np
from typing import List, Dict, Optional
from pymilvus import (
    Collection, 
    CollectionSchema, 
    FieldSchema, 
    DataType, 
    utility
)
from vector_db.milvus_connection import get_connection
from utils.logger import setup_logger
from config import LOGGING

logger = setup_logger(__name__, **LOGGING)


def create_pdf_schema(dim: int) -> CollectionSchema:
    """
    Create schema for PDF document chunks with full metadata.
    
    Parameters
    ----------
    dim : int
        Embedding dimension
        
    Returns
    -------
    CollectionSchema
        Configured schema for PDF chunks
    """
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="chunk_id", dtype=DataType.INT64),
        FieldSchema(name="preview", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="full", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="page_number", dtype=DataType.INT64),
        FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="total_pages", dtype=DataType.INT64),
        FieldSchema(name="text_length", dtype=DataType.INT64),
        FieldSchema(name="num_tables", dtype=DataType.INT64),
        FieldSchema(name="num_images", dtype=DataType.INT64),
        FieldSchema(name="image_coverage_ratio", dtype=DataType.FLOAT),
        FieldSchema(name="is_vision_processed", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="table_content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="image_details", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="bounding_boxes", dtype=DataType.VARCHAR, max_length=65535),
    ]
    
    return CollectionSchema(
        fields=fields,
        description="Document embeddings with full chunk metadata"
    )


def ingest_embeddings(
    embeddings: np.ndarray,
    chunks: List[str],
    metadatas: List[Dict],
    collection_name: str,
    similarity_metric: str = "IP",
    index_type: str = "IVF_FLAT",
    index_params: Optional[Dict] = None,
    host: str = "localhost",
    port: str = "19530",
    alias: str = "default",
    drop_existing: bool = False
) -> Collection:
    """
    Ingest embeddings and metadata into Milvus collection.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Embeddings array of shape (num_chunks, embedding_dim)
    chunks : List[str]
        List of chunk texts
    metadatas : List[Dict]
        List of metadata dictionaries for each chunk
    collection_name : str
        Name of the Milvus collection
    similarity_metric : str
        Similarity metric: "IP", "L2", or "COSINE"
    index_type : str
        Index type: "IVF_FLAT", "HNSW", etc.
    index_params : Dict, optional
        Index-specific parameters
    host : str
        Milvus host address
    port : str
        Milvus port
    alias : str
        Connection alias
    drop_existing : bool
        Whether to drop existing collection
        
    Returns
    -------
    Collection
        The created/updated Milvus collection
    """
    # Connect to Milvus
    logger.info(f"Connecting to Milvus at {host}:{port}")
    get_connection(alias=alias, host=host, port=port)
    
    # Handle existing collection
    if utility.has_collection(collection_name):
        if drop_existing:
            logger.warning(f"Dropping existing collection: {collection_name}")
            utility.drop_collection(collection_name)
        else:
            logger.info(f"Using existing collection: {collection_name}")
            collection = Collection(collection_name)
            logger.info("Appending to existing collection")
            # Insert data and return
            _insert_data(collection, embeddings, chunks, metadatas)
            return collection
    
    # Create new collection
    logger.info(f"Creating new collection: {collection_name}")
    dim = embeddings.shape[1]
    schema = create_pdf_schema(dim)
    collection = Collection(name=collection_name, schema=schema)
    
    # Insert data
    _insert_data(collection, embeddings, chunks, metadatas)
    
    # Create index
    logger.info(f"Creating index on 'embedding' field...")
    if index_params is None:
        index_params = {"nlist": 1024}  # default for IVF_FLAT
    
    index_config = {
        "metric_type": similarity_metric,
        "index_type": index_type,
        "params": index_params
    }
    collection.create_index(field_name="embedding", index_params=index_config)
    
    # Load collection into memory
    collection.load()
    
    logger.info(
        f"✅ Ingestion complete: {embeddings.shape[0]} vectors in '{collection_name}' "
        f"(dim={dim}, index={index_type}, metric={similarity_metric})"
    )
    
    return collection


def _insert_data(
    collection: Collection,
    embeddings: np.ndarray,
    chunks: List[str],
    metadatas: List[Dict]
) -> None:
    """
    Insert embeddings and metadata into collection.
    
    Parameters
    ----------
    collection : Collection
        Milvus collection
    embeddings : np.ndarray
        Embeddings array
    chunks : List[str]
        List of chunk texts
    metadatas : List[Dict]
        List of metadata dictionaries
    """
    logger.info(f"Inserting {len(metadatas)} records...")
    
    insert_data = [
        {
            "embedding": embeddings[i].tolist(),
            "source_file": metadatas[i]["source_file"],
            "chunk_id": metadatas[i].get("chunk_id", i),
            "preview": chunks[i][:500],  # First 500 chars
            "full": chunks[i],
            "page_number": metadatas[i]["page_number"],
            "chunk_type": metadatas[i]["chunk_type"],
            "total_pages": metadatas[i]["total_pages"],
            "text_length": metadatas[i]["text_length"],
            "num_tables": metadatas[i]["num_tables"],
            "num_images": metadatas[i]["num_images"],
            "image_coverage_ratio": metadatas[i]["image_coverage_ratio"],
            "is_vision_processed": metadatas[i]["is_vision_processed"],
            "table_content": json.dumps(metadatas[i]["table_content"]),
            "image_details": json.dumps(metadatas[i]["image_details"]),
            "bounding_boxes": json.dumps(metadatas[i]["bounding_boxes"]),
        }
        for i in range(len(metadatas))
    ]
    
    collection.insert(insert_data)
    logger.info("✅ Data insertion complete")
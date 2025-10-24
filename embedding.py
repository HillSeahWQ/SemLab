"""
Text embedding module.
Supports multiple embedding providers (OpenAI, Sentence Transformers). 
"""
import numpy as np
from typing import List
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from utils.logger import setup_logger
from config import LOGGING

logger = setup_logger(__name__, **LOGGING)


class EmbeddingProvider:
    """Base class for embedding providers."""
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts."""
        raise NotImplementedError


class SentenceTransformerProvider(EmbeddingProvider):
    """Sentence Transformers embedding provider."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 64,
        normalize_embeddings: bool = True
    ):
        """
        Initialize Sentence Transformer provider.
        
        Parameters
        ----------
        model_name : str
            Name of the SentenceTransformers model
        batch_size : int
            Batch size for parallel embedding
        normalize_embeddings : bool
            Whether to L2-normalize embeddings
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts.
        
        Parameters
        ----------
        texts : List[str]
            List of texts to embed
            
        Returns
        -------
        np.ndarray
            Array of shape (num_texts, embedding_dim)
        """
        logger.info(f"Embedding {len(texts)} texts with {self.model_name}")
        
        vectors = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding"):
            batch = texts[i:i + self.batch_size]
            vec = self.model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings
            )
            vectors.append(vec)
        
        embeddings = np.vstack(vectors).astype("float32")
        logger.info(f"Embedding complete. Shape: {embeddings.shape}")
        return embeddings


class OpenAIProvider(EmbeddingProvider):
    """OpenAI embedding provider."""
    
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        batch_size: int = 1000,
        normalize_embeddings: bool = True
    ):
        """
        Initialize OpenAI embedding provider.
        
        Parameters
        ----------
        model_name : str
            OpenAI embedding model name
        batch_size : int
            Number of texts per API call
        normalize_embeddings : bool
            Whether to L2-normalize embeddings
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.client = OpenAI()
        
        logger.info(f"Initialized OpenAI embedding provider: {model_name}")
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts using OpenAI API.
        
        Parameters
        ----------
        texts : List[str]
            List of texts to embed
            
        Returns
        -------
        np.ndarray
            Array of shape (num_texts, embedding_dim)
        """
        logger.info(f"Embedding {len(texts)} texts with {self.model_name}")
        
        vectors = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding"):
            batch = texts[i:i + self.batch_size]
            
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch
            )
            
            batch_vectors = [d.embedding for d in response.data]
            batch_vectors = np.array(batch_vectors, dtype=np.float32)
            
            if self.normalize_embeddings:
                norms = np.linalg.norm(batch_vectors, axis=1, keepdims=True)
                batch_vectors = batch_vectors / np.maximum(norms, 1e-12)
            
            vectors.append(batch_vectors)
        
        embeddings = np.vstack(vectors)
        logger.info(f"Embedding complete. Shape: {embeddings.shape}")
        return embeddings


def get_embedding_provider(
    provider: str,
    model: str,
    batch_size: int = 64,
    normalize_embeddings: bool = True
) -> EmbeddingProvider:
    """
    Factory function to get the appropriate embedding provider.
    
    Parameters
    ----------
    provider : str
        "openai" or "sentence_transformers"
    model : str
        Model name for the provider
    batch_size : int
        Batch size for embedding
    normalize_embeddings : bool
        Whether to normalize embeddings
        
    Returns
    -------
    EmbeddingProvider
        Configured embedding provider
    """
    if provider.lower() == "openai":
        return OpenAIProvider(model, batch_size, normalize_embeddings)
    elif provider.lower() == "sentence_transformers":
        return SentenceTransformerProvider(model, batch_size, normalize_embeddings)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
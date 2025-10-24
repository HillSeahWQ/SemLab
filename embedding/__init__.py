"""
Embedding module for generating vector embeddings.
"""
from .embedding_manager import (
    BaseEmbedder,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
    EmbeddingManager
)

__all__ = [
    'BaseEmbedder',
    'OpenAIEmbedder',
    'SentenceTransformerEmbedder',
    'EmbeddingManager'
]
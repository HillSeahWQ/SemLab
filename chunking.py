"""
Document chunking module.
Handles PDF chunking with multimodal support.
"""
import json
from pathlib import Path
from typing import List, Dict
from pdf_chunker import MultimodalPDFChunker
from utils.logger import setup_logger
from config import LOGGING

logger = setup_logger(__name__, **LOGGING)


def chunk_pdfs(
    input_dir: Path,
    image_coverage_threshold: float = 0.15,
    vision_model: str = "gpt-4o"
) -> List[Dict]:
    """
    Chunk all PDFs in the input directory.
    
    Parameters
    ----------
    input_dir : Path
        Directory containing PDF files
    image_coverage_threshold : float
        Threshold for triggering vision processing
    vision_model : str
        Vision model to use for image-heavy pages
        
    Returns
    -------
    List[Dict]
        List of chunk dictionaries with content and metadata
    """
    logger.info(f"Starting PDF chunking from: {input_dir}")
    
    # Initialize chunker
    chunker = MultimodalPDFChunker(
        image_coverage_threshold=image_coverage_threshold,
        vision_model=vision_model
    )
    
    chunks = []
    pdf_count = 0
    
    # Scan directory for PDFs
    logger.info(f"Scanning directory: {input_dir}")
    for path in input_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".pdf":
            pdf_count += 1
            logger.info(f"Chunking file [{pdf_count}]: {path.name}")
            try:
                file_chunks = chunker.chunk_pdf(path)
                chunks.extend(file_chunks)
                logger.info(f"  → Generated {len(file_chunks)} chunks")
            except Exception as e:
                logger.error(f"  → Failed to chunk {path.name}: {e}")
    
    logger.info(f"Chunking complete: {len(chunks)} total chunks from {pdf_count} PDFs")
    return chunks


def save_chunks(chunks: List, output_path: Path) -> None:
    """
    Save chunks to JSON file.
    
    Parameters
    ----------
    chunks : List
        List of Chunk objects or dictionaries
    output_path : Path
        Path to save JSON file
    """
    logger.info(f"Saving chunks to: {output_path}")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dictionaries if needed
    chunk_dicts = [
        chunk.to_dict() if hasattr(chunk, 'to_dict') else chunk 
        for chunk in chunks
    ]
    
    # Save to JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunk_dicts, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Saved {len(chunk_dicts)} chunks to {output_path}")


def load_chunks(input_path: Path) -> List[Dict]:
    """
    Load chunks from JSON file.
    
    Parameters
    ----------
    input_path : Path
        Path to JSON file containing chunks
        
    Returns
    -------
    List[Dict]
        List of chunk dictionaries
    """
    logger.info(f"Loading chunks from: {input_path}")
    
    if not input_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {input_path}")
    
    with open(input_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    logger.info(f"✅ Loaded {len(chunks)} chunks")
    return chunks
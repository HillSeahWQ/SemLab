"""
Script to run document chunking.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chunking import chunk_pdfs, save_chunks
from config import INPUT_DIR, CHUNKS_FILE, CHUNKING
from utils.logger import setup_logger, LOGGING

logger = setup_logger(__name__, **LOGGING)


def main():
    """Run chunking pipeline."""
    logger.info("Starting chunking pipeline...")
    
    # Check if input directory exists
    if not INPUT_DIR.exists():
        logger.error(f"Input directory not found: {INPUT_DIR}")
        return
    
    # Chunk PDFs
    chunks = chunk_pdfs(
        input_dir=INPUT_DIR,
        image_coverage_threshold=CHUNKING["image_coverage_threshold"],
        vision_model=CHUNKING["vision_model"]
    )
    
    if not chunks:
        logger.warning("No chunks generated!")
        return
    
    # Save chunks
    save_chunks(chunks, CHUNKS_FILE)
    
    logger.info(f"✅ Pipeline complete! Generated {len(chunks)} chunks")


if __name__ == "__main__":
    main()
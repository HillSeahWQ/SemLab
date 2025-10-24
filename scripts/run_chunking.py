"""
Chunking script - processes documents and saves chunks to JSON.
Run this script to chunk documents without immediately ingesting them.
This allows you to reuse chunks across different vector databases.
"""
import logging
import logging.config
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    CHUNKING_CONFIG,
    INPUT_DIR,
    get_chunk_output_path,
    LOGGING_CONFIG,
    EXPERIMENT_CONFIG
)
from chunking.pdf_chunker import MultimodalPDFChunker
from utils.storage import save_chunks, print_chunk_statistics
from utils.logger import get_logger

logger = get_logger(__name__)

def main(
    input_dir: Path = None,
    output_path: Path = None,
    extensions: list = None
):
    """
    Run document chunking pipeline.
    
    Args:
        input_dir: Directory containing documents (uses config default if None)
        output_path: Path to save chunks JSON (uses config default if None)
        extensions: List of file extensions to process (default: [".pdf"])
    """
    load_dotenv()
    
    logger.info("="*80)
    logger.info("DOCUMENT CHUNKING PIPELINE")
    logger.info("="*80)
    logger.info(f"Experiment: {EXPERIMENT_CONFIG['name']}")
    logger.info(f"Description: {EXPERIMENT_CONFIG['description']}")
    logger.info(f"Version: {EXPERIMENT_CONFIG['version']}")
    logger.info("")
    
    # Use defaults from config if not provided
    if input_dir is None:
        input_dir = INPUT_DIR
    if output_path is None:
        output_path = get_chunk_output_path()
    if extensions is None:
        extensions = [".pdf"]
    
    try:
        logger.info("STEP 1: INITIALIZING CHUNKER")
        logger.info("-"*80)
        
        # Initialize chunker based on document type
        # For now, we assume PDF, but this can be extended
        pdf_config = CHUNKING_CONFIG.get("pdf", {})
        chunker = MultimodalPDFChunker(
            image_coverage_threshold=pdf_config.get("image_coverage_threshold", 0.3),
            vision_model=pdf_config.get("vision_model", "gpt-4o"),
            log_level=pdf_config.get("log_level", "INFO")
        )
        
        logger.info(f"Chunker initialized: {chunker.__class__.__name__}")
        logger.info(f"   - Image threshold: {chunker.image_coverage_threshold:.1%}")
        logger.info(f"   - Vision model: {chunker.vision_model.model_name}")
        
        logger.info("")
        logger.info("STEP 2: PROCESSING DOCUMENTS")
        logger.info("-"*80)
        logger.info(f"Input directory: {input_dir}")
        logger.info(f"File extensions: {extensions}")
        
        # Chunk all documents in directory
        chunks = chunker.chunk_directory(input_dir, extensions=extensions)
        
        logger.info("")
        logger.info(f"Document processing complete: {len(chunks)} chunks created")
        
        logger.info("")
        logger.info("STEP 3: SAVING CHUNKS")
        logger.info("-"*80)
        
        # Save chunks to JSON
        save_chunks(chunks, output_path)
        logger.info(f"Chunks saved to: {output_path}")
        
        # Extract metadata for statistics
        metadatas = [chunk.metadata.to_dict() for chunk in chunks]
        
        logger.info("")
        logger.info("STEP 4: STATISTICS")
        logger.info("-"*80)
        print_chunk_statistics(metadatas)
        
        logger.info("")
        logger.info("="*80)
        logger.info("[SUCCESS] - CHUNKING PIPELINE COMPLETE")
        logger.info("="*80)
        logger.info(f"Total chunks: {len(chunks)}")
        logger.info(f"Output file: {output_path}")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Review the chunks in the JSON file")
        logger.info("  2. Run ingestion: python scripts/run_milvus_ingestion.py")
        logger.info("     (or use a different vector DB ingestion script)")
        logger.info("="*80)
        
        return chunks, output_path
        
    except Exception as e:
        logger.error(f"[ERROR] - Chunking pipeline failed: {e}")
        logger.exception("Full error traceback:")
        raise


if __name__ == "__main__":
    # Run with defaults from config
    main()
    
    # Or customize:
    # main(
    #     input_dir=Path("path/to/documents"),
    #     output_path=Path("path/to/output.json"),
    #     extensions=[".pdf", ".txt"]
    # )
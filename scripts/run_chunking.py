"""
Document chunking script with command-line interface.
Supports both fast runs (file paths only) and advanced runs (with hyperparameters).
"""
import sys
import argparse
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    INPUT_DIR,
    get_chunk_output_path,
    CHUNKING_CONFIG,
    EXPERIMENT_CONFIG,
    DATA_DIR
)
from chunking.pdf_chunker import MultimodalPDFChunker
from utils.storage import save_chunks
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Chunk documents for RAG pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Fast run - minimal arguments
        python run_chunking.py --input data/raw-documents --output data/chunks/output.json
        
        # Advanced run - with hyperparameters
        python run_chunking.py \\
            --input data/raw-documents \\
            --output data/chunks/output.json \\
            --image-threshold 0.2 \\
            --vision-model gpt-4o
        """
    )
    
    # === FAST RUN ARGUMENTS (File Paths) ===
    parser.add_argument(
        "--input",
        type=str,
        help=f"Input directory with documents (default: data/raw-documents, or from config)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help=f"Output JSON file for chunks (default: data/chunks/<experiment>_chunks.json)"
    )
    
    # === ADVANCED RUN ARGUMENTS (Hyperparameters) ===
    parser.add_argument(
        "--image-threshold",
        type=float,
        help=f"Image coverage threshold for vision processing (default: from config, currently {CHUNKING_CONFIG['pdf'].get('image_coverage_threshold', 0.15)})"
    )
    parser.add_argument(
        "--vision-model",
        type=str,
        choices=["gpt-4o", "gpt-4-vision-preview"],
        help=f"Vision model for image processing (default: from config, currently {CHUNKING_CONFIG['pdf'].get('vision_model', 'gpt-4o')})"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: from config)"
    )
    
    return parser.parse_args()


def main():
    """Run chunking pipeline."""
    args = parse_args()
    
    # === RESOLVE PATHS ===
    # Input directory (fast run)
    if args.input:
        input_dir = Path(args.input)
        # Support relative paths from data/
        if not input_dir.is_absolute():
            if not input_dir.exists():
                # Try relative to DATA_DIR
                input_dir = DATA_DIR / args.input
    else:
        input_dir = INPUT_DIR
    
    # Output file (fast run)
    if args.output:
        output_file = Path(args.output)
        # Support relative paths from data/
        if not output_file.is_absolute():
            output_file = DATA_DIR / args.output
    else:
        output_file = get_chunk_output_path()
    
    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # === BUILD CONFIG ===
    # Start with config file defaults
    chunking_config = CHUNKING_CONFIG["pdf"].copy()
    
    # Override with command-line arguments (advanced run)
    if args.image_threshold is not None:
        chunking_config["image_coverage_threshold"] = args.image_threshold
    if args.vision_model is not None:
        chunking_config["vision_model"] = args.vision_model
    if args.log_level is not None:
        chunking_config["log_level"] = args.log_level
    
    # === LOG CONFIGURATION ===
    logger.info("="*80)
    logger.info("DOCUMENT CHUNKING PIPELINE")
    logger.info("="*80)
    logger.info(f"Experiment: {EXPERIMENT_CONFIG['name']}")
    logger.info(f"Description: {EXPERIMENT_CONFIG['description']}")
    logger.info("")
    logger.info("Configuration:")
    logger.info(f"  Input directory: {input_dir}")
    logger.info(f"  Output file: {output_file}")
    logger.info(f"  Image threshold: {chunking_config['image_coverage_threshold']}")
    logger.info(f"  Vision model: {chunking_config['vision_model']}")
    logger.info("")
    
    # === VALIDATION ===
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return 1
    
    # === RUN CHUNKING ===
    try:
        logger.info("Initializing PDF chunker...")
        chunker = MultimodalPDFChunker(**chunking_config)
        
        # Get all PDF files
        pdf_files = list(input_dir.glob("**/*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {input_dir}")
            return 1
        
        logger.info(f"Found {len(pdf_files)} PDF files")
        logger.info("")
        
        # Process all files
        all_contents = []
        all_metadatas = []
        
        for pdf_file in pdf_files:
            logger.info(f"Processing: {pdf_file.name}")
            try:
                contents, metadatas = chunker.chunk(pdf_file)
                all_contents.extend(contents)
                all_metadatas.extend(metadatas)
                logger.info(f"Generated {len(contents)} chunks")
            except Exception as e:
                logger.error(f"Failed: {e}")
                continue
        
        # Save chunks
        logger.info("")
        logger.info(f"Saving {len(all_contents)} chunks to {output_file}")
        save_chunks(all_contents, all_metadatas, output_file)
        
        # Summary
        logger.info("")
        logger.info("="*80)
        logger.info("[SUCCESS] - CHUNKING COMPLETE")
        logger.info("="*80)
        logger.info(f"Total chunks: {len(all_contents)}")
        logger.info(f"Output saved to: {output_file}")
        logger.info("")
        logger.info("="*80)
        
        return 0
        
    except Exception as e:
        logger.error(f"[ERROR] - Chunking failed: {e}")
        logger.exception("Full error traceback:")
        return 1


if __name__ == "__main__":
    load_dotenv()
    sys.exit(main())
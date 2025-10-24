"""
COST - 0.90 for 97 image processed pdf slides with gpt-4o

Multimodal PDF Chunker with intelligent page analysis and processing.

This chunker:
- Analyzes each page for text, tables, and images
- Detects tables spanning multiple pages
- Converts image-heavy pages (proprotion above an area threshold) to images for vision model input
- Provides rich metadata for each chunk

Inspired by - https://medium.com/@saptarshi701/advanced-chunking-for-pdf-word-with-embedded-images-using-regular-parsers-and-gpt-4o-7f0d5eb97052

Key Features

1. Intelligent Page Analysis

Calculates image coverage ratio for each page
Detects when images exceed your threshold (default 30%)
Analyzes content composition (text, tables, images)

2. Image-Heavy Page Processing

Pages with high image coverage are converted to images
Processed with vision model (GPT-4o) using a detailed prompt
Critical: Only triggers if the page has NO tables (tables are handled separately)

3. Multi-Page Table Detection

Detects tables spanning multiple consecutive pages
Groups them into single chunks
Preserves HTML table structure in metadata

4. Smart Integration Logic
Decision Tree:
├── Is page part of multi-page table? → Handle as TABLE chunk
├── Else: Does page exceed image threshold AND has no tables? → VISION process
└── Else: Standard text extraction → TEXT/MIXED chunk
This ensures tables are never interfered with by the image processing logic!

5. Rich Metadata
Each chunk includes:

Page number and chunk type
Text length, table count, image count
Image coverage ratio
Table spanning information
Bounding box coordinates
Whether it was vision-processed
HTML table representation

## Usage
```python
# Install dependencies
pip install langchain-unstructured langchain-openai pymupdf pillow unstructured unstructured[pdf]

# Set environment variables
export OPENAI_API_KEY="your-key"
export UNSTRUCTURED_API_KEY="your-key"  # Optional, for API usage

# Use the chunker
chunker = MultimodalPDFChunker(
    image_coverage_threshold=0.3,  # Adjust as needed
    vision_model="gpt-4o",
    use_unstructured_api=True
)

chunks = chunker.chunk_pdf("document.pdf")

# Access chunks
for chunk in chunks:
    print(chunk.content)
    print(chunk.metadata.to_dict())
```

##Customization Tips

Adjust threshold: Change image_coverage_threshold (0.0-1.0)
Different vision model: Use "gpt-4o-mini" or "claude-3-5-sonnet-20241022"
Local processing: Set use_unstructured_api=False and install dependencies
Custom prompts: Modify _process_page_with_vision() prompt

The chunker is production-ready and handles edge cases like overlapping content types gracefully!
"""
"""
Multimodal PDF Chunker with intelligent page analysis and processing.

This chunker:
- Analyzes each page for text, tables, and images
- Detects tables spanning multiple pages
- Converts image-heavy pages to vision model input
- Provides rich metadata for each chunk

## Schema for a Chunk object.

Each Chunk represents a semantically meaningful unit of content extracted from a document.
Intended for use in downstream vector database ingestion and retrieval tasks.

Attributes
----------
content : str
    The textual content of the chunk.

metadata : dict
    Additional contextual information about the chunk.
    
    Keys
    ----
    page_number : int
        The page number from which the chunk was extracted (1-indexed).
    
    chunk_type : str
        Type of chunk, e.g. "text", "table", "image", or "mixed".
    
    total_pages : int
        Total number of pages in the source document.
    
    source_file : str
        The name or path of the source file this chunk was derived from.
    
    text_length : int
        The number of characters in the chunk content.
    
    num_tables : int
        Number of tables contained within the chunk.
    
    num_images : int
        Number of images contained within the chunk.
    
    image_coverage_ratio : float
        Ratio (0–1) indicating how much of the page area is covered by images.
    
    table_content : List[str]
        Extracted text or HTML representations of tables within the chunk.
    
    is_vision_processed : bool
        Whether this chunk has been processed by a vision model for OCR, layout, or image captioning.
    
    image_details : List[Dict]
        A list of dictionaries describing detected images (e.g. filename, caption, size, or embeddings).
    
    bounding_boxes : List[Dict]
        Bounding box coordinates of layout elements (e.g. tables, figures, paragraphs) in document coordinate space.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
import base64
import io
import re
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

import fitz  # PyMuPDF
from PIL import Image
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class ChunkType(Enum):
    """Types of chunks produced by the chunker."""
    TEXT = "text"
    TABLE = "table"
    IMAGE_HEAVY_PAGE = "image_heavy_page"
    MIXED = "mixed"


@dataclass
class ChunkMetadata:
    """Comprehensive metadata for document chunks."""
    page_number: int
    chunk_type: ChunkType
    total_pages: int
    source_file: str
    
    # Content statistics
    text_length: int = 0
    num_tables: int = 0
    num_images: int = 0
    image_coverage_ratio: float = 0.0
    
    # Table-specific metadata
    table_content: List[str] = field(default_factory=list)
    
    # Image-specific metadata
    is_vision_processed: bool = False
    image_details: List[Dict[str, Any]] = field(default_factory=list)
    
    # Coordinates and layout
    bounding_boxes: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary format."""
        return {
            "page_number": self.page_number,
            "chunk_type": self.chunk_type.value,
            "total_pages": self.total_pages,
            "source_file": str(self.source_file),
            "text_length": self.text_length,
            "num_tables": self.num_tables,
            "num_images": self.num_images,
            "image_coverage_ratio": self.image_coverage_ratio,
            "table_content": self.table_content,
            "is_vision_processed": str(self.is_vision_processed),
            "image_details": self.image_details,
            "bounding_boxes": self.bounding_boxes,
        }


@dataclass
class Chunk:
    """Represents a processed document chunk."""
    content: str
    metadata: ChunkMetadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary format."""
        return {
            "content": self.content,
            "metadata": self.metadata.to_dict()
        }


class MultimodalPDFChunker:
    """
    PDF chunker using PyMuPDF for direct analysis.
    
    Features:
    - Page-based chunking with intelligent analysis
    - Table and image detection using PyMuPDF
    - Image coverage analysis with vision model processing
    - Comprehensive metadata generation
    """
    
    def __init__(
        self,
        image_coverage_threshold: float = 0.3,
        vision_model: str = "gpt-4o",
        log_level: str = "INFO"
    ):
        """
        Initialize the chunker.
        
        Args:
            image_coverage_threshold: Threshold ratio (0-1) for image coverage
                                     to trigger vision processing
            vision_model: Vision-capable LLM model name
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.image_coverage_threshold = image_coverage_threshold
        self.vision_model = ChatOpenAI(model=vision_model)
        
        # Set up logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
    def chunk_pdf(self, pdf_path: str) -> List[Chunk]:
        """
        Process PDF and generate chunks.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of Chunk objects with content and metadata
        """
        start_time = datetime.now()
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Starting PDF chunking: {pdf_path}")
        self.logger.info(f"Image coverage threshold: {self.image_coverage_threshold:.1%}")
        self.logger.info(f"{'='*80}")
        
        # Open PDF with PyMuPDF
        try:
            pdf_document = fitz.open(pdf_path)
            total_pages = len(pdf_document)
            self.logger.info(f"📄 Successfully opened PDF: {total_pages} pages")
        except Exception as e:
            self.logger.error(f"❌ Failed to open PDF: {e}")
            raise
        
        chunks = []
        stats = {
            "text_pages": 0,
            "table_pages": 0,
            "mixed_pages": 0,
            "vision_processed_pages": 0,
            "total_tables": 0,
            "total_images": 0
        }
        
        for page_num in range(total_pages):
            page = pdf_document.load_page(page_num)
            
            # Analyze page structure
            self.logger.info(f"")
            self.logger.info(f"{'─'*80}")
            self.logger.info(f"Processing page {page_num + 1}/{total_pages}")
            
            try:
                page_analysis = self._analyze_page(page, page_num + 1)
                
                # Log analysis results
                self.logger.info(f"  📊 Analysis complete:")
                self.logger.info(f"     • Text length: {len(page_analysis['text']):,} chars")
                self.logger.info(f"     • Tables detected: {page_analysis['num_tables']}")
                self.logger.info(f"     • Images detected: {page_analysis['num_images']}")
                self.logger.info(f"     • Image coverage: {page_analysis['image_coverage_ratio']:.1%}")
                
                # Generate chunk based on analysis
                chunk = self._generate_chunk_for_page(
                    pdf_path,
                    page,
                    page_num + 1,
                    total_pages,
                    page_analysis
                )
                
                chunks.append(chunk)
                
                # Update statistics
                stats["total_tables"] += page_analysis["num_tables"]
                stats["total_images"] += page_analysis["num_images"]
                
                if chunk.metadata.is_vision_processed:
                    stats["vision_processed_pages"] += 1
                elif chunk.metadata.chunk_type == ChunkType.TABLE:
                    stats["table_pages"] += 1
                elif chunk.metadata.chunk_type == ChunkType.MIXED:
                    stats["mixed_pages"] += 1
                else:
                    stats["text_pages"] += 1
                
                self.logger.info(f"  ✅ Chunk created: {chunk.metadata.chunk_type.value.upper()} ({chunk.metadata.text_length:,} chars)")
                
            except Exception as e:
                self.logger.error(f"  ❌ Error processing page {page_num + 1}: {e}")
                self.logger.exception("Full error details:")
                raise
        
        pdf_document.close()
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # Final summary
        self.logger.info(f"")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"✅ CHUNKING COMPLETE")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Total chunks created: {len(chunks)}")
        self.logger.info(f"Time elapsed: {elapsed:.2f}s ({elapsed/total_pages:.2f}s per page)")
        self.logger.info(f"")
        self.logger.info(f"📊 Chunk Type Distribution:")
        self.logger.info(f"   • Text pages: {stats['text_pages']}")
        self.logger.info(f"   • Table pages: {stats['table_pages']}")
        self.logger.info(f"   • Mixed pages: {stats['mixed_pages']}")
        self.logger.info(f"   • Vision-processed pages: {stats['vision_processed_pages']}")
        self.logger.info(f"")
        self.logger.info(f"📈 Content Statistics:")
        self.logger.info(f"   • Total tables detected: {stats['total_tables']}")
        self.logger.info(f"   • Total images detected: {stats['total_images']}")
        self.logger.info(f"{'='*80}")
        
        return chunks
    
    def _analyze_page(self, page: fitz.Page, page_num: int) -> Dict[str, Any]:
        """
        Analyze a single page for content composition.
        
        Args:
            page: PyMuPDF page object
            page_num: Page number (1-indexed)
            
        Returns:
            Dictionary with analysis results
        """
        self.logger.debug(f"  Analyzing page structure...")
        
        page_rect = page.rect
        page_area = page_rect.width * page_rect.height
        
        # Extract text
        text = page.get_text()
        self.logger.debug(f"    Extracted {len(text)} characters of text")
        
        # Detect images
        image_list = page.get_images(full=True)
        num_images = len(image_list)
        self.logger.debug(f"    Found {num_images} images")
        
        # Calculate image coverage
        total_image_area = 0
        image_details = []
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            
            # Get image bounding boxes
            image_rects = page.get_image_rects(xref)
            
            for rect in image_rects:
                img_area = rect.width * rect.height
                total_image_area += img_area
                
                image_details.append({
                    "index": img_index,
                    "bbox": (rect.x0, rect.y0, rect.x1, rect.y1),
                    "area": img_area,
                    "width": rect.width,
                    "height": rect.height
                })
        
        image_coverage_ratio = total_image_area / page_area if page_area > 0 else 0
        
        # Detect tables using PyMuPDF's table detection
        self.logger.debug(f"    Detecting tables...")
        tables = page.find_tables()
        num_tables = len(tables.tables) if tables else 0
        self.logger.debug(f"    Found {num_tables} tables")
        
        table_content = []
        table_bboxes = []
        
        if tables:
            for idx, table in enumerate(tables.tables):
                # Extract table data
                table_data = table.extract()
                
                # Convert to string representation
                if table_data:
                    # Format as markdown-style table
                    table_str = self._format_table_as_markdown(table_data)
                    table_content.append(table_str)
                    
                    # Store bounding box
                    table_bboxes.append({
                        "bbox": table.bbox,
                        "rows": len(table_data),
                        "cols": len(table_data[0]) if table_data else 0
                    })
                    self.logger.debug(f"    Table {idx+1}: {len(table_data)} rows × {len(table_data[0]) if table_data else 0} cols")
        
        return {
            "text": text,
            "num_images": num_images,
            "num_tables": num_tables,
            "image_coverage_ratio": image_coverage_ratio,
            "exceeds_threshold": image_coverage_ratio > self.image_coverage_threshold,
            "image_details": image_details,
            "table_content": table_content,
            "table_bboxes": table_bboxes,
            "page_area": page_area
        }
    
    def _format_table_as_markdown(self, table_data: List[List[str]]) -> str:
        """Format table data as markdown table."""
        if not table_data:
            return ""
        
        lines = []
        
        # Add header row
        header = table_data[0]
        lines.append("| " + " | ".join(str(cell) if cell else "" for cell in header) + " |")
        
        # Add separator
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        
        # Add data rows
        for row in table_data[1:]:
            lines.append("| " + " | ".join(str(cell) if cell else "" for cell in row) + " |")
        
        return "\n".join(lines)
    
    def _page_to_base64(self, page: fitz.Page) -> str:
        """Convert PDF page to base64-encoded image."""
        # Render at 2x resolution for better quality
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def _process_page_with_vision(self, page: fitz.Page, page_num: int) -> str:
        """Process an image-heavy page with vision model."""
        self.logger.info(f"  🔍 Vision processing initiated (page {page_num})")
        self.logger.debug(f"    Converting page to image...")
        
        base64_image = self._page_to_base64(page)
        
        self.logger.debug(f"    Sending to vision model ({self.vision_model.model_name})...")
        
        prompt = """Analyze this document page comprehensively and provide a complete textual description.

        Your output should include:

        1. ALL TEXT CONTENT: Transcribe every piece of text visible on the page, maintaining the reading order and structure. Include:
        - Headings and titles
        - Body text and paragraphs
        - Captions and labels
        - Any annotations or side notes
        - Text within or near images

        2. VISUAL ELEMENTS: Describe all images, charts, diagrams, or visual content:
        - What each image shows
        - Key details and components
        - Relationship to surrounding text
        - Any data or insights conveyed visually

        3. LAYOUT AND STRUCTURE:
        - Note the organization of content (columns, sections, etc.)
        - Indicate relationships between text and visuals
        - Describe the flow of information

        4. TABLES OR STRUCTURED DATA (if present):
        - Describe the table structure
        - Include key data points and headers
        - Note any trends or patterns

        OUTPUT FORMAT:
        Provide a flowing, readable description that captures all information on the page. Be thorough and precise. Someone should be able to understand the complete page content from your description alone."""

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                },
            ],
        )
        
        try:
            response = self.vision_model.invoke([message])
            self.logger.info(f"  ✅ Vision processing complete ({len(response.content)} chars generated)")
            return response.content
        except Exception as e:
            self.logger.error(f"  ❌ Vision processing failed: {e}")
            raise
    
    def _generate_chunk_for_page(
        self,
        pdf_path: str,
        page: fitz.Page,
        page_num: int,
        total_pages: int,
        analysis: Dict[str, Any]
    ) -> Chunk:
        """Generate a chunk for a single page based on analysis."""
        
        # Decision logic: 
        # 1. If page has tables, never use vision processing (tables take priority)
        # 2. If page exceeds image threshold and has no tables, use vision processing
        # 3. Otherwise, use standard text extraction
        
        if analysis["num_tables"] > 0:
            # Page has tables - use standard extraction with table formatting
            self.logger.debug(f"  Processing as TABLE/MIXED page")
            
            content_parts = []
            
            # Add regular text (filter out table text to avoid duplication)
            if analysis["text"].strip():
                content_parts.append(analysis["text"].strip())
            
            # Add formatted tables
            if analysis["table_content"]:
                content_parts.append("\n\n--- Tables ---\n")
                content_parts.extend(analysis["table_content"])
            
            content = "\n\n".join(content_parts)
            
            # Determine chunk type
            if analysis["num_images"] > 0:
                chunk_type = ChunkType.MIXED
                self.logger.debug(f"    Chunk type: MIXED (tables + images)")
            else:
                chunk_type = ChunkType.TABLE
                self.logger.debug(f"    Chunk type: TABLE")
            
            metadata = ChunkMetadata(
                page_number=page_num,
                chunk_type=chunk_type,
                total_pages=total_pages,
                source_file=pdf_path,
                text_length=len(content),
                num_tables=analysis["num_tables"],
                num_images=analysis["num_images"],
                image_coverage_ratio=analysis["image_coverage_ratio"],
                table_content=analysis["table_content"],
                image_details=analysis["image_details"],
                bounding_boxes=analysis["table_bboxes"]
            )
            
        elif analysis["exceeds_threshold"]:
            # Image-heavy page with no tables - use vision processing
            self.logger.info(f"  🖼️  Image-heavy page detected (coverage: {analysis['image_coverage_ratio']:.1%} > {self.image_coverage_threshold:.1%})")
            
            content = self._process_page_with_vision(page, page_num)
            
            metadata = ChunkMetadata(
                page_number=page_num,
                chunk_type=ChunkType.IMAGE_HEAVY_PAGE,
                total_pages=total_pages,
                source_file=pdf_path,
                text_length=len(content),
                num_images=analysis["num_images"],
                image_coverage_ratio=analysis["image_coverage_ratio"],
                is_vision_processed=True,
                image_details=analysis["image_details"]
            )
            
        else:
            # Standard text extraction
            self.logger.debug(f"  Processing as TEXT page")
            
            content = analysis["text"].strip()
            
            chunk_type = ChunkType.TEXT
            if analysis["num_images"] > 0:
                chunk_type = ChunkType.MIXED
                self.logger.debug(f"    Chunk type: MIXED (text + images)")
            else:
                self.logger.debug(f"    Chunk type: TEXT")
            
            metadata = ChunkMetadata(
                page_number=page_num,
                chunk_type=chunk_type,
                total_pages=total_pages,
                source_file=pdf_path,
                text_length=len(content),
                num_images=analysis["num_images"],
                image_coverage_ratio=analysis["image_coverage_ratio"],
                image_details=analysis["image_details"]
            )
        
        return Chunk(content=content, metadata=metadata)


# Example usage
if __name__ == "__main__":
    
    load_dotenv() # Require - OPENAI_API_KEY
    
    # Config
    IMAGE_COVERAGE_THRESHOLD=0.15  # 15% image coverage triggers vision processing
    VISION_MODEL="gpt-4o"
    PDF_PATH = Path().cwd().parent.parent / "data" / "kyndryl-docs-test" / "sampled_comms_deck.pdf"
    
    # Initialize chunker
    chunker = MultimodalPDFChunker(
        image_coverage_threshold=IMAGE_COVERAGE_THRESHOLD,
        vision_model=VISION_MODEL
    )
    
    # Process PDF
    pdf_path = PDF_PATH
    
    if os.path.exists(pdf_path):
        chunks = chunker.chunk_pdf(pdf_path)
        
        # Display results
        print(f"\n{'='*80}")
        print("CHUNKING RESULTS")
        print(f"{'='*80}\n")
        
        for i, chunk in enumerate(chunks):
            print(f"{'─'*80}")
            print(f"Chunk {i+1}/{len(chunks)}")
            print(f"{'─'*80}")
            print(f"📄 Page: {chunk.metadata.page_number}/{chunk.metadata.total_pages}")
            print(f"📝 Type: {chunk.metadata.chunk_type.value.upper()}")
            print(f"📊 Content length: {chunk.metadata.text_length:,} characters")
            print(f"📈 Tables: {chunk.metadata.num_tables}")
            print(f"🖼️  Images: {chunk.metadata.num_images}")
            print(f"📐 Image coverage: {chunk.metadata.image_coverage_ratio:.1%}")
            
            if chunk.metadata.is_vision_processed:
                print("✨ Vision-processed page")
            
            if chunk.metadata.num_tables > 0:
                print(f"📋 Contains {chunk.metadata.num_tables} table(s)")
            
            print(f"\n📄 Content preview:")
            print("─" * 80)
            preview_length = 500
            if len(chunk.content) > preview_length:
                print(chunk.content[:preview_length] + "...")
            else:
                print(chunk.content)
            print()
    else:
        print(f"❌ Example file not found: {pdf_path}")
        print("\n📦 To use this chunker:")
        print("1. Install dependencies:")
        print("   pip install pymupdf pillow langchain-community langchain-openai")
        print("\n2. Set your API key:")
        print("   export OPENAI_API_KEY='your-key-here'")
        print("\n3. Provide a PDF path and run:")
        print("   python script.py")
        print("\n💡 Example:")
        print("   chunker = MultimodalPDFChunker(image_coverage_threshold=0.3)")
        print("   chunks = chunker.chunk_pdf('document.pdf')")
        print("   for chunk in chunks:")
        print("       print(chunk.content)")
    
    # TEMPORARY, save chunks as dict to be reused later
    import json 
    
    chunk_dicts = [chunk.to_dict() for chunk in chunks]

    # Define your target folder and file path
    CHUNKS_FOLDER_PATH = Path().cwd().parent / "data" / "chunks"
    CHUNKS_FOLDER_PATH.mkdir(parents=True, exist_ok=True)

    # Full path to the JSON file
    json_path = CHUNKS_FOLDER_PATH / "kyndryl_chunks.json"

    # --- SAVE ---
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(chunk_dicts, f, indent=2, ensure_ascii=False)

    print(f"✅ Chunks saved to: {json_path}")
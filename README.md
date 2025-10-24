# RAG Pipeline with Milvus

A modular, reusable RAG (Retrieval-Augmented Generation) pipeline for document processing, embedding, and vector search using Milvus.

## Project Structure

```
project/
├── config.py                 # Configuration file for all parameters
├── requirements.txt          # Python dependencies
├── README.md                # This file
│
├── utils/
│   └── logger.py            # Logging utilities
│
├── chunking.py              # Document chunking module
├── embedding.py             # Text embedding module (OpenAI, Sentence Transformers)
│
├── vector_db/
│   ├── __init__.py
│   ├── milvus_connection.py # Milvus connection management
│   ├── milvus_ingestion.py  # Milvus data ingestion
│   └── milvus_query.py      # Milvus vector search
│
├── scripts/
│   ├── run_chunking.py      # Run chunking pipeline
│   ├── run_ingestion.py     # Run embedding + ingestion
│   └── run_query.py         # Run queries
│
└── data/
    ├── kyndryl-docs-test/   # Input PDFs (your directory)
    └── chunks/              # Generated chunks (JSON)
```

## Setup

### 1. Install Dependencies

```bash
uv sync
```

### 2. Set Environment Variables

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Start Milvus

Ensure Milvus is running (via Docker):

```bash
# Example using docker-compose
docker-compose up -d
```

### 4. Configure Pipeline

Edit `config.py` to customize:
- **Paths**: Input/output directories
- **Chunking**: Image threshold, vision model
- **Embedding**: Provider (OpenAI/Sentence Transformers), model, batch size
- **Milvus**: Host, port, collection name, index type, similarity metric
- **Search**: Top-k results, output fields

## Usage

### Step 1: Chunk Documents

```bash
python scripts/run_chunking.py
```

This will:
- Process all PDFs in `INPUT_DIR`
- Generate chunks with metadata
- Save to `data/chunks/kyndryl_chunks.json`

### Step 2: Embed and Ingest

```bash
python scripts/run_ingestion.py
```

This will:
- Load chunks from JSON
- Generate embeddings using configured provider
- Create/update Milvus collection
- Build vector index

### Step 3: Query

```bash
python scripts/run_query.py
```

Edit queries in `run_query.py` or create your own query script.

## Configuration Examples

### Use Sentence Transformers Instead of OpenAI

In `config.py`:

```python
EMBEDDING = {
    "provider": "sentence_transformers",
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "batch_size": 64,
    "normalize_embeddings": True
}
```

### Use HNSW Index Instead of IVF_FLAT

In `config.py`:

```python
MILVUS = {
    # ... other settings
    "index_type": "HNSW",
    "index_params": {
        "M": 16,
        "efConstruction": 200
    },
    "search_params": {
        "ef": 200
    }
}
```

### Change Similarity Metric

In `config.py`:

```python
MILVUS = {
    # ... other settings
    "similarity_metric": "COSINE",  # Options: "IP", "L2", "COSINE"
}
```

## Extensibility

### Adding New Embedding Providers

1. Create a new class in `embedding.py` inheriting from `EmbeddingProvider`
2. Implement the `embed()` method
3. Add to `get_embedding_provider()` factory function

```python
class CustomProvider(EmbeddingProvider):
    def embed(self, texts: List[str]) -> np.ndarray:
        # Your implementation
        pass
```

### Adding New Vector Databases

1. Create new modules: `vector_db/newdb_connection.py`, `vector_db/newdb_ingestion.py`, `vector_db/newdb_query.py`
2. Follow the same interface patterns as Milvus modules
3. Create corresponding run scripts in `scripts/`

### Adding Hybrid Search

The current architecture supports extension to hybrid search:

1. Add BM25/keyword search module alongside vector search
2. Combine results in a new `hybrid_query.py` module
3. Configure hybrid parameters in `config.py`

## Logging

Logging is configured in `config.py`:

```python
LOGGING = {
    "level": "INFO",  # DEBUG, INFO, WARNING, ERROR
    "format": "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S"
}
```

## Notes

- **Storage**: Currently stores full chunk text in Milvus. For production, consider external storage (S3, MinIO) and store only references.
- **Costs**: OpenAI embedding costs depend on model and text volume. Monitor usage.
- **Performance**: Adjust batch sizes and index parameters based on your dataset size and query patterns.
- **Testing**: Set `drop_existing: True` in config for testing (recreates collection each time)

## Quick Start Commands

```bash
# 1. Chunk documents
python scripts/run_chunking.py

# 2. Embed and ingest
python scripts/run_ingestion.py

# 3. Query
python scripts/run_query.py
```
```

---

## Installation Instructions

### Create the project structure:

```bash
# Create directories
mkdir -p project/utils project/vector_db project/scripts project/data/kyndryl-docs-test project/data/chunks

# Navigate to project directory
cd project

# Create __init__.py files
touch vector_db/__init__.py
```

### Copy each file above into its respective location

### Install dependencies:

```bash
pip install -r requirements.txt
```

### Create .env file with your OpenAI API key

### Run the pipeline:

```bash
# Step 1: Chunk
python scripts/run_chunking.py

# Step 2: Embed & Ingest
python scripts/run_ingestion.py

# Step 3: Query
python scripts/run_query.py
```

---

That's the complete project! All files are now visible in this single document. You can copy each section into its respective file.
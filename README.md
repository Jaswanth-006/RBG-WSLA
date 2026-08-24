# WSLA Document Parsing API

FastAPI-based REST API for parsing PDF documents using [Docling](https://github.com/DS4SD/docling). Automatically detects whether PDFs are editable or scanned and extracts structured content including tables, images, paragraphs, and section headers.

## Features

- **PDF Type Detection**: Automatically classifies PDFs as editable, scanned, or mixed
- **GPU Acceleration**: Supports CUDA/GPU for faster processing (falls back to CPU)
- **Structured Extraction**: Extracts tables (CSV/HTML/Markdown), images, text, and metadata
- **Markdown Export**: Converts PDFs to clean Markdown format
- **All-in-One Processing**: Single endpoint for upload, detect, parse, and download as ZIP
- **Performance Metrics**: Detailed timing and throughput statistics

## Quick Start with Docker

### Prerequisites

- Docker and Docker Compose installed
- (Optional) NVIDIA GPU with Docker GPU support for acceleration

### GPU Deployment

```bash
# Build and start the service with GPU support
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop the service
docker-compose down
```

### CPU-Only Deployment

```bash
# Build and start the service (CPU-only)
docker-compose -f docker-compose.cpu.yml up -d --build

# View logs
docker-compose -f docker-compose.cpu.yml logs -f

# Stop the service
docker-compose -f docker-compose.cpu.yml down
```

### Verify Service is Running

```bash
# Check health
curl http://localhost:8000/

# API documentation
open http://localhost:8000/docs
```

## API Endpoints

### 1. Health Check
```bash
GET /
```

### 2. Upload PDF
```bash
POST /upload
Content-Type: multipart/form-data

curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf"
```

**Response:**
```json
{
  "doc_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "document.pdf",
  "message": "Upload successful"
}
```

### 3. Parse PDF
```bash
GET /parse/{doc_id}

curl "http://localhost:8000/parse/{doc_id}"
```

**Response:**
```json
{
  "doc_id": "...",
  "filename": "document.pdf",
  "pdf_type": "scanned",
  "page_count": 14,
  "page_classifications": [...],
  "markdown_file": "/app/wsla_output/.../document.md",
  "markdown_text": "# Document Content...",
  "metadata": {
    "tables": [...],
    "images": [...],
    "paragraphs": [...],
    "sections": [...]
  },
  "metrics": {
    "requested_device": "auto",
    "resolved_device": "cuda",
    "gpu_available": true,
    ...
  }
}
```

### 4. Check Status
```bash
GET /status/{doc_id}

curl "http://localhost:8000/status/{doc_id}"
```

### 5. Process All-in-One (Upload + Detect + Parse + Download)
```bash
POST /process
Content-Type: multipart/form-data

curl -X POST "http://localhost:8000/process" \
  -F "file=@document.pdf" \
  -o output.zip
```

Returns a ZIP archive containing:
- `document.md` - Markdown export
- `document-metadata.json` - Structured metadata
- `document-picture-*.png` - Extracted images
- `performance.json` - Processing metrics

## Configuration

### Environment Variables

Create a `.env` file (see `.env.example`):

```bash
# Device selection: auto | gpu | cpu
WSLA_DEVICE=auto

# File size limit (bytes)
WSLA_MAX_FILE_SIZE=104857600

# Character threshold for editable vs scanned classification
WSLA_EDITABLE_CHAR_THRESHOLD=50

# Image export resolution (1.0 = 72 DPI)
WSLA_IMAGE_RESOLUTION_SCALE=2.0
```

### Docker Volumes

Persistent data is stored in Docker volumes:
- `wsla_uploads` - Uploaded PDF files
- `wsla_output` - Parsed outputs (markdown, images, metadata)

## Development Setup

### Local Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install "docling-slim[format-pdf,convert-core,models-local,feat-ocr-rapidocr-onnx,cli]"
pip install -r requirements.txt

# Run development server
uvicorn app:app --reload --port 8000
```

### Project Structure

```
wsla_service/
├── app.py              # FastAPI application and endpoints
├── config.py           # Configuration via pydantic-settings
├── models.py           # Pydantic models for API responses
├── parser_service.py   # Docling integration and PDF parsing
├── pdf_detector.py     # PDF type detection logic
├── requirements.txt    # Python dependencies
├── Dockerfile          # Multi-stage Docker build
├── docker-compose.yml  # GPU deployment config
├── docker-compose.cpu.yml  # CPU-only deployment config
└── .env.example        # Environment variable template
```

## Performance

Processing speed depends on:
- **Hardware**: GPU is ~10-20x faster than CPU for scanned PDFs
- **PDF Type**: Editable PDFs parse faster than scanned (no OCR needed)
- **Page Count**: Linear scaling with number of pages
- **Resolution**: Higher `IMAGE_RESOLUTION_SCALE` = slower processing

Typical throughput (14-page scanned PDF):
- **GPU (RTX 4050)**: ~2-3 pages/second
- **CPU (modern)**: ~0.2-0.5 pages/second

## Troubleshooting

### SSL Certificate Errors

If you encounter SSL certificate errors during model downloads:

```bash
# Check if certifi is installed
python -c "import certifi; print(certifi.where())"

# If needed, install certifi
pip install --upgrade certifi
```

### GPU Not Detected

```bash
# Verify NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# Check Docker GPU support
docker info | grep -i runtime
```

### Out of Memory

For large PDFs or limited RAM:

```bash
# Reduce thread count
export OMP_NUM_THREADS=2

# Force CPU mode
export WSLA_DEVICE=cpu
```

## License

See the main Docling repository for license information.

## Credits

Built with:
- [Docling](https://github.com/DS4SD/docling) - Document parsing and conversion
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [pypdfium2](https://github.com/pypdfium2-team/pypdfium2) - PDF processing
- [RapidOCR](https://github.com/RapidAI/RapidOCR) - OCR engine

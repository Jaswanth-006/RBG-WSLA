"""FastAPI application for WSLA Document Parsing.

Endpoints:
    POST /upload       — Upload a PDF, get back a doc_id
    GET  /parse/{id}   — Detect PDF type + parse via Docling → structured JSON
    GET  /status/{id}  — Check if a document exists and its parse status
"""

import logging
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from wsla_service.config import settings
from wsla_service.models import (
    ParseResponse,
    StatusResponse,
    UploadResponse,
)
from wsla_service.parser_service import parse_pdf
from wsla_service.pdf_detector import detect_pdf_type

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
_log = logging.getLogger(__name__)

app = FastAPI(
    title="WSLA Document Parsing API",
    description=(
        "Upload WSLA/PDF documents, detect whether they are editable or scanned, "
        "and parse them into structured Markdown with metadata (tables, images, "
        "paragraphs, sections) using the Docling pipeline."
    ),
    version="1.0.0",
)


@app.on_event("startup")
def _ensure_directories() -> None:
    """Create upload and output directories on startup."""
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    _log.info("Upload dir: %s", settings.upload_dir.resolve())
    _log.info("Output dir: %s", settings.output_dir.resolve())


# ─── Upload endpoint ─────────────────────────────────────────────────────────


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)) -> UploadResponse:
    """Upload a PDF file and receive a unique document ID.

    The file is saved to ``uploads/{doc_id}/{original_filename}``.
    """
    if file.filename is None or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted. Please upload a .pdf file.",
        )

    # Read file content
    content = await file.read()
    file_size = len(content)

    if file_size > settings.max_file_size:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File size ({file_size} bytes) exceeds the maximum "
                f"allowed size ({settings.max_file_size} bytes)."
            ),
        )

    if file_size == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Generate a unique document ID and save the file
    doc_id = str(uuid.uuid4())
    doc_dir = settings.upload_dir / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = doc_dir / file.filename
    pdf_path.write_bytes(content)

    _log.info("Uploaded %s (%d bytes) as doc_id=%s", file.filename, file_size, doc_id)

    return UploadResponse(
        doc_id=doc_id,
        filename=file.filename,
        file_size=file_size,
    )


# ─── Parse endpoint ──────────────────────────────────────────────────────────


@app.get("/parse/{doc_id}", response_model=ParseResponse)
def parse_document(doc_id: str) -> ParseResponse:
    """Detect PDF type (editable/scanned/mixed) and parse via Docling.

    1. Locates the uploaded PDF by ``doc_id``.
    2. Runs rule-based detection (per-page character count).
    3. Parses the PDF through Docling's pipeline (with OCR for scanned pages).
    4. Returns structured JSON with Markdown text and metadata.
    """
    doc_dir = settings.upload_dir / doc_id
    if not doc_dir.exists():
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found.")

    # Find the PDF file inside the doc directory
    pdf_files = list(doc_dir.glob("*.pdf"))
    if not pdf_files:
        raise HTTPException(
            status_code=404, detail=f"No PDF file found for document {doc_id}."
        )

    pdf_path = pdf_files[0]
    filename = pdf_path.name

    # ── Step 1: Detect PDF type ──────────────────────────────────────────
    _log.info("Detecting PDF type for %s (doc_id=%s)", filename, doc_id)
    pdf_type, page_classifications = detect_pdf_type(pdf_path)

    # ── Step 2: Parse via Docling ────────────────────────────────────────
    output_dir = settings.output_dir / doc_id
    _log.info("Parsing %s (type=%s) via Docling", filename, pdf_type.value)
    markdown_file, markdown_text, metadata = parse_pdf(pdf_path, output_dir)

    page_count = len(page_classifications)

    _log.info("Parse complete for %s: %d pages, type=%s", filename, page_count, pdf_type.value)

    return ParseResponse(
        doc_id=doc_id,
        filename=filename,
        pdf_type=pdf_type,
        page_classifications=page_classifications,
        page_count=page_count,
        markdown_file=markdown_file,
        markdown_text=markdown_text,
        metadata=metadata,
    )


# ─── Status endpoint ─────────────────────────────────────────────────────────


@app.get("/status/{doc_id}", response_model=StatusResponse)
def document_status(doc_id: str) -> StatusResponse:
    """Check whether a document has been uploaded and/or parsed."""
    doc_dir = settings.upload_dir / doc_id
    output_dir = settings.output_dir / doc_id

    if not doc_dir.exists():
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found.")

    pdf_files = list(doc_dir.glob("*.pdf"))
    filename = pdf_files[0].name if pdf_files else "unknown"

    # Check if output exists
    parsed = output_dir.exists() and any(output_dir.glob("*.md"))
    md_file = None
    pdf_type = None

    if parsed:
        md_files = list(output_dir.glob("*.md"))
        if md_files:
            md_file = str(md_files[0])
        # Re-detect type if needed
        if pdf_files:
            pdf_type, _ = detect_pdf_type(pdf_files[0])

    return StatusResponse(
        doc_id=doc_id,
        filename=filename,
        exists=True,
        parsed=parsed,
        pdf_type=pdf_type,
        markdown_file=md_file,
    )


# ─── Health check ────────────────────────────────────────────────────────────


@app.get("/health")
def health_check() -> dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "ok", "service": "WSLA Document Parsing API"}

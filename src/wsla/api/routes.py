"""The API endpoints.

    POST /upload       — Upload a PDF, get back a doc_id
    GET  /parse/{id}   — Detect PDF type + parse via Docling (PaddleOCR fallback) → structured JSON
    POST /process      — Upload + detect + parse + download as ZIP (all-in-one)
    GET  /status/{id}  — Check if a document exists and its parse status
    GET  /             — Health check
"""

import json
import logging
import shutil
import time
import uuid
import zipfile
from io import BytesIO

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from wsla.api.reporting import build_performance_payload, print_processing_summary
from wsla.config import settings
from wsla.pipeline.classify import detect_pdf_type
from wsla.pipeline.orchestrator import parse_pdf
from wsla.pipeline.recover import is_available as paddleocr_available
from wsla.runtime.device import get_runtime_summary
from wsla.runtime.metrics import calculate_throughput
from wsla.schemas import (
    ParseEngine,
    ParseResponse,
    PdfType,
    ProcessingMetrics,
    StatusResponse,
    UploadResponse,
)
from wsla.storage import documents

_log = logging.getLogger(__name__)

router = APIRouter()


# ─── Upload endpoint ─────────────────────────────────────────────────────────


@router.post("/upload", response_model=UploadResponse)
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

    documents.record_upload(doc_id, file.filename, file_size)
    _log.info("Uploaded %s (%d bytes) as doc_id=%s", file.filename, file_size, doc_id)

    return UploadResponse(
        doc_id=doc_id,
        filename=file.filename,
        file_size=file_size,
    )


# ─── Parse endpoint ──────────────────────────────────────────────────────────


@router.get("/parse/{doc_id}", response_model=ParseResponse)
def parse_document(doc_id: str, refresh: bool = False) -> ParseResponse:
    """Detect PDF type (editable/scanned/mixed) and parse via Docling.

    1. Locates the uploaded PDF by ``doc_id``.
    2. Runs rule-based detection (per-page character count).
    3. Parses the PDF through Docling's pipeline (with OCR for scanned pages).
    4. Re-reads pages Docling could not read with the PaddleOCR fallback.
    5. Returns structured JSON with Markdown text, metadata and a fallback report.
    """
    request_start = time.perf_counter()
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

    # ── Step 0: Serve the cached result unless a refresh was asked for ───
    # Parsing is expensive (tens of seconds); repeating it for every GET was
    # pure waste. Pass ?refresh=true to force a re-parse.
    cache_path = settings.output_dir / doc_id / "parse_result.json"
    if not refresh and cache_path.is_file():
        _log.info("Serving cached parse for doc_id=%s", doc_id)
        return ParseResponse.model_validate_json(cache_path.read_text(encoding="utf-8"))

    # ── Step 1: Detect PDF type ──────────────────────────────────────────
    _log.info("Detecting PDF type for %s (doc_id=%s)", filename, doc_id)
    detect_start = time.perf_counter()
    pdf_type, page_classifications = detect_pdf_type(pdf_path)
    detection_seconds = time.perf_counter() - detect_start

    # ── Step 2: Parse via Docling, PaddleOCR rescues weak pages ──────────
    output_dir = settings.output_dir / doc_id
    _log.info("Parsing %s (type=%s) via Docling", filename, pdf_type.value)
    parse_start = time.perf_counter()
    parsed = parse_pdf(pdf_path, output_dir, page_classifications)
    fallback_seconds = parsed.timing["fallback_seconds"]
    parsing_seconds = time.perf_counter() - parse_start - fallback_seconds
    total_seconds = time.perf_counter() - request_start

    page_count = len(page_classifications)

    runtime = get_runtime_summary()
    metrics = ProcessingMetrics(
        requested_device=runtime["requested_device"],
        resolved_device=runtime["resolved_device"],
        gpu_available=bool(runtime["gpu_available"]),
        gpu_name=runtime["gpu_name"],
        cuda_version=runtime["cuda_version"],
        gpu_count=int(runtime["gpu_count"]),
        upload_seconds=0.0,
        detection_seconds=detection_seconds,
        parsing_seconds=parsing_seconds,
        metadata_seconds=parsed.timing["metadata_seconds"],
        fallback_seconds=fallback_seconds,
        zip_seconds=0.0,
        total_seconds=total_seconds,
        page_count=page_count,
        fallback_pages=parsed.fallback.pages_recovered,
    )
    metrics.parsing_pages_per_second, metrics.parsing_pages_per_minute = calculate_throughput(
        page_count,
        metrics.parsing_seconds,
    )
    metrics.overall_pages_per_second, metrics.overall_pages_per_minute = calculate_throughput(
        page_count,
        metrics.total_seconds,
    )

    _log.info(
        "Parse complete for %s: %d pages, type=%s, engine=%s",
        filename, page_count, pdf_type.value, parsed.parse_engine.value,
    )

    response = ParseResponse(
        doc_id=doc_id,
        filename=filename,
        pdf_type=pdf_type,
        page_classifications=page_classifications,
        page_count=page_count,
        parse_engine=parsed.parse_engine,
        markdown_file=parsed.markdown_file,
        markdown_text=parsed.markdown_text,
        metadata=parsed.metadata,
        fallback=parsed.fallback,
    )

    # Cache the result and record what we learned, so repeat calls and /status
    # never touch the PDF again.
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(response.model_dump_json(indent=2), encoding="utf-8")
    documents.record_parse(
        doc_id,
        pdf_type=pdf_type.value,
        page_count=page_count,
        parse_engine=parsed.parse_engine.value,
        markdown_file=parsed.markdown_file,
        fallback_pages=parsed.fallback.pages_recovered,
    )
    return response


# ─── Process endpoint (all-in-one: upload → detect → parse → ZIP) ───────────


@router.post("/process")
def process_pdf(file: UploadFile = File(...)) -> StreamingResponse:
    """Upload a PDF, detect its type, parse it via Docling, and return a ZIP.

    This endpoint combines the entire workflow into a single call:
    1. Accepts a PDF upload.
    2. Detects whether the PDF is editable, scanned, or mixed.
    3. Parses through Docling's StandardPdfPipeline (with OCR for scanned pages),
       re-reading any page Docling could not with the PaddleOCR fallback.
    4. Packages the parsed markdown, metadata JSON, and extracted images into a
       ZIP archive and streams it back for download.

    Declared ``def`` rather than ``async def`` on purpose: the work below is
    tens of seconds of CPU/GPU time, so FastAPI runs it in a worker thread and
    the event loop stays free for other requests.

    The ZIP contains:
        {pdf_stem}/
        ├── {pdf_stem}.md          — Full parsed markdown
        ├── metadata.json          — Structured metadata (tables, images, paragraphs,
        │                            sections) plus parse engine and fallback report
        ├── performance.json       — Performance metrics and timing
        └── images/
            ├── {pdf_stem}-picture-0.png
            ├── {pdf_stem}-picture-1.png
            └── ...
    """
    request_start = time.perf_counter()
    runtime = get_runtime_summary()

    # ── Validate upload ──────────────────────────────────────────────────
    if file.filename is None or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted. Please upload a .pdf file.",
        )

    content = file.file.read()
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

    # ── Save the PDF ─────────────────────────────────────────────────────
    upload_start = time.perf_counter()
    doc_id = str(uuid.uuid4())
    doc_dir = settings.upload_dir / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = doc_dir / file.filename
    pdf_path.write_bytes(content)
    upload_seconds = time.perf_counter() - upload_start
    pdf_stem = pdf_path.stem

    _log.info(
        "[process] Uploaded %s (%d bytes) as doc_id=%s",
        file.filename, file_size, doc_id,
    )

    # ── Step 1: Detect PDF type ──────────────────────────────────────────
    _log.info("[process] Detecting PDF type for %s", file.filename)
    detect_start = time.perf_counter()
    pdf_type, page_classifications = detect_pdf_type(pdf_path)
    detection_seconds = time.perf_counter() - detect_start

    # ── Step 2: Parse via Docling, PaddleOCR rescues weak pages ──────────
    output_dir = settings.output_dir / doc_id
    _log.info("[process] Parsing %s (type=%s) via Docling", file.filename, pdf_type.value)
    parse_start = time.perf_counter()
    parsed = parse_pdf(pdf_path, output_dir, page_classifications)
    fallback_seconds = parsed.timing["fallback_seconds"]
    parsing_seconds = time.perf_counter() - parse_start - fallback_seconds
    page_count = len(page_classifications)

    _log.info(
        "[process] Parse complete for %s: %d pages, type=%s, engine=%s",
        file.filename, page_count, pdf_type.value, parsed.parse_engine.value,
    )

    # ── Build performance metrics ────────────────────────────────────────
    processing_metrics = ProcessingMetrics(
        requested_device=runtime["requested_device"],
        resolved_device=runtime["resolved_device"],
        gpu_available=bool(runtime["gpu_available"]),
        gpu_name=runtime["gpu_name"],
        cuda_version=runtime["cuda_version"],
        gpu_count=int(runtime["gpu_count"]),
        upload_seconds=upload_seconds,
        detection_seconds=detection_seconds,
        parsing_seconds=parsing_seconds,
        metadata_seconds=parsed.timing["metadata_seconds"],
        fallback_seconds=fallback_seconds,
        zip_seconds=0.0,
        total_seconds=0.0,
        page_count=page_count,
        fallback_pages=parsed.fallback.pages_recovered,
    )
    processing_metrics.parsing_pages_per_second, processing_metrics.parsing_pages_per_minute = calculate_throughput(
        page_count,
        processing_metrics.parsing_seconds,
    )

    # ── Step 3: Build ZIP in memory ──────────────────────────────────────
    # The markdown and images go in first; the JSON files are added afterwards,
    # once the totals they report actually exist.
    zip_start = time.perf_counter()
    zip_buffer = BytesIO()
    zip_prefix = pdf_stem  # folder name inside the zip

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        md_path = output_dir / f"{pdf_stem}.md"
        if md_path.exists():
            zf.write(md_path, f"{zip_prefix}/{pdf_stem}.md")

        for img_file in sorted(output_dir.glob("*.png")):
            zf.write(img_file, f"{zip_prefix}/images/{img_file.name}")

    processing_metrics.zip_seconds = time.perf_counter() - zip_start
    processing_metrics.total_seconds = time.perf_counter() - request_start
    processing_metrics.overall_pages_per_second, processing_metrics.overall_pages_per_minute = calculate_throughput(
        page_count,
        processing_metrics.total_seconds,
    )

    # ── Add the JSON reports, now that the metrics are complete ──────────
    metadata_dict = {
        "doc_id": doc_id,
        "filename": file.filename,
        "pdf_type": pdf_type.value,
        "page_count": page_count,
        "parse_engine": parsed.parse_engine.value,
        "page_classifications": [pc.model_dump() for pc in page_classifications],
        "metadata": parsed.metadata.model_dump(),
        "fallback": parsed.fallback.model_dump(),
        "processing_metrics": processing_metrics.model_dump(),
    }
    performance_payload = build_performance_payload(processing_metrics, file.filename)

    zip_buffer.seek(0)
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            f"{zip_prefix}/metadata.json",
            json.dumps(metadata_dict, indent=2, ensure_ascii=False),
        )
        zf.writestr(
            f"{zip_prefix}/performance.json",
            json.dumps(performance_payload, indent=2, ensure_ascii=False),
        )

    zip_buffer.seek(0)
    zip_filename = f"{pdf_stem}_parsed.zip"
    print_processing_summary(
        file.filename,
        page_count,
        pdf_type.value,
        parsed.parse_engine.value,
        processing_metrics,
    )
    _log.info("[process] Returning ZIP %s for doc_id=%s", zip_filename, doc_id)

    # ── Step 4: Cleanup uploaded PDF and outputs ────────────────────────
    try:
        # Remove uploaded PDF directory
        if doc_dir.exists():
            shutil.rmtree(doc_dir)
            _log.info("[cleanup] Removed uploaded PDF directory: %s", doc_dir)

        # Remove output directory (markdown and images)
        if output_dir.exists():
            shutil.rmtree(output_dir)
            _log.info("[cleanup] Removed output directory: %s", output_dir)
        documents.forget(doc_id)
    except Exception as e:
        _log.warning("[cleanup] Failed to cleanup files for doc_id=%s: %s", doc_id, e)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{zip_filename}"',
        },
    )


# ─── Status endpoint ─────────────────────────────────────────────────────────


@router.get("/status/{doc_id}", response_model=StatusResponse)
def document_status(doc_id: str) -> StatusResponse:
    """Check whether a document has been uploaded and/or parsed."""
    doc_dir = settings.upload_dir / doc_id
    output_dir = settings.output_dir / doc_id

    if not doc_dir.exists():
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found.")

    pdf_files = list(doc_dir.glob("*.pdf"))
    record = documents.get_document(doc_id)
    filename = (record or {}).get("filename") or (
        pdf_files[0].name if pdf_files else "unknown"
    )

    parsed = output_dir.exists() and any(output_dir.glob("*.md"))
    md_file = None
    pdf_type = None
    parse_engine = None

    if record and record.get("pdf_type"):
        # Everything we need was stored at parse time - no need to re-read the PDF.
        pdf_type = PdfType(record["pdf_type"])
        parse_engine = (
            ParseEngine(record["parse_engine"]) if record.get("parse_engine") else None
        )
        md_file = record.get("markdown_file")

    if parsed and md_file is None:
        md_files = list(output_dir.glob("*.md"))
        if md_files:
            md_file = str(md_files[0])
    if parsed and pdf_type is None and pdf_files:
        # Document parsed before the store existed: fall back to detection.
        pdf_type, _ = detect_pdf_type(pdf_files[0])

    return StatusResponse(
        doc_id=doc_id,
        filename=filename,
        exists=True,
        parsed=parsed,
        pdf_type=pdf_type,
        parse_engine=parse_engine,
        markdown_file=md_file,
    )


# ─── Health check ────────────────────────────────────────────────────────────


@router.get("/")
def health_check() -> dict[str, object]:
    """Health check, including which parsing engines are available."""
    return {
        "status": "ok",
        "service": "WSLA Document Parsing API",
        "engines": {
            "docling": True,
            "paddleocr": paddleocr_available(),
        },
        "ocr_fallback_enabled": settings.enable_ocr_fallback,
    }

"""FastAPI application for WSLA Document Parsing.

Endpoints:
    POST /upload       — Upload a PDF, get back a doc_id
    GET  /parse/{id}   — Detect PDF type + parse via Docling (PaddleOCR fallback) → structured JSON
    POST /process      — Upload + detect + parse + download as ZIP (all-in-one)
    GET  /status/{id}  — Check if a document exists and its parse status
"""

import json
import logging
import os
import shutil
import sys
import time
import uuid
import zipfile
from io import BytesIO

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from config import settings

# Load environment variables from .env file
load_dotenv()
import storage
from models import (
    ParseEngine,
    ParseResponse,
    PdfType,
    ProcessingMetrics,
    StatusResponse,
    UploadResponse,
)
from ocr_fallback import is_available as paddleocr_available
from parser_service import (
    calculate_throughput,
    get_runtime_summary,
    parse_pdf,
)
from pdf_detector import detect_pdf_type

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
        "paragraphs, sections) using the Docling pipeline, with PaddleOCR "
        "re-reading any page Docling could not."
    ),
    version="1.0.0",
)


def _print_runtime_summary() -> None:
    """Print a clean startup summary for CPU execution."""
    runtime = get_runtime_summary()
    if not settings.enable_ocr_fallback:
        fallback = "disabled"
    elif paddleocr_available():
        fallback = f"PaddleOCR ({settings.paddle_rec_model})"
    else:
        fallback = "unavailable (paddleocr not installed)"
    print("=" * 60)
    print("WSLA Document Parsing API")
    print("=" * 60)
    print(f"Device           : {runtime['resolved_device']}")
    print(f"OCR fallback     : {fallback}")
    print("=" * 60)


def _build_performance_payload(metrics: ProcessingMetrics, filename: str) -> dict:
    return {
        "document": filename,
        "page_count": metrics.page_count,
        "fallback_pages": metrics.fallback_pages,
        "device": metrics.resolved_device,
        "timing_seconds": {
            "upload": round(metrics.upload_seconds, 6),
            "detection": round(metrics.detection_seconds, 6),
            "parsing": round(metrics.parsing_seconds, 6),
            "metadata": round(metrics.metadata_seconds, 6),
            "fallback": round(metrics.fallback_seconds, 6),
            "zip": round(metrics.zip_seconds, 6),
            "total": round(metrics.total_seconds, 6),
        },
        "throughput": {
            "parsing_pages_per_second": round(metrics.parsing_pages_per_second, 6),
            "parsing_pages_per_minute": round(metrics.parsing_pages_per_minute, 6),
            "overall_pages_per_second": round(metrics.overall_pages_per_second, 6),
            "overall_pages_per_minute": round(metrics.overall_pages_per_minute, 6),
        },
    }


def _print_processing_summary(
    filename: str,
    page_count: int,
    pdf_type: str,
    parse_engine: str,
    metrics: ProcessingMetrics,
) -> None:
    """Print a terminal-friendly PDF processing summary."""
    total = metrics.total_seconds if metrics.total_seconds > 0 else 1.0
    stage_rows = [
        ("Upload / Save", metrics.upload_seconds),
        ("PDF Detection", metrics.detection_seconds),
        ("Docling Parsing", metrics.parsing_seconds),
        ("Metadata Extraction", metrics.metadata_seconds),
        ("PaddleOCR Fallback", metrics.fallback_seconds),
        ("ZIP Creation", metrics.zip_seconds),
    ]

    print("=" * 80)
    print("WSLA PDF PROCESSING PERFORMANCE")
    print("=" * 80)
    print(f"Document              : {filename}")
    print(f"Pages                 : {page_count}")
    print(f"PDF Type              : {pdf_type}")
    print(f"Parse Engine          : {parse_engine}")
    print(f"Fallback Pages        : {metrics.fallback_pages}")
    print(f"Device                : {metrics.resolved_device}")
    print("-" * 80)
    print(f"{'Stage':<30} {'Time (sec)':>15} {'% Total':>12}")
    print("-" * 80)
    for label, value in stage_rows:
        pct = (value / total) * 100 if total > 0 else 0.0
        print(f"{label:<30} {value:>15.2f} {pct:>11.1f}%")
    print("-" * 80)
    print(f"{'Total':<30} {metrics.total_seconds:>15.2f} {100.0:>11.1f}%")
    print("-" * 80)
    print(f"Docling throughput     : {metrics.parsing_pages_per_second:.2f} pages/sec")
    print(f"Docling throughput     : {metrics.parsing_pages_per_minute:.2f} pages/min")
    print(f"Overall throughput     : {metrics.overall_pages_per_second:.2f} pages/sec")
    print(f"Overall throughput     : {metrics.overall_pages_per_minute:.2f} pages/min")
    print("=" * 80)
    _log.info(
        "[PERF] doc=%s pages=%s device=%s engine=%s parse=%.2fs fallback=%.2fs "
        "total=%.2fs pps=%.2f ppm=%.2f",
        filename,
        page_count,
        metrics.resolved_device,
        parse_engine,
        metrics.parsing_seconds,
        metrics.fallback_seconds,
        metrics.total_seconds,
        metrics.parsing_pages_per_second,
        metrics.parsing_pages_per_minute,
    )


@app.on_event("startup")
def _ensure_directories() -> None:
    """Create upload and output directories on startup."""
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    storage.init_db()
    _log.info("Upload dir: %s", settings.upload_dir.resolve())
    _log.info("Output dir: %s", settings.output_dir.resolve())
    _print_runtime_summary()


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

    storage.record_upload(doc_id, file.filename, file_size)
    _log.info("Uploaded %s (%d bytes) as doc_id=%s", file.filename, file_size, doc_id)

    return UploadResponse(
        doc_id=doc_id,
        filename=file.filename,
        file_size=file_size,
    )


# ─── Parse endpoint ──────────────────────────────────────────────────────────


@app.get("/parse/{doc_id}", response_model=ParseResponse)
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
    storage.record_parse(
        doc_id,
        pdf_type=pdf_type.value,
        page_count=page_count,
        parse_engine=parsed.parse_engine.value,
        markdown_file=parsed.markdown_file,
        fallback_pages=parsed.fallback.pages_recovered,
    )
    return response


# ─── Process endpoint (all-in-one: upload → detect → parse → ZIP) ───────────


@app.post("/process")
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
    performance_payload = _build_performance_payload(processing_metrics, file.filename)

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
    _print_processing_summary(
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
        storage.forget(doc_id)
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


@app.get("/status/{doc_id}", response_model=StatusResponse)
def document_status(doc_id: str) -> StatusResponse:
    """Check whether a document has been uploaded and/or parsed."""
    doc_dir = settings.upload_dir / doc_id
    output_dir = settings.output_dir / doc_id

    if not doc_dir.exists():
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found.")

    pdf_files = list(doc_dir.glob("*.pdf"))
    record = storage.get_document(doc_id)
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


@app.get("/")
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


# ─── Application startup ─────────────────────────────────────────────────────


if __name__ == "__main__":
    import uvicorn

    # Read PORT from environment variable
    port = os.getenv("PORT")

    if not port:
        print("ERROR: PORT environment variable is not set in .env file")
        sys.exit(1)

    try:
        port_int = int(port)
    except ValueError:
        print(f"ERROR: PORT value '{port}' is not a valid integer")
        sys.exit(1)

    print(f"Starting WSLA Document Parsing API on port {port_int}")

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port_int,
        log_level="info",
    )

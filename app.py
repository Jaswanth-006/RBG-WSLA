"""FastAPI application for WSLA Document Parsing.

Endpoints:
    POST /upload       — Upload a PDF, get back a doc_id
    GET  /parse/{id}   — Detect PDF type + parse via Docling → structured JSON
    POST /process      — Upload + detect + parse + download as ZIP (all-in-one)
    GET  /status/{id}  — Check if a document exists and its parse status
"""

import asyncio
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
from models import (
    ParseResponse,
    ProcessingMetrics,
    StatusResponse,
    UploadResponse,
)
from parser_service import (
    calculate_throughput,
    get_runtime_summary,
    parse_pdf,
)
from pdf_detector import detect_pdf_type
from scheduler import CpuAwarePdfScheduler

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


# One scheduler for the process. It detects the CPU count visible inside the
# container and keeps at least two logical CPUs per concurrent PDF when more
# than two CPUs are available.
pdf_scheduler = CpuAwarePdfScheduler()


def _print_runtime_summary() -> None:
    """Print a clean startup summary for CPU execution."""
    runtime = get_runtime_summary()
    print("=" * 60)
    print("WSLA Document Parsing API")
    print("=" * 60)
    print(f"Device           : {runtime['resolved_device']}")
    print("=" * 60)


def _build_performance_payload(metrics: ProcessingMetrics, filename: str) -> dict:
    return {
        "document": filename,
        "page_count": metrics.page_count,
        "device": metrics.resolved_device,
        "timing_seconds": {
            "upload": round(metrics.upload_seconds, 6),
            "detection": round(metrics.detection_seconds, 6),
            "parsing": round(metrics.parsing_seconds, 6),
            "metadata": round(metrics.metadata_seconds, 6),
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
    metrics: ProcessingMetrics,
) -> None:
    """Print a terminal-friendly PDF processing summary."""
    total = metrics.total_seconds if metrics.total_seconds > 0 else 1.0
    stage_rows = [
        ("Upload / Save", metrics.upload_seconds),
        ("PDF Detection", metrics.detection_seconds),
        ("Docling Parsing", metrics.parsing_seconds),
        ("Metadata Extraction", metrics.metadata_seconds),
        ("ZIP Creation", metrics.zip_seconds),
    ]

    print("=" * 80)
    print("WSLA PDF PROCESSING PERFORMANCE")
    print("=" * 80)
    print(f"Document              : {filename}")
    print(f"Pages                 : {page_count}")
    print(f"PDF Type              : {pdf_type}")
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
        "[PERF] doc=%s pages=%s device=%s parse=%.2fs total=%.2fs pps=%.2f ppm=%.2f",
        filename,
        page_count,
        metrics.resolved_device,
        metrics.parsing_seconds,
        metrics.total_seconds,
        metrics.parsing_pages_per_second,
        metrics.parsing_pages_per_minute,
    )


@app.on_event("startup")
def _ensure_directories() -> None:
    """Create upload and output directories on startup."""
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
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

    # ── Step 1: Detect PDF type ──────────────────────────────────────────
    _log.info("Detecting PDF type for %s (doc_id=%s)", filename, doc_id)
    detect_start = time.perf_counter()
    pdf_type, page_classifications = detect_pdf_type(pdf_path)
    detection_seconds = time.perf_counter() - detect_start

    # ── Step 2: Parse via Docling ────────────────────────────────────────
    output_dir = settings.output_dir / doc_id
    _log.info("Parsing %s (type=%s) via Docling", filename, pdf_type.value)
    parse_start = time.perf_counter()
    markdown_file, markdown_text, metadata, parse_timing = parse_pdf(
        pdf_path,
        output_dir,
        include_timing=True,
    )
    parsing_seconds = time.perf_counter() - parse_start
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
        metadata_seconds=parse_timing.get("metadata_seconds", 0.0),
        zip_seconds=0.0,
        total_seconds=total_seconds,
        page_count=page_count,
    )
    metrics.parsing_pages_per_second, metrics.parsing_pages_per_minute = calculate_throughput(
        page_count,
        metrics.parsing_seconds,
    )
    metrics.overall_pages_per_second, metrics.overall_pages_per_minute = calculate_throughput(
        page_count,
        metrics.total_seconds,
    )

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


# ─── Process endpoint (all-in-one: upload → detect → parse → ZIP) ───────────


def _process_saved_pdf(
    *,
    doc_id: str,
    file_path,
    filename: str,
    file_size: int,
    upload_seconds: float,
    allocated_cores: int,
) -> tuple[bytes, str]:
    """Run detection, Docling parsing, ZIP creation, and cleanup for one job.

    ``allocated_cores`` is the scheduler's logical CPU budget for this job.
    The shared Docling converter remains a single instance; Docling itself
    controls its internal thread pools.
    """
    request_start = time.perf_counter()
    pdf_path = file_path
    doc_dir = pdf_path.parent
    output_dir = settings.output_dir / doc_id
    runtime = get_runtime_summary()

    _log.info(
        "[process] job=%s allocated_cores=%d starting %s",
        doc_id,
        allocated_cores,
        filename,
    )

    # ── Step 1: Detect PDF type ──────────────────────────────────────────
    _log.info("[process] Detecting PDF type for %s", filename)
    detect_start = time.perf_counter()
    pdf_type, page_classifications = detect_pdf_type(pdf_path)
    detection_seconds = time.perf_counter() - detect_start

    # ── Step 2: Parse via shared Docling converter ───────────────────────
    _log.info(
        "[process] Parsing %s (type=%s, allocated_cores=%d) via Docling",
        filename,
        pdf_type.value,
        allocated_cores,
    )
    parse_start = time.perf_counter()
    markdown_file, markdown_text, metadata, parse_timing = parse_pdf(
        pdf_path,
        output_dir,
        include_timing=True,
    )
    parsing_seconds = time.perf_counter() - parse_start
    page_count = len(page_classifications)

    _log.info(
        "[process] Parse complete for %s: %d pages, type=%s",
        filename,
        page_count,
        pdf_type.value,
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
        metadata_seconds=parse_timing.get("metadata_seconds", 0.0),
        zip_seconds=0.0,
        total_seconds=0.0,
        page_count=page_count,
    )
    processing_metrics.parsing_pages_per_second, processing_metrics.parsing_pages_per_minute = calculate_throughput(
        page_count,
        processing_metrics.parsing_seconds,
    )

    # ── Step 3: Build ZIP in memory ──────────────────────────────────────
    zip_start = time.perf_counter()
    zip_buffer = BytesIO()
    zip_prefix = pdf_path.stem

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        md_path = output_dir / f"{pdf_path.stem}.md"
        if md_path.exists():
            zf.write(md_path, f"{zip_prefix}/{pdf_path.stem}.md")

        metadata_dict = {
            "doc_id": doc_id,
            "filename": filename,
            "pdf_type": pdf_type.value,
            "page_count": page_count,
            "page_classifications": [
                pc.model_dump() for pc in page_classifications
            ],
            "metadata": metadata.model_dump(),
        }
        metadata_dict["processing_metrics"] = processing_metrics.model_dump()
        zf.writestr(
            f"{zip_prefix}/metadata.json",
            json.dumps(metadata_dict, indent=2, ensure_ascii=False),
        )

        for img_file in sorted(output_dir.glob("*.png")):
            zf.write(img_file, f"{zip_prefix}/images/{img_file.name}")

    processing_metrics.zip_seconds = time.perf_counter() - zip_start
    processing_metrics.total_seconds = time.perf_counter() - request_start
    processing_metrics.overall_pages_per_second, processing_metrics.overall_pages_per_minute = calculate_throughput(
        page_count,
        processing_metrics.total_seconds,
    )

    performance_payload = _build_performance_payload(processing_metrics, filename)
    zip_buffer.seek(0)
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            f"{zip_prefix}/performance.json",
            json.dumps(performance_payload, indent=2, ensure_ascii=False),
        )

    zip_buffer.seek(0)
    zip_bytes = zip_buffer.getvalue()
    zip_filename = f"{pdf_path.stem}_parsed.zip"

    _print_processing_summary(
        filename,
        page_count,
        pdf_type.value,
        processing_metrics,
    )

    # ── Step 4: Cleanup ──────────────────────────────────────────────────
    try:
        if doc_dir.exists():
            shutil.rmtree(doc_dir)
            _log.info("[cleanup] Removed uploaded PDF directory: %s", doc_dir)

        if output_dir.exists():
            shutil.rmtree(output_dir)
            _log.info("[cleanup] Removed output directory: %s", output_dir)
    except Exception as exc:
        _log.warning(
            "[cleanup] Failed to cleanup files for doc_id=%s: %s",
            doc_id,
            exc,
        )

    _log.info(
        "[process] Returning ZIP %s for doc_id=%s allocated_cores=%d",
        zip_filename,
        doc_id,
        allocated_cores,
    )
    return zip_bytes, zip_filename


@app.post("/process")
async def process_pdf(file: UploadFile = File(...)) -> StreamingResponse:
    """Upload a PDF, queue it, process it, and return the ZIP.

    Requests are placed into the CPU-aware scheduler. The HTTP request waits
    for its own job while other PDFs may be processed concurrently.
    """
    if file.filename is None or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted. Please upload a .pdf file.",
        )

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

    upload_start = time.perf_counter()
    doc_id = str(uuid.uuid4())
    doc_dir = settings.upload_dir / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = doc_dir / file.filename
    pdf_path.write_bytes(content)
    upload_seconds = time.perf_counter() - upload_start

    _log.info(
        "[process] Uploaded %s (%d bytes) as doc_id=%s",
        file.filename,
        file_size,
        doc_id,
    )

    job_id, future = pdf_scheduler.submit(
        lambda allocated_cores: _process_saved_pdf(
            doc_id=doc_id,
            file_path=pdf_path,
            filename=file.filename,
            file_size=file_size,
            upload_seconds=upload_seconds,
            allocated_cores=allocated_cores,
        )
    )

    _log.info(
        "[process] Queued doc_id=%s job_id=%s scheduler=%s",
        doc_id,
        job_id,
        pdf_scheduler.snapshot(),
    )

    try:
        zip_bytes, zip_filename = await asyncio.wrap_future(future)
    except Exception as exc:
        # The worker normally performs cleanup on successful processing. If
        # an unexpected exception occurs before that, clean up this request's
        # temporary files here.
        try:
            if doc_dir.exists():
                shutil.rmtree(doc_dir)
            output_dir = settings.output_dir / doc_id
            if output_dir.exists():
                shutil.rmtree(output_dir)
        except Exception:
            _log.warning(
                "[cleanup] Failed after job error for doc_id=%s",
                doc_id,
                exc_info=True,
            )

        _log.exception("[process] Job failed doc_id=%s job_id=%s", doc_id, job_id)
        raise HTTPException(
            status_code=500,
            detail=f"PDF processing failed: {exc}",
        ) from exc

    return StreamingResponse(
        BytesIO(zip_bytes),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{zip_filename}"',
            "X-WSLA-Job-ID": job_id,
        },
    )


@app.get("/scheduler")
def scheduler_status() -> dict[str, object]:
    """Return the current CPU scheduler state."""
    return pdf_scheduler.snapshot()


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


@app.get("/")
def health_check() -> dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "ok", "service": "WSLA Document Parsing API"}


@app.on_event("shutdown")
def _shutdown_scheduler() -> None:
    """Cleanly stop scheduler workers during container shutdown."""
    pdf_scheduler.shutdown()


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


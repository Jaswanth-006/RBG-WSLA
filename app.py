"""FastAPI application for WSLA Document Parsing.

Endpoints:
    POST /upload       — Upload a PDF, get back a doc_id
    GET  /parse/{id}   — Detect PDF type + parse via Docling → structured JSON
    POST /process      — Upload + detect + parse + download as ZIP (all-in-one)
    GET  /status/{id}  — Check if a document exists and its parse status
"""

import json
import logging
import time
import uuid
import zipfile
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from config import settings
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


def _print_runtime_summary() -> None:
    """Print a clean startup summary for the selected runtime."""
    runtime = get_runtime_summary(settings.device)
    gpu_name = runtime["gpu_name"] if runtime["gpu_name"] else "unavailable"
    cuda_version = runtime["cuda_version"] if runtime["cuda_version"] else "n/a"
    print("=" * 60)
    print("WSLA Document Parsing API")
    print("=" * 60)
    print(f"Requested device : {runtime['requested_device']}")
    print(f"Resolved device  : {runtime['resolved_device']}")
    print(f"GPU available    : {'yes' if runtime['gpu_available'] else 'no'}")
    print(f"GPU name         : {gpu_name}")
    print(f"CUDA version     : {cuda_version}")
    print("=" * 60)


def _build_performance_payload(metrics: ProcessingMetrics, filename: str) -> dict:
    return {
        "document": filename,
        "page_count": metrics.page_count,
        "requested_device": metrics.requested_device,
        "resolved_device": metrics.resolved_device,
        "gpu_available": metrics.gpu_available,
        "gpu_name": metrics.gpu_name,
        "cuda_version": metrics.cuda_version,
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
    print(f"Requested Device      : {metrics.requested_device}")
    print(f"Resolved Device       : {metrics.resolved_device}")
    print(f"GPU                   : {metrics.gpu_name if metrics.gpu_name else 'unavailable'}")
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
    """Upload a PDF file and receive a unique document ID."""
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
    """Detect PDF type and parse via Docling."""
    request_start = time.perf_counter()
    doc_dir = settings.upload_dir / doc_id
    if not doc_dir.exists():
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found.")

    pdf_files = list(doc_dir.glob("*.pdf"))
    if not pdf_files:
        raise HTTPException(
            status_code=404, detail=f"No PDF file found for document {doc_id}."
        )

    pdf_path = pdf_files[0]
    filename = pdf_path.name

    detect_start = time.perf_counter()
    pdf_type, page_classifications = detect_pdf_type(pdf_path)
    detection_seconds = time.perf_counter() - detect_start

    parse_start = time.perf_counter()
    markdown_file, markdown_text, metadata, parse_timing = parse_pdf(
        pdf_path,
        settings.output_dir / doc_id,
        include_timing=True,
    )
    parsing_seconds = time.perf_counter() - parse_start
    total_seconds = time.perf_counter() - request_start

    runtime = get_runtime_summary(settings.device)
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
        page_count=len(page_classifications),
    )
    metrics.parsing_pages_per_second, metrics.parsing_pages_per_minute = calculate_throughput(
        metrics.page_count,
        metrics.parsing_seconds,
    )
    metrics.overall_pages_per_second, metrics.overall_pages_per_minute = calculate_throughput(
        metrics.page_count,
        metrics.total_seconds,
    )

    return ParseResponse(
        doc_id=doc_id,
        filename=filename,
        pdf_type=pdf_type,
        page_classifications=page_classifications,
        page_count=len(page_classifications),
        markdown_file=markdown_file,
        markdown_text=markdown_text,
        metadata=metadata,
    )


# ─── Process endpoint (all-in-one: upload → detect → parse → ZIP) ───────────


@app.post("/process")
async def process_pdf(file: UploadFile = File(...)) -> StreamingResponse:
    """Upload a PDF, detect its type, parse it via Docling, and return a ZIP."""
    request_start = time.perf_counter()
    runtime = get_runtime_summary(settings.device)

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
    pdf_stem = pdf_path.stem

    detect_start = time.perf_counter()
    pdf_type, page_classifications = detect_pdf_type(pdf_path)
    detection_seconds = time.perf_counter() - detect_start

    output_dir = settings.output_dir / doc_id
    parse_start = time.perf_counter()
    markdown_file, markdown_text, metadata, parse_timing = parse_pdf(
        pdf_path,
        output_dir,
        include_timing=True,
    )
    parsing_seconds = time.perf_counter() - parse_start
    page_count = len(page_classifications)

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

    zip_start = time.perf_counter()
    zip_buffer = BytesIO()
    zip_prefix = pdf_stem

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        md_path = output_dir / f"{pdf_stem}.md"
        if md_path.exists():
            zf.write(md_path, f"{zip_prefix}/{pdf_stem}.md")

        metadata_dict = {
            "doc_id": doc_id,
            "filename": file.filename,
            "pdf_type": pdf_type.value,
            "page_count": page_count,
            "page_classifications": [pc.model_dump() for pc in page_classifications],
            "metadata": metadata.model_dump(),
        }
        metadata_dict["processing_metrics"] = processing_metrics.model_dump()
        zf.writestr(f"{zip_prefix}/metadata.json", json.dumps(metadata_dict, indent=2, ensure_ascii=False))

        for img_file in sorted(output_dir.glob("*.png")):
            zf.write(img_file, f"{zip_prefix}/images/{img_file.name}")

    processing_metrics.zip_seconds = time.perf_counter() - zip_start
    processing_metrics.total_seconds = time.perf_counter() - request_start
    processing_metrics.overall_pages_per_second, processing_metrics.overall_pages_per_minute = calculate_throughput(
        page_count,
        processing_metrics.total_seconds,
    )

    performance_payload = _build_performance_payload(processing_metrics, file.filename)
    zip_buffer.seek(0)
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            f"{zip_prefix}/performance.json",
            json.dumps(performance_payload, indent=2, ensure_ascii=False),
        )

    zip_buffer.seek(0)
    zip_filename = f"{pdf_stem}_parsed.zip"
    _print_processing_summary(file.filename, page_count, pdf_type.value, processing_metrics)
    _log.info("[process] Returning ZIP %s for doc_id=%s", zip_filename, doc_id)

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
    filename = pdf_files[0].name if pdf_files else "unknown"

    parsed = output_dir.exists() and any(output_dir.glob("*.md"))
    md_file = None
    pdf_type = None

    if parsed:
        md_files = list(output_dir.glob("*.md"))
        if md_files:
            md_file = str(md_files[0])
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


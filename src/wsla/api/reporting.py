"""Performance reporting: the startup banner, performance.json, and the terminal summary."""

import logging

from wsla.config import settings
from wsla.pipeline.recover import is_available as paddleocr_available
from wsla.runtime.device import get_runtime_summary
from wsla.schemas import ProcessingMetrics

_log = logging.getLogger(__name__)


def print_runtime_summary() -> None:
    """Print the startup banner: resolved device and OCR fallback status."""
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


def build_performance_payload(metrics: ProcessingMetrics, filename: str) -> dict:
    """The contents of performance.json."""
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


def print_processing_summary(
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

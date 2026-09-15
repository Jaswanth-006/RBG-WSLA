"""Run stages 2-5 for one PDF: extract, verify and recover, then assemble.

Stage 2 (classify) runs in the API layer first, because its result is reported
to the caller; this module receives the classifications only for the page count.
"""

import logging
import time
from pathlib import Path
from typing import NamedTuple

from wsla.pipeline.assemble import merge_recovered_text
from wsla.pipeline.extract import build_converter, count_pages, extract_metadata, page_text
from wsla.pipeline.recover import run_fallback
from wsla.schemas import (
    DocumentMetadata,
    FallbackReport,
    PageClassification,
    ParseEngine,
)

_log = logging.getLogger(__name__)


class ParseOutput(NamedTuple):
    """Everything produced by parsing one PDF."""

    markdown_file: str
    markdown_text: str
    metadata: DocumentMetadata
    timing: dict[str, float]
    parse_engine: ParseEngine
    fallback: FallbackReport


def parse_pdf(
    pdf_path: Path,
    output_dir: Path,
    page_classifications: list[PageClassification] | None = None,
) -> ParseOutput:
    """Parse a PDF with Docling, rescuing unreadable pages with PaddleOCR.

    Args:
        pdf_path: Path to the uploaded PDF file.
        output_dir: Directory to save the markdown and image outputs.
        page_classifications: Per-page detection results, used for the total
            page count. Docling only reports pages that produced content, so
            a page it dropped entirely is otherwise invisible. When omitted,
            pages are counted from the PDF.

    Returns:
        A ParseOutput with the markdown path and text, metadata, stage timings,
        the engine that produced the output, and the fallback report.

    Raises:
        Exception: Docling's error, re-raised when Docling fails and the
            PaddleOCR fallback cannot run.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if page_classifications is not None:
        page_count = len(page_classifications)
    else:
        page_count = count_pages(pdf_path)

    markdown_text = ""
    metadata = DocumentMetadata()
    markdown_seconds = 0.0
    metadata_seconds = 0.0
    docling_exc: Exception | None = None

    # ── Stage 3: extract with Docling ────────────────────────────────────
    conv_start = time.perf_counter()
    try:
        converter = build_converter()
        _log.info("Starting Docling conversion for %s", pdf_path.name)
        conv_start = time.perf_counter()
        conv_result = converter.convert(pdf_path)
    except Exception as exc:
        conversion_seconds = time.perf_counter() - conv_start
        docling_exc = exc
        _log.error(
            "Docling conversion failed for %s after %.4fs",
            pdf_path.name,
            conversion_seconds,
            exc_info=True,
        )
    else:
        conversion_seconds = time.perf_counter() - conv_start
        _log.info(
            "Docling conversion complete for %s in %.4fs", pdf_path.name, conversion_seconds
        )

        document = conv_result.document

        markdown_start = time.perf_counter()
        markdown_text = document.export_to_markdown()
        markdown_seconds = time.perf_counter() - markdown_start

        metadata, metadata_seconds = extract_metadata(document, pdf_path, output_dir)

    # ── Stage 4: verify each page, re-read the weak ones ─────────────────
    fallback, rescued = run_fallback(
        pdf_path,
        page_count,
        page_text(metadata),
        docling_error=None if docling_exc is None else f"{type(docling_exc).__name__}: {docling_exc}",
    )
    if docling_exc is not None and not fallback.triggered:
        raise docling_exc

    # ── Stage 5: assemble ────────────────────────────────────────────────
    markdown_text = merge_recovered_text(markdown_text, metadata, rescued)

    if docling_exc is not None:
        parse_engine = ParseEngine.PADDLEOCR
    elif rescued:
        parse_engine = ParseEngine.DOCLING_PADDLEOCR
    else:
        parse_engine = ParseEngine.DOCLING

    # Save markdown with the same stem name as the original PDF
    md_path = output_dir / (pdf_path.stem + ".md")
    md_path.write_text(markdown_text, encoding="utf-8")
    _log.info("Markdown saved to %s (engine=%s)", md_path, parse_engine.value)

    timing_info = {
        "markdown_export_seconds": markdown_seconds,
        "conversion_seconds": conversion_seconds,
        "metadata_seconds": metadata_seconds,
        "fallback_seconds": fallback.seconds,
    }

    return ParseOutput(
        markdown_file=str(md_path),
        markdown_text=markdown_text,
        metadata=metadata,
        timing=timing_info,
        parse_engine=parse_engine,
        fallback=fallback,
    )

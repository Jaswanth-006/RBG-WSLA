"""Docling integration: convert PDFs and extract structured metadata.

Uses Docling's DocumentConverter pipeline to parse PDFs and extracts tables,
images, paragraphs, and section headers into structured Pydantic models.

Docling OCRs scanned pages with EasyOCR (Spanish). Any page it still cannot
read is re-read by the PaddleOCR fallback in ``ocr_fallback``, a different OCR
architecture, so the two engines do not fail in the same places.
"""

import logging
import re
import time
from io import StringIO
from pathlib import Path
from typing import NamedTuple

import pypdfium2 as pdfium
import torch
from docling_core.types.doc import PictureItem, SectionHeaderItem, TableItem, TextItem

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from config import settings
from models import (
    DocumentMetadata,
    FallbackReport,
    ImageMetadata,
    PageClassification,
    ParagraphMetadata,
    ParseEngine,
    SectionMetadata,
    TableMetadata,
)
from ocr_fallback import RescuedPage, run_fallback

_log = logging.getLogger(__name__)

# Paragraph label for text recovered by the PaddleOCR fallback
OCR_FALLBACK_LABEL = "ocr_fallback"


class ParseOutput(NamedTuple):
    """Everything produced by parsing one PDF."""

    markdown_file: str
    markdown_text: str
    metadata: DocumentMetadata
    timing: dict[str, float]
    parse_engine: ParseEngine
    fallback: FallbackReport


def calculate_throughput(page_count: int, elapsed_seconds: float) -> tuple[float, float]:
    """Return pages/sec and pages/minute with zero-safe handling."""
    if elapsed_seconds <= 0:
        return 0.0, 0.0
    pages_per_second = page_count / elapsed_seconds
    return pages_per_second, pages_per_second * 60.0


def get_gpu_info() -> tuple[bool, str | None, str | None, int]:
    """Read actual CUDA/GPU state from PyTorch."""
    try:
        if not torch.cuda.is_available():
            return False, None, None, 0
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else None
        return True, gpu_name, torch.version.cuda, gpu_count
    except Exception:
        return False, None, None, 0


def resolve_device(requested_device: str) -> str:
    """Resolve the requested mode (auto/gpu/cpu) to a runtime device."""
    normalized = (requested_device or "auto").strip().lower()
    if normalized not in {"auto", "gpu", "cpu"}:
        raise ValueError("Unsupported device setting. Use one of: auto, gpu, cpu.")

    if normalized == "cpu":
        return "cpu"
    if normalized == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "WSLA_DEVICE=gpu was requested, but CUDA/GPU is unavailable."
            )
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_runtime_summary(requested_device: str | None = None) -> dict[str, object]:
    """Return a runtime summary used for startup output and metrics."""
    requested = (requested_device or settings.device or "auto").strip().lower()
    resolved = resolve_device(requested)
    gpu_available, gpu_name, cuda_version, gpu_count = get_gpu_info()
    return {
        "requested_device": requested,
        "resolved_device": resolved,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "cuda_version": cuda_version,
        "gpu_count": gpu_count,
    }


def _build_converter(requested_device: str | None = None) -> DocumentConverter:
    """Create a DocumentConverter for PDF parsing with OCR on the chosen device."""
    resolved_device = resolve_device(requested_device or settings.device)
    accelerator_options = AcceleratorOptions(device=resolved_device)
    pipeline_options = PdfPipelineOptions(accelerator_options=accelerator_options)
    pipeline_options.images_scale = settings.image_resolution_scale
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = True
    pipeline_options.generate_table_images = False

    # Docling OCRs image regions with EasyOCR in Spanish. RapidOCR, Docling's
    # other option, is not usable here: its checkpoints are served from
    # ModelScope, which our network cannot reach. EasyOCR is also a different
    # architecture from the PaddleOCR fallback, so the two engines fail on
    # different pages — which is what makes the fallback worth having.
    #
    # The models are baked into DOCLING_ARTIFACTS_PATH. When that is set and
    # model_storage_directory is left unset, Docling points EasyOCR at those
    # files and turns downloading off, so the container never needs network.
    pipeline_options.do_ocr = True
    # use_gpu is deliberately not set: Docling derives it from
    # accelerator_options.device, and setting it explicitly is deprecated.
    pipeline_options.ocr_options = EasyOcrOptions(lang=["es"])

    _log.info("Initializing Docling converter (device=%s)", resolved_device)

    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        },
    )


def _count_pages(pdf_path: Path) -> int:
    """Return the number of pages in a PDF."""
    pdf_doc = pdfium.PdfDocument(pdf_path)
    try:
        return len(pdf_doc)
    finally:
        pdf_doc.close()


def _extract_metadata(
    document, pdf_path: Path, output_dir: Path
) -> tuple[DocumentMetadata, float]:
    """Extract tables, images, paragraphs and sections from a Docling document.

    Returns:
        The metadata and the seconds spent extracting images and text items.
    """
    # ── Extract tables ───────────────────────────────────────────────────
    tables: list[TableMetadata] = []
    for table_ix, table in enumerate(document.tables):
        page_no = None
        if table.prov and len(table.prov) > 0:
            page_no = table.prov[0].page_no

        # Export to different formats
        try:
            df = table.export_to_dataframe(doc=document)
            csv_buf = StringIO()
            df.to_csv(csv_buf, index=False)
            csv_data = csv_buf.getvalue()
            markdown_data = df.to_markdown(index=False)
        except Exception:
            _log.warning("Could not export table %d to dataframe", table_ix)
            csv_data = ""
            markdown_data = ""

        try:
            html_data = table.export_to_html(doc=document)
        except Exception:
            html_data = ""

        tables.append(
            TableMetadata(
                index=table_ix,
                page_no=page_no,
                num_rows=table.data.num_rows if table.data else 0,
                num_cols=table.data.num_cols if table.data else 0,
                csv_data=csv_data,
                html_data=html_data,
                markdown_data=markdown_data,
            )
        )

    # ── Extract images and paragraphs ────────────────────────────────────
    metadata_start = time.perf_counter()
    images: list[ImageMetadata] = []
    paragraphs: list[ParagraphMetadata] = []
    sections: list[SectionMetadata] = []

    picture_counter = 0
    paragraph_counter = 0
    section_counter = 0

    for element, _level in document.iterate_items():
        # Determine page number from provenance
        page_no = None
        if hasattr(element, "prov") and element.prov and len(element.prov) > 0:
            page_no = element.prov[0].page_no

        if isinstance(element, PictureItem):
            # Save picture as PNG
            image_filename = f"{pdf_path.stem}-picture-{picture_counter}.png"
            image_path = output_dir / image_filename
            try:
                pil_image = element.get_image(document)
                if pil_image is not None:
                    pil_image.save(str(image_path), format="PNG")
                    _log.info("Saved picture %d to %s", picture_counter, image_path)
            except Exception:
                _log.warning(
                    "Could not save image for picture %d", picture_counter, exc_info=True
                )

            images.append(
                ImageMetadata(
                    index=picture_counter,
                    page_no=page_no,
                    image_path=str(image_path),
                )
            )
            picture_counter += 1

        elif isinstance(element, SectionHeaderItem):
            sections.append(
                SectionMetadata(
                    index=section_counter,
                    page_no=page_no,
                    level=_level,
                    text=element.text or "",
                )
            )
            section_counter += 1

        elif isinstance(element, TextItem):
            label = element.label.value if element.label else "paragraph"
            paragraphs.append(
                ParagraphMetadata(
                    index=paragraph_counter,
                    page_no=page_no,
                    label=label,
                    text=element.text or "",
                )
            )
            paragraph_counter += 1

    metadata = DocumentMetadata(
        tables=tables,
        images=images,
        paragraphs=paragraphs,
        sections=sections,
    )
    metadata_seconds = time.perf_counter() - metadata_start

    _log.info(
        "Extracted %d tables, %d images, %d paragraphs, %d sections from %s in %.4fs",
        len(tables),
        len(images),
        len(paragraphs),
        len(sections),
        pdf_path.name,
        metadata_seconds,
    )
    return metadata, metadata_seconds


def _page_text(metadata: DocumentMetadata) -> dict[int, str]:
    """The text Docling produced, gathered per page.

    Paragraphs, section headers and table contents all count, so a page that is
    one large table is not mistaken for an unreadable one. The fallback needs
    the text itself and not just its length, because a page can be full of
    characters and still be garbled.
    """
    items = [
        *((paragraph.page_no, paragraph.text) for paragraph in metadata.paragraphs),
        *((section.page_no, section.text) for section in metadata.sections),
        *((table.page_no, table.markdown_data) for table in metadata.tables),
    ]
    pages: dict[int, list[str]] = {}
    for page_no, text in items:
        if page_no is not None and text:
            pages.setdefault(page_no, []).append(text)
    return {page_no: "\n".join(parts) for page_no, parts in pages.items()}


def _strip_pages_from_markdown(markdown_text: str, texts: list[str]) -> str:
    """Remove specific Docling fragments from the exported markdown.

    Used when a page is replaced rather than supplemented: the text Docling
    produced for it was wrong, so leaving it in would duplicate — and contradict
    — the corrected text appended further down.
    """
    for text in texts:
        fragment = text.strip()
        if len(fragment) >= 4 and fragment in markdown_text:
            markdown_text = markdown_text.replace(fragment, "", 1)
    # tidy up the holes: empty headings and runs of blank lines
    markdown_text = re.sub(r"(?m)^#{1,6}\s*$", "", markdown_text)
    return re.sub(r"\n{3,}", "\n\n", markdown_text)


def _merge_recovered_text(
    markdown_text: str,
    metadata: DocumentMetadata,
    rescued: dict[int, "RescuedPage"],
) -> str:
    """Fold PaddleOCR's text back into the metadata and the markdown.

    Recovered blocks are inserted into ``metadata.paragraphs`` at their page's
    position, so paragraphs stay in page order, and appended to the markdown as
    marked sections — Docling's markdown has no page anchors, so splicing
    mid-document is not reliable.

    When a page is flagged ``replace`` (Docling's text was wrong, not missing),
    Docling's own items for that page are dropped from both the metadata and the
    markdown, so the corrected text does not sit next to the broken text.
    """
    if not rescued:
        return markdown_text

    replaced_pages = {page_no for page_no, page in rescued.items() if page.replace}
    if replaced_pages:
        stale = [
            item.text
            for item in (*metadata.paragraphs, *metadata.sections)
            if item.page_no in replaced_pages
        ]
        markdown_text = _strip_pages_from_markdown(markdown_text, stale)
        metadata.paragraphs = [
            item for item in metadata.paragraphs if item.page_no not in replaced_pages
        ]
        metadata.sections = [
            item for item in metadata.sections if item.page_no not in replaced_pages
        ]

    paragraphs = metadata.paragraphs
    markdown_sections: list[str] = []
    for page_no in sorted(rescued):
        blocks = rescued[page_no].blocks

        insert_at = len(paragraphs)
        for position, paragraph in enumerate(paragraphs):
            if paragraph.page_no is not None and paragraph.page_no > page_no:
                insert_at = position
                break
        paragraphs[insert_at:insert_at] = [
            ParagraphMetadata(index=0, page_no=page_no, label=OCR_FALLBACK_LABEL, text=block)
            for block in blocks
        ]

        markdown_sections.append(
            f"## Page {page_no} (OCR fallback)\n\n" + "\n\n".join(blocks)
        )

    for index, paragraph in enumerate(paragraphs):
        paragraph.index = index

    appended = "\n\n".join(markdown_sections)
    if not markdown_text.strip():
        return appended + "\n"
    return f"{markdown_text.rstrip()}\n\n{appended}\n"


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
        page_count = _count_pages(pdf_path)

    markdown_text = ""
    metadata = DocumentMetadata()
    markdown_seconds = 0.0
    metadata_seconds = 0.0
    docling_exc: Exception | None = None

    conv_start = time.perf_counter()
    try:
        converter = _build_converter()
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

        # ── Export full markdown ──────────────────────────────────────────
        markdown_start = time.perf_counter()
        markdown_text = document.export_to_markdown()
        markdown_seconds = time.perf_counter() - markdown_start

        metadata, metadata_seconds = _extract_metadata(document, pdf_path, output_dir)

    # ── PaddleOCR fallback for pages Docling could not read ──────────────
    fallback, rescued = run_fallback(
        pdf_path,
        page_count,
        _page_text(metadata),
        docling_error=None if docling_exc is None else f"{type(docling_exc).__name__}: {docling_exc}",
    )
    if docling_exc is not None and not fallback.triggered:
        raise docling_exc

    markdown_text = _merge_recovered_text(markdown_text, metadata, rescued)

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

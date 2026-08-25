"""Docling integration: convert PDFs and extract structured metadata.

Uses Docling's DocumentConverter pipeline to parse PDFs (both editable and
scanned via OCR) and extracts tables, images, paragraphs, and section headers
into structured Pydantic models.
"""

import logging
import time
from io import StringIO
from pathlib import Path

from docling_core.types.doc import PictureItem, SectionHeaderItem, TableItem, TextItem

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    RapidOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

from config import settings
from models import (
    DocumentMetadata,
    ImageMetadata,
    ParagraphMetadata,
    SectionMetadata,
    TableMetadata,
)

_log = logging.getLogger(__name__)


def calculate_throughput(page_count: int, elapsed_seconds: float) -> tuple[float, float]:
    """Return pages/sec and pages/minute with zero-safe handling."""
    if elapsed_seconds <= 0:
        return 0.0, 0.0
    pages_per_second = page_count / elapsed_seconds
    return pages_per_second, pages_per_second * 60.0


def get_runtime_summary(requested_device: str | None = None) -> dict[str, object]:
    """Return a runtime summary for CPU-only execution."""
    return {
        "requested_device": "cpu",
        "resolved_device": "cpu",
        "gpu_available": False,
        "gpu_name": None,
        "cuda_version": None,
        "gpu_count": 0,
    }


def _build_converter() -> DocumentConverter:
    """Create a DocumentConverter configured for CPU-only PDF parsing with OCR."""
    # Force CPU-only execution by explicitly setting device to "cpu"
    accelerator_options = AcceleratorOptions(device="cpu")
    pipeline_options = PdfPipelineOptions(accelerator_options=accelerator_options)
    pipeline_options.images_scale = settings.image_resolution_scale
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = True
    pipeline_options.generate_table_images = False

    # Enable OCR for scanned pages — Docling applies OCR only where needed
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options = RapidOcrOptions()

    _log.info("Initializing Docling converter with CPU-only mode (device=cpu)")

    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        },
    )


converter = _build_converter()
def parse_pdf(
    pdf_path: Path,
    output_dir: Path,
    *,
    include_timing: bool = False,
) -> tuple[str, str, DocumentMetadata] | tuple[str, str, DocumentMetadata, dict[str, float]]:
    """Parse a PDF using Docling and extract all metadata.

    Args:
        pdf_path: Path to the uploaded PDF file.
        output_dir: Directory to save the markdown and image outputs.
        include_timing: If True, return timing information as a fourth element.

    Returns:
        A tuple of (markdown_file_path, markdown_text, metadata) or
        (markdown_file_path, markdown_text, metadata, timing_info) if include_timing=True.
    """
    output_dir.mkdir(parents=True, exist_ok=True)


    _log.info("Starting Docling conversion for %s", pdf_path.name)
    conv_start = time.perf_counter()
    conv_result = converter.convert(pdf_path)
    conversion_seconds = time.perf_counter() - conv_start
    _log.info("Docling conversion complete for %s in %.4fs", pdf_path.name, conversion_seconds)

    document = conv_result.document

    # ── Export full markdown ──────────────────────────────────────────────
    markdown_start = time.perf_counter()
    markdown_text = document.export_to_markdown()
    markdown_seconds = time.perf_counter() - markdown_start

    # Save markdown with the same stem name as the original PDF
    md_filename = pdf_path.stem + ".md"
    md_path = output_dir / md_filename
    md_path.write_text(markdown_text, encoding="utf-8")
    _log.info("Markdown saved to %s", md_path)

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

    timing_info = {
        "markdown_export_seconds": markdown_seconds,
        "conversion_seconds": conversion_seconds,
        "metadata_seconds": metadata_seconds,
    }

    if include_timing:
        return str(md_path), markdown_text, metadata, timing_info
    return str(md_path), markdown_text, metadata

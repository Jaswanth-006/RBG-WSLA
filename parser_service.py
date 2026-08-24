"""Docling integration: convert PDFs and extract structured metadata.

Uses Docling's DocumentConverter pipeline to parse PDFs (both editable and
scanned via OCR) and extracts tables, images, paragraphs, and section headers
into structured Pydantic models.
"""

import logging
import time
from io import StringIO
from pathlib import Path

import torch
from docling_core.types.doc import PictureItem, SectionHeaderItem, TextItem

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
    """Resolve the user-requested mode to a runtime value."""
    normalized = (requested_device or "auto").strip().lower()
    if normalized not in {"auto", "gpu", "cpu"}:
        raise ValueError("Unsupported device setting. Use one of: auto, gpu, cpu.")

    if normalized == "cpu":
        return "cpu"
    if normalized == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "ERROR: WSLA_DEVICE=gpu was requested, but CUDA/GPU is unavailable."
            )
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_runtime_summary(requested_device: str | None = None) -> dict[str, object]:
    """Return a runtime summary used for startup and metrics."""
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
    """Create a Docling converter configured for PDF parsing with OCR."""
    resolved_device = resolve_device(requested_device or settings.device)
    accelerator_options = AcceleratorOptions(device=resolved_device)
    pipeline_options = PdfPipelineOptions(accelerator_options=accelerator_options)
    pipeline_options.images_scale = settings.image_resolution_scale
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = True
    pipeline_options.generate_table_images = False
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options = RapidOcrOptions()

    _log.info(
        "Initializing Docling converter with resolved device=%s (GPU available=%s)",
        resolved_device,
        torch.cuda.is_available(),
    )

    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        },
    )


def parse_pdf(
    pdf_path: Path,
    output_dir: Path,
    *,
    include_timing: bool = False,
) -> tuple[str, str, DocumentMetadata] | tuple[str, str, DocumentMetadata, dict[str, float]]:
    """Parse a PDF file and extract metadata while preserving existing output behavior."""
    output_dir.mkdir(parents=True, exist_ok=True)
    converter = _build_converter(settings.device)

    _log.info("Starting Docling conversion for %s", pdf_path.name)
    conv_start = time.perf_counter()
    conv_result = converter.convert(pdf_path)
    conversion_seconds = time.perf_counter() - conv_start
    _log.info("Docling conversion complete for %s in %.4fs", pdf_path.name, conversion_seconds)

    document = conv_result.document

    markdown_start = time.perf_counter()
    markdown_text = document.export_to_markdown()
    markdown_seconds = time.perf_counter() - markdown_start

    md_filename = pdf_path.stem + ".md"
    md_path = output_dir / md_filename
    md_path.write_text(markdown_text, encoding="utf-8")
    _log.info("Markdown saved to %s", md_path)

    metadata_start = time.perf_counter()
    tables: list[TableMetadata] = []
    for table_ix, table in enumerate(document.tables):
        page_no = None
        if table.prov and len(table.prov) > 0:
            page_no = table.prov[0].page_no

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

    images: list[ImageMetadata] = []
    paragraphs: list[ParagraphMetadata] = []
    sections: list[SectionMetadata] = []

    picture_counter = 0
    paragraph_counter = 0
    section_counter = 0

    for element, _level in document.iterate_items():
        page_no = None
        if hasattr(element, "prov") and element.prov and len(element.prov) > 0:
            page_no = element.prov[0].page_no

        if isinstance(element, PictureItem):
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

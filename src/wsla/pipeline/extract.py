"""Stage 3 — extract: turn the PDF into structured content with Docling.

Docling renders each page, its layout model marks regions (headings, paragraphs,
figures, tables), EasyOCR reads regions with no embedded text, and TableFormer
works out table structure. This module configures that converter and flattens
the resulting document into the metadata the API returns.

Because OCR runs inside Docling's layout pass, tables on scanned pages receive
real text cells instead of coming out empty.
"""

import logging
import time
from io import StringIO
from pathlib import Path

import pypdfium2 as pdfium
from docling_core.types.doc import PictureItem, SectionHeaderItem, TextItem

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from wsla.config import settings
from wsla.runtime.device import resolve_device
from wsla.schemas import (
    DocumentMetadata,
    ImageMetadata,
    ParagraphMetadata,
    SectionMetadata,
    TableMetadata,
)

_log = logging.getLogger(__name__)


def build_converter(requested_device: str | None = None) -> DocumentConverter:
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


def count_pages(pdf_path: Path) -> int:
    """Return the number of pages in a PDF."""
    pdf_doc = pdfium.PdfDocument(pdf_path)
    try:
        return len(pdf_doc)
    finally:
        pdf_doc.close()


def extract_metadata(
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


def page_text(metadata: DocumentMetadata) -> dict[int, str]:
    """The text Docling produced, gathered per page.

    Paragraphs, section headers and table contents all count, so a page that is
    one large table is not mistaken for an unreadable one. Stage 4 needs the text
    itself and not just its length, because a page can be full of characters and
    still be garbled.
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

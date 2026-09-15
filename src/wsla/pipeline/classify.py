"""Stage 2 — classify: is this PDF digital, scanned, or a mix?

Rule-based detection using pypdfium2 (already a Docling dependency): count the
text characters each page already carries and classify by threshold. No
rendering and no models, so this takes milliseconds.

The result is reported to the caller and supplies the page count to stage 4. It
does not route anything — Docling decides per region whether OCR is needed.
"""

import logging
from pathlib import Path

import pypdfium2 as pdfium

from wsla.config import settings
from wsla.schemas import PageClassification, PdfType

_log = logging.getLogger(__name__)


def _extract_page_char_count(pdf_doc: pdfium.PdfDocument, page_index: int) -> int:
    """Return the number of text characters on a single page."""
    try:
        page = pdf_doc[page_index]
        text_page = page.get_textpage()
        char_count = text_page.count_chars()
        text_page.close()
        page.close()
        return max(char_count, 0)
    except Exception:
        _log.warning("Failed to read text from page %d", page_index + 1, exc_info=True)
        return 0


def detect_pdf_type(
    pdf_path: Path,
) -> tuple[PdfType, list[PageClassification]]:
    """Classify each page of a PDF as editable or scanned.

    Returns:
        A tuple of (overall PdfType, per-page classifications).
        - If ALL pages are editable → PdfType.EDITABLE
        - If ALL pages are scanned → PdfType.SCANNED
        - If a mix of both → PdfType.MIXED
    """
    threshold = settings.editable_char_threshold

    pdf_doc = pdfium.PdfDocument(pdf_path)
    page_count = len(pdf_doc)

    classifications: list[PageClassification] = []
    editable_count = 0
    scanned_count = 0

    for page_index in range(page_count):
        char_count = _extract_page_char_count(pdf_doc, page_index)

        if char_count >= threshold:
            page_type = PdfType.EDITABLE
            editable_count += 1
        else:
            page_type = PdfType.SCANNED
            scanned_count += 1

        classifications.append(
            PageClassification(
                page_no=page_index + 1,
                char_count=char_count,
                classification=page_type,
            )
        )

    pdf_doc.close()

    # Determine overall classification
    if scanned_count == 0:
        overall = PdfType.EDITABLE
    elif editable_count == 0:
        overall = PdfType.SCANNED
    else:
        overall = PdfType.MIXED

    _log.info(
        "PDF %s: %d pages — %d editable, %d scanned → %s",
        pdf_path.name,
        page_count,
        editable_count,
        scanned_count,
        overall.value,
    )

    return overall, classifications

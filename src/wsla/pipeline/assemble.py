"""Stage 5 — assemble: fold rescued text back into the document.

Recovered blocks are inserted into the paragraph list at their page's position,
so downstream consumers get a page-ordered sequence, and appended to the
markdown as marked sections. When a page's Docling text was wrong rather than
missing, that text is removed from both, so the page never carries two versions.
"""

import re
from typing import TYPE_CHECKING

from wsla.schemas import DocumentMetadata, ParagraphMetadata

if TYPE_CHECKING:
    from wsla.pipeline.recover import RescuedPage

# Paragraph label for text recovered by the PaddleOCR fallback
OCR_FALLBACK_LABEL = "ocr_fallback"


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


def merge_recovered_text(
    markdown_text: str,
    metadata: DocumentMetadata,
    rescued: "dict[int, RescuedPage]",
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

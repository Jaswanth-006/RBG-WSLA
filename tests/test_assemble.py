"""Stage 5 assembly: replacing garbled pages versus supplementing empty ones."""

import pytest

from wsla.pipeline import assemble
from wsla.schemas import DocumentMetadata, ParagraphMetadata, SectionMetadata

recover = pytest.importorskip("wsla.pipeline.recover")


def test_replaced_page_drops_docling_text_everywhere(good_spanish, garbled_spanish):
    metadata = DocumentMetadata(
        paragraphs=[
            ParagraphMetadata(index=0, page_no=1, text="buena pagina uno"),
            ParagraphMetadata(index=1, page_no=2, text=garbled_spanish),
            ParagraphMetadata(index=2, page_no=3, text="buena pagina tres"),
        ],
        sections=[SectionMetadata(index=0, page_no=2, text="SECCION MALA")],
    )
    markdown = f"buena pagina uno\n\n## SECCION MALA\n\n{garbled_spanish}\n\nbuena pagina tres"
    rescued = {2: recover.RescuedPage(blocks=[good_spanish], replace=True)}

    out = assemble.merge_recovered_text(markdown, metadata, rescued)

    # the markdown loses the garbled text and gains the correction
    assert garbled_spanish not in out
    assert "SECCION MALA" not in out
    assert good_spanish in out
    assert "buena pagina uno" in out and "buena pagina tres" in out

    # the metadata is corrected, stays in page order, and is re-indexed
    assert [p.text for p in metadata.paragraphs] == [
        "buena pagina uno",
        good_spanish,
        "buena pagina tres",
    ]
    assert [(p.page_no, p.index) for p in metadata.paragraphs] == [(1, 0), (2, 1), (3, 2)]
    assert metadata.sections == []


def test_supplemented_page_keeps_docling_text():
    metadata = DocumentMetadata(
        paragraphs=[ParagraphMetadata(index=0, page_no=1, text="algo")]
    )
    rescued = {2: recover.RescuedPage(blocks=["nuevo"], replace=False)}

    out = assemble.merge_recovered_text("algo", metadata, rescued)

    assert "algo" in out and "nuevo" in out
    assert out.endswith("## Page 2 (OCR fallback)\n\nnuevo\n")


def test_nothing_rescued_leaves_markdown_untouched():
    assert assemble.merge_recovered_text("abc", DocumentMetadata(), {}) == "abc"

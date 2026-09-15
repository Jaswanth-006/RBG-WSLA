"""Stage 1 persistence: the SQLite document store."""

import pytest

from wsla.config import settings
from wsla.storage import documents


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "output_dir", tmp_path)
    documents.init_db()
    return documents


def test_upload_is_recorded(store):
    store.record_upload("doc-1", "a.pdf", 123)
    assert store.get_document("doc-1")["filename"] == "a.pdf"


def test_parse_result_is_recorded(store):
    store.record_upload("doc-1", "a.pdf", 123)
    store.record_parse(
        "doc-1",
        pdf_type="scanned",
        page_count=14,
        parse_engine="docling+paddleocr",
        markdown_file="/x/a.md",
        fallback_pages=3,
    )
    record = store.get_document("doc-1")
    assert (record["pdf_type"], record["page_count"], record["fallback_pages"]) == (
        "scanned",
        14,
        3,
    )


def test_unknown_id_returns_none(store):
    assert store.get_document("missing") is None


def test_forget_removes_the_record(store):
    store.record_upload("doc-1", "a.pdf", 123)
    store.forget("doc-1")
    assert store.get_document("doc-1") is None

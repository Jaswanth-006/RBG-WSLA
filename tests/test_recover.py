"""Stage 4 recovery: which pages get re-read, and in what order OCR lines are read."""

import pytest

from wsla.config import settings

# The recovery module needs numpy and pypdfium2 (not PaddleOCR itself).
recover = pytest.importorskip("wsla.pipeline.recover")


def test_weak_pages_are_selected_by_reason(good_spanish, garbled_spanish):
    pages = {1: good_spanish, 2: "x", 3: garbled_spanish, 4: good_spanish}
    assert recover._weak_pages(4, pages, None) == {2: "low_yield", 3: "poor_quality"}


def test_quality_trigger_can_be_switched_off(monkeypatch, good_spanish, garbled_spanish):
    monkeypatch.setattr(settings, "enable_quality_trigger", False)
    pages = {1: good_spanish, 2: "x", 3: garbled_spanish, 4: good_spanish}
    assert recover._weak_pages(4, pages, None) == {2: "low_yield"}


def test_docling_failure_marks_every_page():
    assert recover._weak_pages(3, {}, "RuntimeError: boom") == {
        1: "docling_error",
        2: "docling_error",
        3: "docling_error",
    }


def test_two_columns_are_read_left_column_first():
    line = recover.OcrLine
    lines = []
    for i in range(4):
        y = 100 + i * 40
        lines.append(line(f"izquierda{i}", 0.9, 0, y, 200, y + 20))
        lines.append(line(f"derecha{i}", 0.9, 400, y, 600, y + 20))

    joined = " ".join(recover._group_blocks(lines))

    assert joined.index("izquierda3") < joined.index("derecha0")
    assert all(f"izquierda{i}" in joined and f"derecha{i}" in joined for i in range(4))


def test_single_column_lines_still_group_into_rows():
    line = recover.OcrLine
    lines = [
        line("una", 0.9, 0, 10, 100, 30),
        line("linea", 0.9, 110, 10, 200, 30),
        line("segunda", 0.9, 0, 40, 200, 60),
    ]
    assert recover._group_blocks(lines) == ["una linea\nsegunda"]


def test_no_lines_means_no_blocks():
    assert recover._group_blocks([]) == []

"""Stage 4 — recover: re-read pages Docling got wrong or got nothing from.

Docling (with EasyOCR) remains the primary engine. A page is handed to PaddleOCR
when either signal fires:

* **low yield** — Docling produced almost no text for the page, or it failed
  outright, in which case every page qualifies;
* **poor quality** — the page is full of text, but the text reads as garbled
  Spanish. See ``wsla.pipeline.quality``. Such a page is *replaced* rather than
  added to, because what Docling produced is wrong rather than missing.

The engine is loaded lazily, once per process, and inference is serialised
behind a lock. When ``paddleocr`` is not installed the service keeps working and
the fallback reports itself as unavailable.
"""

import importlib.util
import logging
import statistics
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium

from wsla.config import settings
from wsla.pipeline.quality import count_alnum, is_garbled, quality_score
from wsla.schemas import FallbackReport, PageOcrResult

_log = logging.getLogger(__name__)

_engine = None
_engine_error: str | None = None
_engine_lock = threading.Lock()
_predict_lock = threading.Lock()


class FallbackUnavailableError(RuntimeError):
    """Raised when the PaddleOCR engine cannot be loaded."""


@dataclass
class OcrLine:
    """A single recognised text line and its bounding box in pixels."""

    text: str
    confidence: float
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def height(self) -> float:
        return max(self.y1 - self.y0, 1.0)

    @property
    def y_centre(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def x_centre(self) -> float:
        return (self.x0 + self.x1) / 2


@dataclass
class RescuedPage:
    """What PaddleOCR recovered for one page."""

    blocks: list[str] = field(default_factory=list)
    # True when Docling's own text for this page should be dropped, because it
    # was wrong rather than missing.
    replace: bool = False


def is_available() -> bool:
    """Return True if PaddleOCR can be used in this process.

    Checks the packages are installed without importing them (Paddle is slow to
    import), and returns False once an engine load has failed.
    """
    if _engine_error is not None:
        return False
    return (
        importlib.util.find_spec("paddleocr") is not None
        and importlib.util.find_spec("paddle") is not None
    )


def _model_kwargs() -> dict[str, str]:
    """Model names, plus local model directories when they are baked in."""
    kwargs = {
        "text_detection_model_name": settings.paddle_det_model,
        "text_recognition_model_name": settings.paddle_rec_model,
    }
    model_dir = settings.paddle_model_dir
    if model_dir is None:
        return kwargs

    det_dir = model_dir / settings.paddle_det_model
    rec_dir = model_dir / settings.paddle_rec_model
    if det_dir.is_dir() and rec_dir.is_dir():
        kwargs["text_detection_model_dir"] = str(det_dir)
        kwargs["text_recognition_model_dir"] = str(rec_dir)
    else:
        _log.warning(
            "PaddleOCR models not found under %s; PaddleOCR will resolve them "
            "from its cache or download them",
            model_dir,
        )
    return kwargs


def get_engine():
    """Return the process-wide PaddleOCR engine, loading it on first use.

    A failed load is remembered, so the fallback stays disabled for the rest of
    the process instead of retrying an expensive load on every request.
    """
    global _engine, _engine_error

    if _engine is not None:
        return _engine

    with _engine_lock:
        if _engine is not None:
            return _engine
        if _engine_error is not None:
            raise FallbackUnavailableError(_engine_error)

        load_start = time.perf_counter()
        try:
            from paddleocr import PaddleOCR

            _engine = PaddleOCR(
                device=settings.paddle_device,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=settings.paddle_use_textline_orientation,
                cpu_threads=settings.paddle_cpu_threads,
                enable_mkldnn=settings.paddle_enable_mkldnn,
                **_model_kwargs(),
            )
        except Exception as exc:
            _engine_error = f"{type(exc).__name__}: {exc}"
            _log.error(
                "Could not load PaddleOCR; OCR fallback disabled for this process",
                exc_info=True,
            )
            raise FallbackUnavailableError(_engine_error) from exc

        _log.info(
            "PaddleOCR engine loaded in %.2fs (device=%s, det=%s, rec=%s)",
            time.perf_counter() - load_start,
            settings.paddle_device,
            settings.paddle_det_model,
            settings.paddle_rec_model,
        )
        return _engine


def _render_page(pdf_doc: pdfium.PdfDocument, page_no: int) -> np.ndarray:
    """Render a 1-based page to a BGR array, the channel order PaddleOCR expects."""
    page = pdf_doc[page_no - 1]
    try:
        bitmap = page.render(scale=settings.fallback_dpi / 72)
        try:
            rgb = np.asarray(bitmap.to_pil().convert("RGB"))
        finally:
            bitmap.close()
    finally:
        page.close()
    return np.ascontiguousarray(rgb[:, :, ::-1])


def _recognise(image: np.ndarray) -> list[OcrLine]:
    """Run PaddleOCR on one page image, dropping low-confidence lines."""
    engine = get_engine()
    with _predict_lock:
        results = engine.predict(image)

    lines: list[OcrLine] = []
    for res in results:
        texts = res.get("rec_texts")
        scores = res.get("rec_scores")
        polys = res.get("rec_polys")
        if texts is None or scores is None or polys is None:
            continue

        for text, score, poly in zip(texts, scores, polys):
            text = (text or "").strip()
            confidence = float(score)
            if not text or confidence < settings.paddle_min_confidence:
                continue
            points = np.asarray(poly, dtype=float).reshape(-1, 2)
            lines.append(
                OcrLine(
                    text=text,
                    confidence=confidence,
                    x0=float(points[:, 0].min()),
                    y0=float(points[:, 1].min()),
                    x1=float(points[:, 0].max()),
                    y1=float(points[:, 1].max()),
                )
            )
    return lines


def _split_columns(lines: list[OcrLine]) -> list[list[OcrLine]]:
    """Split a page into columns along vertical gutters.

    Without this, two-column layouts and side-by-side flowchart boxes interleave:
    every line at the same height is read as one row, so the left and right
    columns alternate mid-sentence. A gutter is a horizontal gap that no line
    crosses, at least ``column_gutter_ratio`` of the page width wide.
    """
    if not settings.enable_column_split or len(lines) < 6:
        return [lines]

    left = min(line.x0 for line in lines)
    right = max(line.x1 for line in lines)
    width = right - left
    if width <= 0:
        return [lines]

    spans = sorted((line.x0, line.x1) for line in lines)
    gutters: list[tuple[float, float]] = []
    reach = spans[0][1]
    for x0, x1 in spans[1:]:
        if x0 > reach:
            gutters.append((reach, x0))
        reach = max(reach, x1)

    min_gutter = max(width * settings.column_gutter_ratio, 1.0)
    gutters = [g for g in gutters if g[1] - g[0] >= min_gutter]
    if not gutters:
        return [lines]

    bounds = [left] + [(a + b) / 2 for a, b in gutters] + [right + 1.0]
    columns: list[list[OcrLine]] = []
    for lo, hi in zip(bounds, bounds[1:]):
        column = [line for line in lines if lo <= line.x_centre < hi]
        if column:
            columns.append(column)
    return columns or [lines]


def _rows_to_blocks(lines: list[OcrLine]) -> list[str]:
    """Group one column's lines into rows, then rows into blocks."""
    rows: list[list[OcrLine]] = []
    for line in sorted(lines, key=lambda item: (item.y_centre, item.x0)):
        if rows:
            row = rows[-1]
            row_centre = statistics.fmean(item.y_centre for item in row)
            row_height = statistics.fmean(item.height for item in row)
            if abs(line.y_centre - row_centre) <= 0.5 * max(row_height, line.height):
                row.append(line)
                continue
        rows.append([line])

    blocks: list[list[str]] = []
    prev_bottom: float | None = None
    for row in rows:
        row.sort(key=lambda item: item.x0)
        top = min(item.y0 for item in row)
        height = statistics.fmean(item.height for item in row)
        text = " ".join(item.text for item in row)
        if prev_bottom is None or top - prev_bottom > height:
            blocks.append([text])
        else:
            blocks[-1].append(text)
        prev_bottom = max(item.y1 for item in row)

    return ["\n".join(block) for block in blocks]


def _group_blocks(lines: list[OcrLine]) -> list[str]:
    """Turn recognised lines into text blocks in reading order.

    PaddleOCR returns lines in detection order. Columns are separated first, then
    each column is read top to bottom, and columns are emitted left to right.
    """
    if not lines:
        return []

    blocks: list[str] = []
    for column in sorted(_split_columns(lines), key=lambda col: min(c.x0 for c in col)):
        blocks.extend(_rows_to_blocks(column))
    return blocks


def _ocr_page(
    pdf_doc: pdfium.PdfDocument,
    page_no: int,
    docling_text: str,
    reason: str,
) -> tuple[PageOcrResult, RescuedPage]:
    """OCR one page and decide whether its text beats what Docling produced."""
    start = time.perf_counter()
    docling_chars = count_alnum(docling_text)
    result = PageOcrResult(page_no=page_no, reason=reason, docling_chars=docling_chars)

    try:
        lines = _recognise(_render_page(pdf_doc, page_no))
    except FallbackUnavailableError:
        raise
    except Exception as exc:
        _log.warning("PaddleOCR failed on page %d", page_no, exc_info=True)
        result.error = f"{type(exc).__name__}: {exc}"
        result.seconds = time.perf_counter() - start
        return result, RescuedPage()

    blocks = _group_blocks(lines)
    ocr_text = "\n\n".join(blocks)
    result.line_count = len(lines)
    result.char_count = count_alnum(ocr_text)
    if lines:
        result.mean_confidence = statistics.fmean(line.confidence for line in lines)

    if reason == "poor_quality":
        # Docling's text is wrong, not missing: keep whichever reads better and
        # drop the loser, rather than ending up with both.
        result.recovered = quality_score(ocr_text) > quality_score(docling_text)
        result.replaced = result.recovered
    else:
        result.recovered = result.char_count > docling_chars

    result.seconds = time.perf_counter() - start
    if not result.recovered:
        return result, RescuedPage()
    return result, RescuedPage(blocks=blocks, replace=result.replaced)


def _weak_pages(
    page_count: int,
    page_text: dict[int, str],
    docling_error: str | None,
) -> dict[int, str]:
    """Return {page_no: reason} for every page that needs a second opinion."""
    pages = range(1, page_count + 1)
    if docling_error is not None:
        return {page: "docling_error" for page in pages}

    reasons: dict[int, str] = {}
    for page in pages:
        text = page_text.get(page, "")
        if count_alnum(text) < settings.fallback_min_chars_per_page:
            reasons[page] = "low_yield"
        elif settings.enable_quality_trigger and is_garbled(
            text,
            min_accent_rate=settings.fallback_min_accent_rate,
            min_letters=settings.quality_min_letters,
        ):
            reasons[page] = "poor_quality"
    return reasons


def run_fallback(
    pdf_path: Path,
    page_count: int,
    page_text: dict[int, str],
    *,
    docling_error: str | None = None,
) -> tuple[FallbackReport, dict[int, RescuedPage]]:
    """Re-read weak or garbled pages with PaddleOCR.

    Args:
        pdf_path: The source PDF.
        page_count: Total pages in the PDF.
        page_text: The text Docling produced, keyed by page number.
        docling_error: Set when Docling failed; every page is then treated as weak.

    Returns:
        The fallback report, and the rescued pages keyed by page number. Only
        pages where PaddleOCR did better than Docling are included.
    """
    report = FallbackReport(
        enabled=settings.enable_ocr_fallback,
        available=is_available(),
        threshold=settings.fallback_min_chars_per_page,
        accent_threshold=settings.fallback_min_accent_rate,
        docling_error=docling_error,
    )

    reasons = _weak_pages(page_count, page_text, docling_error)
    weak_pages = sorted(reasons)
    report.weak_pages = weak_pages
    if not weak_pages:
        return report, {}

    report.reason = "docling_error" if docling_error else (
        "poor_quality" if all(r == "poor_quality" for r in reasons.values()) else "low_yield"
    )

    if not settings.enable_ocr_fallback:
        report.skipped_reason = "OCR fallback disabled (WSLA_ENABLE_OCR_FALLBACK=false)"
        return report, {}
    if not report.available:
        report.skipped_reason = "paddleocr is not installed"
        _log.warning(
            "%s has %d weak pages but PaddleOCR is not installed",
            pdf_path.name, len(weak_pages),
        )
        return report, {}

    limit = settings.paddle_max_pages
    to_attempt = weak_pages[:limit] if limit > 0 else weak_pages
    report.pages_skipped = weak_pages[len(to_attempt):]

    start = time.perf_counter()
    try:
        get_engine()
    except FallbackUnavailableError as exc:
        report.available = False
        report.skipped_reason = str(exc)
        report.seconds = time.perf_counter() - start
        return report, {}

    report.triggered = True
    rescued: dict[int, RescuedPage] = {}
    pdf_doc = pdfium.PdfDocument(pdf_path)
    try:
        for page_no in to_attempt:
            page_result, page_rescue = _ocr_page(
                pdf_doc, page_no, page_text.get(page_no, ""), reasons[page_no]
            )
            report.pages.append(page_result)
            if page_rescue.blocks:
                rescued[page_no] = page_rescue
    finally:
        pdf_doc.close()

    report.pages_attempted = len(report.pages)
    report.pages_recovered = len(rescued)
    report.pages_replaced = sum(1 for page in rescued.values() if page.replace)
    report.chars_recovered = sum(page.char_count for page in report.pages if page.recovered)
    report.seconds = time.perf_counter() - start

    _log.info(
        "PaddleOCR fallback for %s: %d weak (%s), %d attempted, %d recovered, "
        "%d replaced (%d chars) in %.2fs",
        pdf_path.name,
        len(weak_pages),
        ", ".join(sorted({r for r in reasons.values()})),
        report.pages_attempted,
        report.pages_recovered,
        report.pages_replaced,
        report.chars_recovered,
        report.seconds,
    )
    return report, rescued

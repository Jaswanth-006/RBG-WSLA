"""Configuration settings for the WSLA Document Parsing API."""

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings

# The repository root (src/wsla/config.py -> parents[2]). The default data folders
# are anchored here rather than beside this file, so they stay at the top of the
# repo instead of ending up inside the package. Docker overrides both.
_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Directory where uploaded PDFs are stored, keyed by doc_id
    upload_dir: Path = _REPO_ROOT / "uploads"

    # Directory where parsed outputs (markdown, images) are stored
    output_dir: Path = _REPO_ROOT / "output"

    # Maximum file size in bytes (default 100 MB)
    max_file_size: int = 100 * 1024 * 1024

    # Minimum average characters per page to consider a page "editable"
    # Pages below this threshold are classified as scanned/image-based
    editable_char_threshold: int = 50

    # Image resolution scale for exported pictures (1.0 = 72 DPI)
    image_resolution_scale: float = 2.0

    # Runtime device for Docling and EasyOCR: auto | gpu | cpu
    device: str = "auto"

    # PaddleOCR fallback device. Kept on CPU: PaddlePaddle 3.x GPU wheels are
    # served only from paddlepaddle.org.cn, which our network cannot reach
    # (PyPI stops at 2.6.2, which PaddleOCR 3.7 does not support).
    paddle_device: str = "cpu"

    # ── PaddleOCR fallback ───────────────────────────────────────────────
    # After Docling runs, pages yielding fewer alphanumeric characters than
    # this are re-read with PaddleOCR. Every page is re-read if Docling fails.
    enable_ocr_fallback: bool = True
    fallback_min_chars_per_page: int = 50

    # Quality trigger: a page can be full of text and still be wrong (a garbled
    # OCR layer). Pages whose Spanish reads as accent-starved are re-OCR'd even
    # though they pass the character count. See pipeline/quality.py for the
    # measured thresholds. Raise the rate to send more pages to PaddleOCR.
    enable_quality_trigger: bool = True
    fallback_min_accent_rate: float = 0.008
    quality_min_letters: int = 60

    # Reading order: treat a vertical gap this wide (as a fraction of page width)
    # as a column separator, so side-by-side columns stop interleaving.
    enable_column_split: bool = True
    column_gutter_ratio: float = 0.06

    # Render resolution for fallback OCR (Docling renders pictures at 144 DPI)
    fallback_dpi: int = 200

    # Maximum pages rescued per document (0 = no limit)
    paddle_max_pages: int = 0

    # Recognised lines below this confidence are dropped
    paddle_min_confidence: float = 0.5

    # PP-OCRv5 models. The Latin recogniser covers Spanish (á é í ó ú ñ ü).
    paddle_det_model: str = "PP-OCRv5_mobile_det"
    paddle_rec_model: str = "latin_PP-OCRv5_mobile_rec"

    # Directory of pre-downloaded models laid out as <dir>/<model_name>/.
    # When unset, PaddleOCR resolves models from its own cache.
    paddle_model_dir: Path | None = None

    # Correct rotated/upside-down text lines (uses the textline orientation model)
    paddle_use_textline_orientation: bool = True

    paddle_cpu_threads: int = 4
    paddle_enable_mkldnn: bool = True

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        normalized = (value or "auto").strip().lower()
        if normalized not in {"auto", "gpu", "cpu"}:
            raise ValueError("Unsupported device setting. Use one of: auto, gpu, cpu.")
        return normalized

    model_config = {"env_prefix": "WSLA_"}


settings = Settings()

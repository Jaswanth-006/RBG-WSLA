"""Configuration settings for the WSLA Document Parsing API."""

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings

# Resolve the wsla_service package directory so data directories live inside it
_PACKAGE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Device options:
        - auto: Use GPU automatically when available; otherwise CPU.
        - gpu: Require CUDA/GPU and fail clearly if unavailable.
        - cpu: Force CPU even if GPU is present.
    """

    # Directory where uploaded PDFs are stored, keyed by doc_id
    upload_dir: Path = _PACKAGE_DIR / "uploads"

    # Directory where parsed outputs (markdown, images) are stored
    output_dir: Path = _PACKAGE_DIR / "output"

    # Maximum file size in bytes (default 100 MB)
    max_file_size: int = 100 * 1024 * 1024

    # Minimum average characters per page to consider a page "editable"
    # Pages below this threshold are classified as scanned/image-based
    editable_char_threshold: int = 50

    # Image resolution scale for exported pictures (1.0 = 72 DPI)
    image_resolution_scale: float = 2.0

    # Runtime device selection: auto | gpu | cpu
    # Environment variable: WSLA_DEVICE
    device: str = "auto"

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        normalized = (value or "auto").strip().lower()
        if normalized not in {"auto", "gpu", "cpu"}:
            raise ValueError(
                "Unsupported device setting. Use one of: auto, gpu, cpu."
            )
        return normalized

    model_config = {"env_prefix": "WSLA_"}


settings = Settings()


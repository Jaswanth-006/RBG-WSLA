"""Configuration settings for the WSLA Document Parsing API."""

from pathlib import Path

from pydantic_settings import BaseSettings

# Resolve the wsla_service package directory so data directories live inside it
_PACKAGE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

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

    model_config = {"env_prefix": "WSLA_"}


settings = Settings()


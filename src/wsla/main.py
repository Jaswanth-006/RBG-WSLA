"""WSLA Document Parsing API — application entry point.

Run with ``python -m wsla.main`` (``PORT`` must be set), or point any ASGI server
at ``wsla.main:app``.
"""

import logging
import os
import sys

from dotenv import load_dotenv

# Load .env before anything reads settings: wsla.config builds its Settings at
# import time, so WSLA_* values from .env must already be in the environment.
load_dotenv()

from fastapi import FastAPI  # noqa: E402

from wsla.api.reporting import print_runtime_summary  # noqa: E402
from wsla.api.routes import router  # noqa: E402
from wsla.config import settings  # noqa: E402
from wsla.storage import documents  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
_log = logging.getLogger(__name__)

app = FastAPI(
    title="WSLA Document Parsing API",
    description=(
        "Upload WSLA/PDF documents, detect whether they are editable or scanned, "
        "and parse them into structured Markdown with metadata (tables, images, "
        "paragraphs, sections) using the Docling pipeline, with PaddleOCR "
        "re-reading any page Docling could not."
    ),
    version="1.0.0",
)
app.include_router(router)


@app.on_event("startup")
def _ensure_directories() -> None:
    """Create the data directories and the document store on startup."""
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    documents.init_db()
    _log.info("Upload dir: %s", settings.upload_dir.resolve())
    _log.info("Output dir: %s", settings.output_dir.resolve())
    print_runtime_summary()


def main() -> None:
    """Start the API on the port given by the PORT environment variable."""
    import uvicorn

    port = os.getenv("PORT")
    if not port:
        print("ERROR: PORT environment variable is not set in .env file")
        sys.exit(1)

    try:
        port_int = int(port)
    except ValueError:
        print(f"ERROR: PORT value '{port}' is not a valid integer")
        sys.exit(1)

    print(f"Starting WSLA Document Parsing API on port {port_int}")
    uvicorn.run(app, host="0.0.0.0", port=port_int, log_level="info")


if __name__ == "__main__":
    main()

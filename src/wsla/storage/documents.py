"""SQLite record of every document the service has handled.

Without this the service is filesystem-only: a ``doc_id`` is just a directory
name, and everything a parse discovered — document type, engine used, how many
pages the fallback rescued — is lost the moment the caller drops the response.

The database lives beside the parsed output, so it shares the same volume and
survives container restarts.
"""

import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from wsla.config import settings

_log = logging.getLogger(__name__)
_lock = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id         TEXT PRIMARY KEY,
    filename       TEXT NOT NULL,
    file_size      INTEGER NOT NULL DEFAULT 0,
    uploaded_at    REAL NOT NULL,
    pdf_type       TEXT,
    page_count     INTEGER,
    parse_engine   TEXT,
    markdown_file  TEXT,
    fallback_pages INTEGER,
    parsed_at      REAL
);
"""


def _db_path() -> Path:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    return settings.output_dir / "wsla.db"


@contextmanager
def _session():
    """Open a connection, commit on success, and always close it.

    ``sqlite3.Connection`` as a context manager commits but does not close, which
    leaks a file handle per call - on Windows that keeps the database file
    locked, so the close is explicit here.
    """
    conn = sqlite3.connect(_db_path(), timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_db() -> None:
    """Create the table if it does not exist. Safe to call on every startup."""
    with _lock, _session() as conn:
        conn.executescript(_SCHEMA)
    _log.info("Document store ready at %s", _db_path())


def record_upload(doc_id: str, filename: str, file_size: int) -> None:
    """Insert a freshly uploaded document, or reset an id that is being reused."""
    with _lock, _session() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO documents "
            "(doc_id, filename, file_size, uploaded_at) VALUES (?, ?, ?, ?)",
            (doc_id, filename, file_size, time.time()),
        )


def record_parse(
    doc_id: str,
    *,
    pdf_type: str,
    page_count: int,
    parse_engine: str,
    markdown_file: str,
    fallback_pages: int,
) -> None:
    """Store what the parse discovered, so /status never has to re-read the PDF."""
    with _lock, _session() as conn:
        conn.execute(
            "UPDATE documents SET pdf_type = ?, page_count = ?, parse_engine = ?, "
            "markdown_file = ?, fallback_pages = ?, parsed_at = ? WHERE doc_id = ?",
            (pdf_type, page_count, parse_engine, markdown_file, fallback_pages,
             time.time(), doc_id),
        )


def get_document(doc_id: str) -> dict | None:
    """Return the stored record, or None when the id is unknown."""
    with _lock, _session() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
    return dict(row) if row else None


def forget(doc_id: str) -> None:
    """Drop a record, used when /process cleans up after itself."""
    with _lock, _session() as conn:
        conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

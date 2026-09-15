"""Pydantic request/response schemas for the WSLA API."""

from enum import Enum

from pydantic import BaseModel, Field


class PdfType(str, Enum):
    """Classification of PDF content type."""

    EDITABLE = "editable"
    SCANNED = "scanned"
    MIXED = "mixed"  # some pages editable, some scanned


class ParseEngine(str, Enum):
    """Which engine(s) produced the parsed output."""

    DOCLING = "docling"
    PADDLEOCR = "paddleocr"  # Docling failed; PaddleOCR carried the document
    DOCLING_PADDLEOCR = "docling+paddleocr"  # Docling output plus rescued pages


class PageClassification(BaseModel):
    """Per-page editable/scanned classification."""

    page_no: int
    char_count: int
    classification: PdfType


class ProcessingMetrics(BaseModel):
    """Runtime and throughput metrics for a document parse."""

    requested_device: str = ""
    resolved_device: str = ""
    gpu_available: bool = False
    gpu_name: str | None = None
    cuda_version: str | None = None
    gpu_count: int = 0

    upload_seconds: float = 0.0
    detection_seconds: float = 0.0
    parsing_seconds: float = 0.0
    metadata_seconds: float = 0.0
    fallback_seconds: float = 0.0
    zip_seconds: float = 0.0
    total_seconds: float = 0.0

    page_count: int = 0
    fallback_pages: int = 0

    parsing_pages_per_second: float = 0.0
    parsing_pages_per_minute: float = 0.0
    overall_pages_per_second: float = 0.0
    overall_pages_per_minute: float = 0.0


class UploadResponse(BaseModel):
    """Response returned after uploading a PDF."""

    doc_id: str
    filename: str
    file_size: int
    message: str = "File uploaded successfully"


class TableMetadata(BaseModel):
    """Metadata for a single detected table."""

    index: int
    page_no: int | None = None
    num_rows: int = 0
    num_cols: int = 0
    csv_data: str = ""
    html_data: str = ""
    markdown_data: str = ""


class ImageMetadata(BaseModel):
    """Metadata for a single detected image/picture."""

    index: int
    page_no: int | None = None
    image_path: str = ""


class ParagraphMetadata(BaseModel):
    """Metadata for a single text paragraph."""

    index: int
    page_no: int | None = None
    label: str = "paragraph"
    text: str = ""


class SectionMetadata(BaseModel):
    """Metadata for a section header."""

    index: int
    page_no: int | None = None
    level: int = 1
    text: str = ""


class DocumentMetadata(BaseModel):
    """Aggregated metadata extracted from the document."""

    tables: list[TableMetadata] = Field(default_factory=list)
    images: list[ImageMetadata] = Field(default_factory=list)
    paragraphs: list[ParagraphMetadata] = Field(default_factory=list)
    sections: list[SectionMetadata] = Field(default_factory=list)


class PageOcrResult(BaseModel):
    """Outcome of re-reading one weak page with PaddleOCR."""

    page_no: int
    reason: str  # "low_yield", "poor_quality" or "docling_error"
    docling_chars: int = 0
    char_count: int = 0
    line_count: int = 0
    mean_confidence: float = 0.0
    recovered: bool = False  # PaddleOCR read more text than Docling
    replaced: bool = False  # Docling's text for this page was dropped as wrong
    seconds: float = 0.0
    error: str | None = None


class FallbackReport(BaseModel):
    """What the PaddleOCR fallback did for a document."""

    enabled: bool = False
    available: bool = False
    triggered: bool = False
    reason: str | None = None  # "low_yield" or "docling_error"
    skipped_reason: str | None = None
    docling_error: str | None = None
    threshold: int = 0
    accent_threshold: float = 0.0
    weak_pages: list[int] = Field(default_factory=list)
    pages_skipped: list[int] = Field(default_factory=list)  # beyond paddle_max_pages
    pages_attempted: int = 0
    pages_recovered: int = 0
    pages_replaced: int = 0
    chars_recovered: int = 0
    seconds: float = 0.0
    pages: list[PageOcrResult] = Field(default_factory=list)


class ParseResponse(BaseModel):
    """Full response from the parse endpoint."""

    doc_id: str
    filename: str
    pdf_type: PdfType
    page_classifications: list[PageClassification] = Field(default_factory=list)
    page_count: int = 0
    parse_engine: ParseEngine = ParseEngine.DOCLING
    markdown_file: str = ""
    markdown_text: str = ""
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    fallback: FallbackReport = Field(default_factory=FallbackReport)


class StatusResponse(BaseModel):
    """Response from the status endpoint."""

    doc_id: str
    filename: str
    exists: bool
    parsed: bool
    pdf_type: PdfType | None = None
    parse_engine: ParseEngine | None = None
    markdown_file: str | None = None

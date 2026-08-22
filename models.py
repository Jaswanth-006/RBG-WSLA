"""Pydantic request/response schemas for the WSLA API."""

from enum import Enum

from pydantic import BaseModel, Field


class PdfType(str, Enum):
    """Classification of PDF content type."""

    EDITABLE = "editable"
    SCANNED = "scanned"
    MIXED = "mixed"  # some pages editable, some scanned


class PageClassification(BaseModel):
    """Per-page editable/scanned classification."""

    page_no: int
    char_count: int
    classification: PdfType


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


class ParseResponse(BaseModel):
    """Full response from the parse endpoint."""

    doc_id: str
    filename: str
    pdf_type: PdfType
    page_classifications: list[PageClassification] = Field(default_factory=list)
    page_count: int = 0
    markdown_file: str = ""
    markdown_text: str = ""
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)


class StatusResponse(BaseModel):
    """Response from the status endpoint."""

    doc_id: str
    filename: str
    exists: bool
    parsed: bool
    pdf_type: PdfType | None = None
    markdown_file: str | None = None

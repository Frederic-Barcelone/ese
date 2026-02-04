# corpus_metadata/A_core/A08_document_metadata_models.py
"""
Pydantic domain models for document metadata extraction.

This module defines data structures for extracting and organizing document-level metadata
from PDFs. It handles file system metadata, PDF properties (author, creator, DOI), document
classification (protocol, CSR, publication), LLM-generated descriptions, and intelligent
date extraction with a priority-based fallback chain (filename -> content -> PDF -> filesystem).

Key Components:
    - DocumentMetadata: Complete metadata container for a document
    - FileMetadata: File system information (size, dates, permissions)
    - PDFMetadata: PDF properties (title, author, DOI, page count, encryption status)
    - DocumentClassification: Document type with confidence (protocol, article, CSR)
    - DocumentDescription: LLM-generated title and descriptions
    - DateExtractionResult: Extracted dates with source tracking and fallback chain
    - ExtractedDate: Individual date with source type and confidence
    - DocumentMetadataExport: Simplified export format

Example:
    >>> from A_core.A08_document_metadata_models import DocumentMetadata, PDFMetadata
    >>> pdf_meta = PDFMetadata(
    ...     title="Phase 3 Study Protocol",
    ...     author="Sponsor Inc.",
    ...     page_count=120,
    ...     doi="10.1016/j.example.2024.001",
    ... )

Dependencies:
    - None (self-contained module with no A_core dependencies)
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


# -------------------------
# Document Metadata Enums
# -------------------------


class DocumentMetadataFieldType(str, Enum):
    """Type of document metadata extracted."""

    FILE_METADATA = "FILE_METADATA"
    PDF_METADATA = "PDF_METADATA"
    CLASSIFICATION = "CLASSIFICATION"
    DESCRIPTION = "DESCRIPTION"
    DATE_EXTRACTION = "DATE_EXTRACTION"


class DocumentMetadataGeneratorType(str, Enum):
    """Tracks which strategy produced the metadata."""

    FILE_SYSTEM = "gen:file_system"
    PDF_PARSER = "gen:pdf_parser"
    PATTERN_MATCH = "gen:pattern_match"
    LLM_EXTRACTION = "gen:llm_extraction"


class DateSourceType(str, Enum):
    """Source of extracted date, in fallback order."""

    FILENAME = "filename"  # Priority 1: date from filename
    CONTENT = "content"  # Priority 2: date from document content
    PDF_METADATA = "pdf_metadata"  # Priority 3: PDF creation/modification date
    FILE_SYSTEM = "file_system"  # Priority 4: file system dates


# -------------------------
# File Metadata
# -------------------------


class FileMetadata(BaseModel):
    """File system metadata."""

    file_path: str
    filename: str
    extension: str
    size_bytes: int
    size_human: str  # e.g., "2.3 MB"
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    accessed_at: Optional[datetime] = None
    permissions: Optional[str] = None  # e.g., "rw-r--r--"

    model_config = ConfigDict(extra="forbid")


# -------------------------
# PDF Metadata
# -------------------------


class PDFMetadata(BaseModel):
    """PDF-specific metadata from document properties."""

    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    keywords: Optional[List[str]] = None
    creator: Optional[str] = None  # Application that created the PDF
    producer: Optional[str] = None  # PDF producer (e.g., "Adobe PDF Library")
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None
    page_count: Optional[int] = None
    pdf_version: Optional[str] = None
    is_encrypted: bool = False
    is_tagged: bool = False  # Accessibility tagging
    has_form_fields: bool = False
    has_annotations: bool = False
    doi: Optional[str] = None  # DOI identifier (e.g., "10.1016/S0140-6736(25)01148-1")
    doi_url: Optional[str] = None  # Full DOI URL (e.g., "https://doi.org/10.1016/...")

    model_config = ConfigDict(extra="forbid")


# -------------------------
# Document Classification
# -------------------------


class DocumentType(BaseModel):
    """Classification result for document type."""

    code: str  # e.g., "PRO", "PUB", "CSR"
    name: str  # e.g., "Protocol & Synopsis"
    group: str  # e.g., "Essential Trial Documents"
    description: Optional[str] = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    matched_patterns: List[str] = Field(default_factory=list)
    matched_aliases: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class DocumentClassification(BaseModel):
    """Document classification with primary and alternative types."""

    primary_type: DocumentType
    alternative_types: List[DocumentType] = Field(default_factory=list)
    llm_reasoning: Optional[str] = None  # LLM explanation for classification
    classification_method: str = "llm"  # "llm", "pattern", "hybrid"

    model_config = ConfigDict(extra="forbid")


# -------------------------
# Document Description
# -------------------------


class DocumentDescription(BaseModel):
    """LLM-generated document descriptions."""

    title: str  # Extracted or generated title
    title_source: str = "llm"  # "pdf_metadata", "content", "llm"
    short_description: str  # 1-2 sentences
    long_description: str  # Detailed paragraph
    key_topics: List[str] = Field(default_factory=list)  # Main topics/themes
    language: str = "en"  # ISO language code
    llm_model: Optional[str] = None  # Model used for generation

    model_config = ConfigDict(extra="forbid")


# -------------------------
# Date Extraction
# -------------------------


class ExtractedDate(BaseModel):
    """
    Date extracted from document with source tracking.

    Implements fallback chain:
    1. Filename (e.g., "2024_08_protocol.pdf" -> 2024-08)
    2. Content (e.g., "Date: January 15, 2024" in document)
    3. PDF metadata (creation/modification date)
    4. File system (file creation/modification date)
    """

    date: datetime
    source: DateSourceType
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    original_text: Optional[str] = None  # Raw text from which date was extracted
    date_type: str = "document_date"  # "document_date", "publication_date", "version_date"
    is_approximate: bool = False  # True if only year/month known

    model_config = ConfigDict(extra="forbid")


class DateExtractionResult(BaseModel):
    """All dates extracted from document with fallback chain."""

    primary_date: Optional[ExtractedDate] = None  # Best date after fallback
    filename_date: Optional[ExtractedDate] = None
    content_dates: List[ExtractedDate] = Field(default_factory=list)
    pdf_creation_date: Optional[ExtractedDate] = None
    pdf_modification_date: Optional[ExtractedDate] = None
    file_system_created: Optional[ExtractedDate] = None
    file_system_modified: Optional[ExtractedDate] = None
    fallback_chain_used: List[DateSourceType] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


# -------------------------
# Provenance
# -------------------------


class DocumentMetadataProvenance(BaseModel):
    """Provenance metadata for document metadata extraction."""

    pipeline_version: str
    run_id: str
    doc_fingerprint: str
    generator_name: DocumentMetadataGeneratorType
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(frozen=True, extra="forbid")


# -------------------------
# Main Document Metadata Container
# -------------------------


class DocumentMetadata(BaseModel):
    """
    Complete document metadata extraction result.

    Contains all metadata extracted from a document:
    - File system metadata
    - PDF metadata
    - Classification
    - Descriptions
    - Dates
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    doc_id: str
    doc_filename: str

    # Extracted metadata
    file_metadata: Optional[FileMetadata] = None
    pdf_metadata: Optional[PDFMetadata] = None
    classification: Optional[DocumentClassification] = None
    description: Optional[DocumentDescription] = None
    date_extraction: Optional[DateExtractionResult] = None

    # Provenance
    provenance: Optional[DocumentMetadataProvenance] = None
    extraction_timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(extra="forbid")


# -------------------------
# Export Models
# -------------------------


class TopEntity(BaseModel):
    """Top mentioned entity for document metadata export."""

    name: str
    mention_count: int
    entity_type: str  # "disease", "drug", "gene"

    model_config = ConfigDict(extra="forbid")


class DocumentMetadataExport(BaseModel):
    """Simplified export format for document metadata."""

    doc_id: str
    doc_filename: str

    # Top mentioned entities
    top_entities: Optional[List[TopEntity]] = None

    # File info
    file_size_bytes: Optional[int] = None
    file_size_human: Optional[str] = None
    file_extension: Optional[str] = None

    # PDF info
    pdf_title: Optional[str] = None
    pdf_author: Optional[str] = None
    pdf_page_count: Optional[int] = None
    pdf_creation_date: Optional[str] = None

    # Identifiers
    doi: Optional[str] = None  # DOI identifier
    doi_url: Optional[str] = None  # Full DOI URL

    # Classification
    document_type_code: Optional[str] = None
    document_type_name: Optional[str] = None
    document_type_group: Optional[str] = None
    classification_confidence: Optional[float] = None

    # Description
    title: Optional[str] = None
    short_description: Optional[str] = None
    long_description: Optional[str] = None

    # Date
    document_date: Optional[str] = None
    document_date_source: Optional[str] = None

    extraction_timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(extra="forbid")

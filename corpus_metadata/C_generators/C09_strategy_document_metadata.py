# corpus_metadata/corpus_metadata/C_generators/C09_strategy_document_metadata.py
"""
Document metadata extraction strategy.

Extracts comprehensive document metadata:
1. FILE METADATA - Size, dates, permissions from file system
2. PDF METADATA - Author, creator, creation date from PDF properties
3. CLASSIFICATION - Document type classification using LLM
4. DESCRIPTIONS - Title, short/long descriptions using LLM
5. DATE EXTRACTION - Date with fallback chain: filename → content → PDF → file system

Uses pattern matching for dates and LLM for classification/description.
"""

from __future__ import annotations

import json
import re
import stat
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from A_core.A03_provenance import generate_run_id, get_git_revision_hash
from A_core.A08_document_metadata_models import (
    DateExtractionResult,
    DateSourceType,
    DocumentClassification,
    DocumentDescription,
    DocumentMetadata,
    DocumentMetadataGeneratorType,
    DocumentMetadataProvenance,
    DocumentType,
    ExtractedDate,
    FileMetadata,
    PDFMetadata,
)
from B_parsing.B02_doc_graph import DocumentGraph

# PDF parsing - optional
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    fitz = None  # type: ignore
    PYMUPDF_AVAILABLE = False


# =============================================================================
# DOCUMENT TYPE LOADER
# =============================================================================


class DocumentTypeRegistry:
    """
    Loads and manages document type definitions from JSON.

    Document types are organized in groups with:
    - code: Short code (e.g., "PRO", "PUB")
    - name: Full name
    - patterns: Regex patterns for matching
    - aliases: Alternative names
    - confidence_boosters: Patterns for boosting confidence
    """

    def __init__(self, document_types_path: Optional[str] = None):
        self.types: List[Dict[str, Any]] = []
        self.groups: Dict[str, List[Dict[str, Any]]] = {}
        self.code_to_type: Dict[str, Dict[str, Any]] = {}

        if document_types_path:
            self.load_from_file(document_types_path)

    def load_from_file(self, path: str) -> None:
        """Load document types from JSON file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for group in data:
                group_name = group.get("group", "Unknown")
                types = group.get("types", [])

                self.groups[group_name] = types

                for t in types:
                    t["group"] = group_name  # Add group to each type
                    self.types.append(t)
                    self.code_to_type[t["code"]] = t
        except Exception as e:
            print(f"[WARN] Failed to load document types from {path}: {e}")

    def get_type_by_code(self, code: str) -> Optional[Dict[str, Any]]:
        """Get document type by code."""
        return self.code_to_type.get(code.upper())

    def get_all_types_summary(self) -> str:
        """Get summary of all types for LLM prompt."""
        lines = []
        for group_name, types in self.groups.items():
            lines.append(f"\n## {group_name}")
            for t in types:
                lines.append(f"- {t['code']}: {t['name']} - {t.get('desc', '')}")
        return "\n".join(lines)

    def get_types_for_prompt(self) -> str:
        """Get formatted document types for LLM classification prompt."""
        lines = []
        for t in self.types:
            code = t["code"]
            name = t["name"]
            desc = t.get("desc", "")
            aliases = ", ".join(t.get("aliases", [])[:3])  # Limit aliases
            lines.append(f"{code} | {name} | {desc} | Aliases: {aliases}")
        return "\n".join(lines)


# =============================================================================
# DATE EXTRACTION PATTERNS
# =============================================================================


class DateExtractor:
    """
    Extracts dates from filenames, content, and metadata.

    Implements fallback chain:
    1. Filename (e.g., "2024_08_protocol.pdf")
    2. Content (e.g., "Date: January 15, 2024")
    3. PDF metadata
    4. File system dates
    """

    # Filename date patterns (most specific first)
    FILENAME_PATTERNS = [
        # YYYY_MM or YYYY-MM or YYYYMM at start
        (r"^(\d{4})[-_]?(\d{2})(?:[-_]|\b)", "%Y-%m"),
        # YYYY at start
        (r"^(\d{4})[-_\s]", "%Y"),
        # YYYY_MM_DD or YYYY-MM-DD
        (r"(\d{4})[-_](\d{2})[-_](\d{2})", "%Y-%m-%d"),
        # DD_MM_YYYY or DD-MM-YYYY
        (r"(\d{2})[-_](\d{2})[-_](\d{4})", "%d-%m-%Y"),
        # Month YYYY
        (r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-_\s]+(\d{4})", "%b %Y"),
        # YYYY only (less specific)
        (r"\b(20\d{2})\b", "%Y"),
    ]

    # Content date patterns
    CONTENT_PATTERNS = [
        # "Date: January 15, 2024" or "Date: 15 January 2024"
        (r"(?:date|dated|effective|published|version)[\s:]+(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})", "dmy"),
        (r"(?:date|dated|effective|published|version)[\s:]+(\w+)\s+(\d{1,2}),?\s+(\d{4})", "mdy"),
        # ISO format in content
        (r"(?:date|dated|version)[\s:]+(\d{4})-(\d{2})-(\d{2})", "iso"),
        # "Version 2.0, 15 March 2024"
        (r"version\s+[\d.]+[,\s]+(\d{1,2})\s+(\w+)\s+(\d{4})", "dmy"),
        # Standalone month year
        (r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})\b", "my"),
    ]

    MONTH_MAP = {
        "january": 1, "jan": 1,
        "february": 2, "feb": 2,
        "march": 3, "mar": 3,
        "april": 4, "apr": 4,
        "may": 5,
        "june": 6, "jun": 6,
        "july": 7, "jul": 7,
        "august": 8, "aug": 8,
        "september": 9, "sep": 9, "sept": 9,
        "october": 10, "oct": 10,
        "november": 11, "nov": 11,
        "december": 12, "dec": 12,
    }

    def extract_from_filename(self, filename: str) -> Optional[ExtractedDate]:
        """Extract date from filename."""
        filename_lower = filename.lower()

        for pattern, fmt in self.FILENAME_PATTERNS:
            match = re.search(pattern, filename_lower, re.IGNORECASE)
            if match:
                try:
                    groups = match.groups()

                    if fmt == "%Y-%m-%d" and len(groups) >= 3:
                        year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                        dt = datetime(year, month, day)
                    elif fmt == "%d-%m-%Y" and len(groups) >= 3:
                        day, month, year = int(groups[0]), int(groups[1]), int(groups[2])
                        dt = datetime(year, month, day)
                    elif fmt == "%Y-%m" and len(groups) >= 2:
                        year, month = int(groups[0]), int(groups[1])
                        if 1 <= month <= 12:
                            dt = datetime(year, month, 1)
                        else:
                            continue
                    elif fmt == "%b %Y" and len(groups) >= 2:
                        month_str = groups[0].lower()[:3]
                        month = self.MONTH_MAP.get(month_str)
                        year = int(groups[1])
                        if month:
                            dt = datetime(year, month, 1)
                        else:
                            continue
                    elif fmt == "%Y" and len(groups) >= 1:
                        year = int(groups[0])
                        if 2000 <= year <= 2030:
                            dt = datetime(year, 1, 1)
                        else:
                            continue
                    else:
                        continue

                    # Validate reasonable date range
                    if 2000 <= dt.year <= 2030:
                        is_approx = fmt in ["%Y", "%Y-%m", "%b %Y"]
                        return ExtractedDate(
                            date=dt,
                            source=DateSourceType.FILENAME,
                            confidence=0.9 if not is_approx else 0.7,
                            original_text=match.group(0),
                            date_type="document_date",
                            is_approximate=is_approx,
                        )
                except (ValueError, IndexError):
                    continue

        return None

    def extract_from_content(self, text: str, limit: int = 5000) -> List[ExtractedDate]:
        """Extract dates from document content (first N chars)."""
        dates = []
        text_lower = text[:limit].lower()

        for pattern, fmt_type in self.CONTENT_PATTERNS:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                try:
                    groups = match.groups()

                    if fmt_type == "iso" and len(groups) >= 3:
                        year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                        dt = datetime(year, month, day)
                        is_approx = False
                    elif fmt_type == "dmy" and len(groups) >= 3:
                        day = int(groups[0])
                        month_str = groups[1].lower()[:3]
                        month = self.MONTH_MAP.get(month_str)
                        year = int(groups[2])
                        if month:
                            dt = datetime(year, month, day)
                            is_approx = False
                        else:
                            continue
                    elif fmt_type == "mdy" and len(groups) >= 3:
                        month_str = groups[0].lower()[:3]
                        month = self.MONTH_MAP.get(month_str)
                        day = int(groups[1])
                        year = int(groups[2])
                        if month:
                            dt = datetime(year, month, day)
                            is_approx = False
                        else:
                            continue
                    elif fmt_type == "my" and len(groups) >= 2:
                        month_str = groups[0].lower()[:3]
                        month = self.MONTH_MAP.get(month_str)
                        year = int(groups[1])
                        if month:
                            dt = datetime(year, month, 1)
                            is_approx = True
                        else:
                            continue
                    else:
                        continue

                    if 2000 <= dt.year <= 2030:
                        dates.append(ExtractedDate(
                            date=dt,
                            source=DateSourceType.CONTENT,
                            confidence=0.85 if not is_approx else 0.65,
                            original_text=match.group(0),
                            date_type="document_date",
                            is_approximate=is_approx,
                        ))
                except (ValueError, IndexError):
                    continue

        return dates

    def create_from_datetime(
        self,
        dt: Optional[datetime],
        source: DateSourceType,
        confidence: float = 0.5,
    ) -> Optional[ExtractedDate]:
        """Create ExtractedDate from a datetime object."""
        if dt is None:
            return None
        return ExtractedDate(
            date=dt,
            source=source,
            confidence=confidence,
            original_text=None,
            date_type="document_date",
            is_approximate=False,
        )


# =============================================================================
# DOI EXTRACTION
# =============================================================================


class DOIExtractor:
    """
    Extracts DOI (Digital Object Identifier) from document content and PDF annotations.

    Supports formats:
    - Bare DOI: 10.1016/S0140-6736(25)01148-1
    - URL format: https://doi.org/10.1016/S0140-6736(25)01148-1
    - With prefix: doi:10.1016/S0140-6736(25)01148-1
    """

    # DOI patterns - order matters (most specific first)
    # DOIs can contain parentheses, hyphens, dots, etc. so we use a permissive pattern
    # and clean up trailing punctuation afterwards
    DOI_PATTERNS = [
        # Full URL format: https://doi.org/10.xxxx/...
        re.compile(r"https?://(?:dx\.)?doi\.org/(10\.\d{4,9}/\S+)", re.IGNORECASE),
        # With doi: prefix
        re.compile(r"doi[:\s]+(10\.\d{4,9}/\S+)", re.IGNORECASE),
        # Bare DOI format (10.xxxx/...)
        re.compile(r"\b(10\.\d{4,9}/\S+)", re.IGNORECASE),
    ]

    @classmethod
    def extract_from_text(cls, text: str) -> tuple[Optional[str], Optional[str]]:
        """
        Extract DOI from text content.

        Returns:
            Tuple of (doi, doi_url) where doi is the bare identifier
            and doi_url is the full https://doi.org/... URL
        """
        if not text:
            return None, None

        # Normalize whitespace (DOIs sometimes have line breaks)
        text = re.sub(r"\s+", " ", text)

        for pattern in cls.DOI_PATTERNS:
            match = pattern.search(text)
            if match:
                doi = match.group(1)
                # Clean up trailing punctuation and common delimiters
                doi = doi.rstrip(".,;:\"'<>]}")
                # Remove trailing HTML/XML if present
                doi = re.sub(r"<[^>]+>.*$", "", doi)
                # Validate DOI format
                if cls._is_valid_doi(doi):
                    doi_url = f"https://doi.org/{doi}"
                    return doi, doi_url

        return None, None

    @classmethod
    def extract_from_pdf_annotations(cls, doc) -> tuple[Optional[str], Optional[str]]:
        """
        Extract DOI from PDF link annotations.

        Args:
            doc: PyMuPDF document object

        Returns:
            Tuple of (doi, doi_url)
        """
        if doc is None:
            return None, None

        try:
            for page in doc:
                for link in page.get_links():
                    uri = link.get("uri", "")
                    if uri and "doi.org" in uri:
                        # Extract DOI from URL
                        match = re.search(r"doi\.org/(10\.\d{4,9}/[^\s\]\)>\",]+)", uri)
                        if match:
                            doi = match.group(1).rstrip(".,;:")
                            if cls._is_valid_doi(doi):
                                return doi, uri
        except Exception:
            pass

        return None, None

    @staticmethod
    def _is_valid_doi(doi: str) -> bool:
        """Validate DOI format."""
        if not doi:
            return False
        # DOI must start with 10. and have a suffix after /
        if not doi.startswith("10."):
            return False
        if "/" not in doi:
            return False
        # Suffix must be at least 1 character
        parts = doi.split("/", 1)
        if len(parts) != 2 or not parts[1]:
            return False
        return True


# =============================================================================
# MAIN STRATEGY
# =============================================================================


class DocumentMetadataStrategy:
    """
    Document metadata extraction strategy.

    Extracts:
    1. File metadata (size, dates, permissions)
    2. PDF metadata (author, creator, creation date)
    3. Document classification using LLM
    4. Document descriptions using LLM
    5. Dates with fallback chain
    """

    def __init__(
        self,
        document_types_path: Optional[str] = None,
        llm_client: Optional[Any] = None,
        llm_model: str = "claude-sonnet-4-20250514",
        run_id: Optional[str] = None,
        pipeline_version: Optional[str] = None,
    ):
        self.type_registry = DocumentTypeRegistry(document_types_path)
        self.date_extractor = DateExtractor()
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.run_id = run_id or generate_run_id("DOC")
        self.pipeline_version = pipeline_version or get_git_revision_hash()

    def extract(
        self,
        file_path: str,
        doc_graph: Optional[DocumentGraph] = None,
        doc_id: Optional[str] = None,
        content_sample: Optional[str] = None,
    ) -> DocumentMetadata:
        """
        Extract all document metadata.

        Args:
            file_path: Path to the document file
            doc_graph: Optional DocumentGraph for content access
            doc_id: Optional document ID
            content_sample: Optional pre-extracted content sample for classification

        Returns:
            DocumentMetadata with all extracted information
        """
        path = Path(file_path)
        filename = path.name
        doc_id = doc_id or filename

        # 1. Extract file metadata
        file_meta = self._extract_file_metadata(path)

        # 2. Extract PDF metadata (if PDF)
        pdf_meta = None
        if path.suffix.lower() == ".pdf" and PYMUPDF_AVAILABLE:
            pdf_meta = self._extract_pdf_metadata(path)

        # 3. Get content sample for classification/description
        if content_sample is None and doc_graph:
            content_sample = self._get_content_sample(doc_graph)

        # 3b. Extract DOI from content if not found in PDF annotations
        if pdf_meta and not pdf_meta.doi and content_sample:
            doi, doi_url = DOIExtractor.extract_from_text(content_sample)
            if doi:
                # Update pdf_meta with DOI found in content
                pdf_meta = PDFMetadata(
                    **{**pdf_meta.model_dump(), "doi": doi, "doi_url": doi_url}
                )
        elif not pdf_meta and content_sample:
            # No PDF metadata but we have content - check for DOI
            doi, doi_url = DOIExtractor.extract_from_text(content_sample)
            if doi:
                pdf_meta = PDFMetadata(doi=doi, doi_url=doi_url)

        # 4. Extract dates with fallback chain
        date_result = self._extract_dates(
            filename=filename,
            content=content_sample,
            pdf_meta=pdf_meta,
            file_meta=file_meta,
        )

        # 5. Classify document (LLM)
        classification = None
        if self.llm_client and content_sample:
            classification = self._classify_document(filename, content_sample)

        # 6. Generate descriptions (LLM)
        description = None
        if self.llm_client and content_sample:
            description = self._generate_descriptions(
                filename=filename,
                content=content_sample,
                pdf_title=pdf_meta.title if pdf_meta else None,
            )

        # Build provenance
        provenance = DocumentMetadataProvenance(
            pipeline_version=self.pipeline_version,
            run_id=self.run_id,
            doc_fingerprint=doc_id,
            generator_name=DocumentMetadataGeneratorType.LLM_EXTRACTION,
        )

        return DocumentMetadata(
            doc_id=doc_id,
            doc_filename=filename,
            file_metadata=file_meta,
            pdf_metadata=pdf_meta,
            classification=classification,
            description=description,
            date_extraction=date_result,
            provenance=provenance,
        )

    def _extract_file_metadata(self, path: Path) -> FileMetadata:
        """Extract file system metadata."""
        try:
            stat_info = path.stat()

            # Format size
            size_bytes = stat_info.st_size
            if size_bytes < 1024:
                size_human = f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                size_human = f"{size_bytes / 1024:.1f} KB"
            else:
                size_human = f"{size_bytes / (1024 * 1024):.1f} MB"

            # Format permissions (Unix-style)
            mode = stat_info.st_mode
            permissions = stat.filemode(mode)[1:]  # Remove leading 'd' or '-'

            return FileMetadata(
                file_path=str(path.absolute()),
                filename=path.name,
                extension=path.suffix.lower(),
                size_bytes=size_bytes,
                size_human=size_human,
                created_at=datetime.fromtimestamp(stat_info.st_ctime),
                modified_at=datetime.fromtimestamp(stat_info.st_mtime),
                accessed_at=datetime.fromtimestamp(stat_info.st_atime),
                permissions=permissions,
            )
        except Exception:
            # Return minimal metadata on error
            return FileMetadata(
                file_path=str(path),
                filename=path.name,
                extension=path.suffix.lower(),
                size_bytes=0,
                size_human="Unknown",
            )

    def _extract_pdf_metadata(self, path: Path) -> Optional[PDFMetadata]:
        """Extract PDF metadata using PyMuPDF."""
        if not PYMUPDF_AVAILABLE:
            return None

        try:
            doc = fitz.open(str(path))
            meta = doc.metadata

            # Parse dates from PDF metadata
            creation_date = self._parse_pdf_date(meta.get("creationDate"))
            mod_date = self._parse_pdf_date(meta.get("modDate"))

            # Parse keywords
            keywords = None
            if meta.get("keywords"):
                keywords = [k.strip() for k in meta["keywords"].split(",") if k.strip()]

            # Check for forms and annotations
            has_forms = False
            has_annotations = False
            for page in doc:
                if page.widgets():
                    has_forms = True
                if page.annots():
                    has_annotations = True
                if has_forms and has_annotations:
                    break

            # Extract DOI from PDF link annotations
            doi, doi_url = DOIExtractor.extract_from_pdf_annotations(doc)

            pdf_meta = PDFMetadata(
                title=meta.get("title") or None,
                author=meta.get("author") or None,
                subject=meta.get("subject") or None,
                keywords=keywords,
                creator=meta.get("creator") or None,
                producer=meta.get("producer") or None,
                creation_date=creation_date,
                modification_date=mod_date,
                page_count=len(doc),
                pdf_version=f"PDF {doc.metadata.get('format', '1.x')}",
                is_encrypted=doc.is_encrypted,
                is_tagged=doc.is_pdf,  # Simplified check
                has_form_fields=has_forms,
                has_annotations=has_annotations,
                doi=doi,
                doi_url=doi_url,
            )

            doc.close()
            return pdf_meta
        except Exception as e:
            print(f"[WARN] Failed to extract PDF metadata: {e}")
            return None

    def _parse_pdf_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse PDF date string (D:YYYYMMDDHHmmSS format)."""
        if not date_str:
            return None

        try:
            # Remove "D:" prefix and timezone info
            date_str = date_str.replace("D:", "")
            date_str = re.sub(r"[+-]\d{2}'\d{2}'?$", "", date_str)
            date_str = date_str.replace("Z", "")

            # Try parsing
            if len(date_str) >= 14:
                return datetime.strptime(date_str[:14], "%Y%m%d%H%M%S")
            elif len(date_str) >= 8:
                return datetime.strptime(date_str[:8], "%Y%m%d")
            elif len(date_str) >= 6:
                return datetime.strptime(date_str[:6], "%Y%m")
            elif len(date_str) >= 4:
                return datetime.strptime(date_str[:4], "%Y")
        except ValueError:
            pass

        return None

    def _get_content_sample(
        self,
        doc_graph: DocumentGraph,
        max_chars: int = 5000,
    ) -> str:
        """Get content sample from document graph."""
        # Get first N characters of content
        full_text = doc_graph.get_full_text() if hasattr(doc_graph, "get_full_text") else ""
        return full_text[:max_chars] if full_text else ""

    def _extract_dates(
        self,
        filename: str,
        content: Optional[str],
        pdf_meta: Optional[PDFMetadata],
        file_meta: Optional[FileMetadata],
    ) -> DateExtractionResult:
        """
        Extract dates with fallback chain.

        Priority:
        1. Filename date
        2. Content dates
        3. PDF metadata dates
        4. File system dates
        """
        result = DateExtractionResult()
        fallback_chain = []

        # 1. Try filename
        filename_date = self.date_extractor.extract_from_filename(filename)
        if filename_date:
            result.filename_date = filename_date
            fallback_chain.append(DateSourceType.FILENAME)

        # 2. Try content
        if content:
            content_dates = self.date_extractor.extract_from_content(content)
            result.content_dates = content_dates
            if content_dates:
                fallback_chain.append(DateSourceType.CONTENT)

        # 3. Try PDF metadata
        if pdf_meta:
            if pdf_meta.creation_date:
                result.pdf_creation_date = self.date_extractor.create_from_datetime(
                    pdf_meta.creation_date,
                    DateSourceType.PDF_METADATA,
                    confidence=0.6,
                )
                if DateSourceType.PDF_METADATA not in fallback_chain:
                    fallback_chain.append(DateSourceType.PDF_METADATA)

            if pdf_meta.modification_date:
                result.pdf_modification_date = self.date_extractor.create_from_datetime(
                    pdf_meta.modification_date,
                    DateSourceType.PDF_METADATA,
                    confidence=0.5,
                )

        # 4. File system dates
        if file_meta:
            if file_meta.created_at:
                result.file_system_created = self.date_extractor.create_from_datetime(
                    file_meta.created_at,
                    DateSourceType.FILE_SYSTEM,
                    confidence=0.3,
                )
                if DateSourceType.FILE_SYSTEM not in fallback_chain:
                    fallback_chain.append(DateSourceType.FILE_SYSTEM)

            if file_meta.modified_at:
                result.file_system_modified = self.date_extractor.create_from_datetime(
                    file_meta.modified_at,
                    DateSourceType.FILE_SYSTEM,
                    confidence=0.3,
                )

        result.fallback_chain_used = fallback_chain

        # Select primary date based on fallback chain
        if result.filename_date:
            result.primary_date = result.filename_date
        elif result.content_dates:
            # Pick highest confidence content date
            result.primary_date = max(result.content_dates, key=lambda d: d.confidence)
        elif result.pdf_creation_date:
            result.primary_date = result.pdf_creation_date
        elif result.file_system_modified:
            result.primary_date = result.file_system_modified

        return result

    def _classify_document(
        self,
        filename: str,
        content: str,
    ) -> Optional[DocumentClassification]:
        """Classify document using LLM."""
        if not self.llm_client:
            return None

        # Build prompt with document types
        types_list = self.type_registry.get_types_for_prompt()

        system_prompt = """You are a document classifier for clinical research documents.
Classify the document into ONE of the provided document types based on its filename and content.

Respond in JSON format:
{
    "primary_code": "PRO",
    "primary_confidence": 0.85,
    "reasoning": "Brief explanation",
    "alternatives": [
        {"code": "AME", "confidence": 0.3}
    ]
}

Be specific - choose the most appropriate type. Confidence should reflect how certain you are."""

        user_prompt = f"""Document types available:
{types_list}

---
Filename: {filename}

Content sample:
{content[:3000]}

---
Classify this document. Return JSON only."""

        try:
            if hasattr(self.llm_client, "complete_json_any"):
                response = self.llm_client.complete_json_any(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=self.llm_model,
                    temperature=0.0,
                    max_tokens=500,
                )
            else:
                response = self.llm_client.complete_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=self.llm_model,
                    temperature=0.0,
                    max_tokens=500,
                )

            if not response:
                return None

            # Handle case where LLM returns a list instead of dict
            if isinstance(response, list):
                if len(response) > 0 and isinstance(response[0], dict):
                    response = response[0]
                else:
                    print(f"[WARN] Unexpected LLM response format: {type(response)}")
                    return None

            if not isinstance(response, dict):
                print(f"[WARN] Expected dict response, got: {type(response)}")
                return None

            # Parse response - handle both "primary_code" and "code" formats
            primary_code = response.get("primary_code") or response.get("code", "")
            primary_conf = float(response.get("primary_confidence") or response.get("confidence", 0.5))
            reasoning = response.get("reasoning", "")

            # Get type info
            type_info = self.type_registry.get_type_by_code(primary_code)
            if not type_info:
                print(f"  [WARN] Unknown document type code: '{primary_code}'")
                return None

            print(f"  Classification: {primary_code} - {type_info['name']} (conf: {primary_conf:.2f})")

            primary_type = DocumentType(
                code=type_info["code"],
                name=type_info["name"],
                group=type_info.get("group", "Unknown"),
                description=type_info.get("desc"),
                confidence=primary_conf,
            )

            # Parse alternatives
            alternatives = []
            for alt in response.get("alternatives", [])[:3]:
                alt_code = alt.get("code", "")
                alt_info = self.type_registry.get_type_by_code(alt_code)
                if alt_info:
                    alternatives.append(DocumentType(
                        code=alt_info["code"],
                        name=alt_info["name"],
                        group=alt_info.get("group", "Unknown"),
                        confidence=float(alt.get("confidence", 0.3)),
                    ))

            return DocumentClassification(
                primary_type=primary_type,
                alternative_types=alternatives,
                llm_reasoning=reasoning,
                classification_method="llm",
            )
        except Exception as e:
            print(f"[WARN] Document classification failed: {e}")
            return None

    def _generate_descriptions(
        self,
        filename: str,
        content: str,
        pdf_title: Optional[str] = None,
    ) -> Optional[DocumentDescription]:
        """Generate document descriptions using LLM."""
        if not self.llm_client:
            return None

        system_prompt = """You are a document analyst for clinical research documents.
Generate a title and descriptions for the document.

Respond in JSON format:
{
    "title": "Concise document title",
    "short_description": "1-2 sentence summary",
    "long_description": "Detailed paragraph about the document content and purpose",
    "key_topics": ["topic1", "topic2", "topic3"],
    "language": "en"
}

Be accurate and professional. Extract the actual title if visible in the content."""

        title_hint = f"\nPDF title metadata: {pdf_title}" if pdf_title else ""

        user_prompt = f"""Filename: {filename}{title_hint}

Content:
{content[:4000]}

---
Generate title and descriptions. Return JSON only."""

        try:
            if hasattr(self.llm_client, "complete_json_any"):
                response = self.llm_client.complete_json_any(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=self.llm_model,
                    temperature=0.0,
                    max_tokens=800,
                )
            else:
                response = self.llm_client.complete_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=self.llm_model,
                    temperature=0.0,
                    max_tokens=800,
                )

            if not response:
                return None

            # Handle case where LLM returns a list instead of dict
            if isinstance(response, list):
                if len(response) > 0 and isinstance(response[0], dict):
                    response = response[0]
                else:
                    return None

            if not isinstance(response, dict):
                return None

            # Determine title source
            title = response.get("title", filename)
            title_source = "llm"
            if pdf_title and title.lower() == pdf_title.lower():
                title_source = "pdf_metadata"

            return DocumentDescription(
                title=title,
                title_source=title_source,
                short_description=response.get("short_description", ""),
                long_description=response.get("long_description", ""),
                key_topics=response.get("key_topics", [])[:10],
                language=response.get("language", "en"),
                llm_model=self.llm_model,
            )
        except Exception as e:
            print(f"[WARN] Description generation failed: {e}")
            return None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def extract_document_metadata(
    file_path: str,
    doc_graph: Optional[DocumentGraph] = None,
    document_types_path: Optional[str] = None,
    llm_client: Optional[Any] = None,
    llm_model: str = "claude-sonnet-4-20250514",
) -> DocumentMetadata:
    """
    Convenience function to extract document metadata.

    Args:
        file_path: Path to the document
        doc_graph: Optional DocumentGraph
        document_types_path: Path to document_types.json
        llm_client: Optional LLM client for classification/description
        llm_model: LLM model to use

    Returns:
        DocumentMetadata with all extracted information
    """
    strategy = DocumentMetadataStrategy(
        document_types_path=document_types_path,
        llm_client=llm_client,
        llm_model=llm_model,
    )

    return strategy.extract(file_path=file_path, doc_graph=doc_graph)

# corpus_metadata/E_normalization/E14_citation_validator.py
"""
Citation identifier validation via external APIs.

Validates extracted identifiers by checking against authoritative sources:
- DOI: doi.org / CrossRef API
- NCT: ClinicalTrials.gov API
- PMID: PubMed/NCBI E-utilities API
- URL: HTTP HEAD request for accessibility
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import quote

logger = logging.getLogger(__name__)

# Rate limiting settings
REQUEST_DELAY = 0.5  # seconds between API calls
REQUEST_TIMEOUT = 10  # seconds


@dataclass
class ValidationResult:
    """Result of validating a single identifier."""

    identifier: str
    identifier_type: str  # "doi", "nct", "pmid", "url"
    is_valid: bool
    status_code: Optional[int] = None
    resolved_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


@dataclass
class CitationValidationReport:
    """Summary of all validation results."""

    total_checked: int = 0
    valid_count: int = 0
    invalid_count: int = 0
    error_count: int = 0
    results: List[ValidationResult] = field(default_factory=list)

    def add_result(self, result: ValidationResult) -> None:
        self.results.append(result)
        self.total_checked += 1
        if result.is_valid:
            self.valid_count += 1
        elif result.error_message:
            self.error_count += 1
        else:
            self.invalid_count += 1

    def to_summary(self) -> Dict[str, Any]:
        return {
            "total_checked": self.total_checked,
            "valid": self.valid_count,
            "invalid": self.invalid_count,
            "errors": self.error_count,
            "validation_rate": f"{(self.valid_count / self.total_checked * 100):.1f}%" if self.total_checked > 0 else "0%",
        }


class CitationValidator:
    """
    Validates citation identifiers against external APIs.

    Supports:
    - DOI validation via doi.org
    - NCT validation via ClinicalTrials.gov API
    - PMID validation via PubMed E-utilities
    - URL accessibility check via HTTP HEAD
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.timeout = config.get("timeout", REQUEST_TIMEOUT)
        self.delay = config.get("delay", REQUEST_DELAY)
        self.validate_urls = config.get("validate_urls", True)
        self._session = None

    def _get_session(self):
        """Lazy load requests session."""
        if self._session is None:
            try:
                import requests
                self._session = requests.Session()
                self._session.headers.update({
                    "User-Agent": "CorpusMetadata/0.8 (Clinical Trial Feasibility Extractor)"
                })
            except ImportError:
                logger.warning("requests library not available for citation validation")
                return None
        return self._session

    def validate_doi(self, doi: str) -> ValidationResult:
        """
        Validate DOI via doi.org resolution.

        Args:
            doi: DOI string (e.g., "10.1016/S0140-6736(25)01148-1")

        Returns:
            ValidationResult with metadata if valid
        """
        result = ValidationResult(
            identifier=doi,
            identifier_type="doi",
            is_valid=False,
        )

        session = self._get_session()
        if not session:
            result.error_message = "requests library not available"
            return result

        try:
            # Use doi.org content negotiation to get metadata
            url = f"https://doi.org/{quote(doi, safe='')}"

            # First, check if DOI resolves (HEAD request)
            response = session.head(
                url,
                timeout=self.timeout,
                allow_redirects=True,
            )

            result.status_code = response.status_code
            result.resolved_url = response.url if response.status_code == 200 else None

            if response.status_code == 200:
                result.is_valid = True

                # Try to get metadata via CrossRef API
                time.sleep(self.delay)
                try:
                    crossref_url = f"https://api.crossref.org/works/{quote(doi, safe='')}"
                    meta_response = session.get(
                        crossref_url,
                        timeout=self.timeout,
                        headers={"Accept": "application/json"},
                    )
                    if meta_response.status_code == 200:
                        data = meta_response.json()
                        work = data.get("message", {})
                        result.metadata = {
                            "title": work.get("title", [None])[0],
                            "authors": [
                                f"{a.get('given', '')} {a.get('family', '')}".strip()
                                for a in work.get("author", [])[:3]  # First 3 authors
                            ],
                            "journal": work.get("container-title", [None])[0],
                            "year": work.get("published-print", {}).get("date-parts", [[None]])[0][0]
                                    or work.get("published-online", {}).get("date-parts", [[None]])[0][0],
                            "type": work.get("type"),
                        }
                except Exception as e:
                    logger.debug(f"CrossRef metadata fetch failed: {e}")

            elif response.status_code == 404:
                result.error_message = "DOI not found"
            else:
                result.error_message = f"HTTP {response.status_code}"

        except Exception as e:
            result.error_message = str(e)
            logger.warning(f"DOI validation failed for {doi}: {e}")

        time.sleep(self.delay)
        return result

    def validate_nct(self, nct: str) -> ValidationResult:
        """
        Validate NCT number via ClinicalTrials.gov API.

        Args:
            nct: NCT number (e.g., "NCT04817618")

        Returns:
            ValidationResult with trial metadata if valid
        """
        # Normalize NCT format
        nct_clean = nct.upper().replace(" ", "")
        if not nct_clean.startswith("NCT"):
            nct_clean = f"NCT{nct_clean}"

        result = ValidationResult(
            identifier=nct_clean,
            identifier_type="nct",
            is_valid=False,
        )

        session = self._get_session()
        if not session:
            result.error_message = "requests library not available"
            return result

        try:
            # Use ClinicalTrials.gov v2 API
            url = f"https://clinicaltrials.gov/api/v2/studies/{nct_clean}"
            response = session.get(
                url,
                timeout=self.timeout,
                headers={"Accept": "application/json"},
            )

            result.status_code = response.status_code
            result.resolved_url = f"https://clinicaltrials.gov/study/{nct_clean}"

            if response.status_code == 200:
                result.is_valid = True
                data = response.json()

                # Extract key metadata
                protocol = data.get("protocolSection", {})
                id_module = protocol.get("identificationModule", {})
                status_module = protocol.get("statusModule", {})
                design_module = protocol.get("designModule", {})

                result.metadata = {
                    "title": id_module.get("briefTitle"),
                    "official_title": id_module.get("officialTitle"),
                    "status": status_module.get("overallStatus"),
                    "phase": design_module.get("phases", [None])[0] if design_module.get("phases") else None,
                    "study_type": design_module.get("studyType"),
                    "start_date": status_module.get("startDateStruct", {}).get("date"),
                    "completion_date": status_module.get("completionDateStruct", {}).get("date"),
                }

            elif response.status_code == 404:
                result.error_message = "NCT not found"
            else:
                result.error_message = f"HTTP {response.status_code}"

        except Exception as e:
            result.error_message = str(e)
            logger.warning(f"NCT validation failed for {nct}: {e}")

        time.sleep(self.delay)
        return result

    def validate_pmid(self, pmid: str) -> ValidationResult:
        """
        Validate PMID via PubMed E-utilities API.

        Args:
            pmid: PubMed ID (e.g., "12345678")

        Returns:
            ValidationResult with article metadata if valid
        """
        result = ValidationResult(
            identifier=pmid,
            identifier_type="pmid",
            is_valid=False,
        )

        session = self._get_session()
        if not session:
            result.error_message = "requests library not available"
            return result

        try:
            # Use NCBI E-utilities esummary
            url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            params = {
                "db": "pubmed",
                "id": pmid,
                "retmode": "json",
            }
            response = session.get(url, params=params, timeout=self.timeout)

            result.status_code = response.status_code
            result.resolved_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

            if response.status_code == 200:
                data = response.json()
                doc_sum = data.get("result", {}).get(pmid, {})

                if doc_sum and "error" not in doc_sum:
                    result.is_valid = True
                    result.metadata = {
                        "title": doc_sum.get("title"),
                        "authors": [a.get("name") for a in doc_sum.get("authors", [])[:3]],
                        "journal": doc_sum.get("fulljournalname") or doc_sum.get("source"),
                        "year": doc_sum.get("pubdate", "").split()[0] if doc_sum.get("pubdate") else None,
                        "doi": doc_sum.get("elocationid", "").replace("doi: ", "") if "doi:" in doc_sum.get("elocationid", "") else None,
                    }
                else:
                    result.error_message = "PMID not found"
            else:
                result.error_message = f"HTTP {response.status_code}"

        except Exception as e:
            result.error_message = str(e)
            logger.warning(f"PMID validation failed for {pmid}: {e}")

        time.sleep(self.delay)
        return result

    def validate_url(self, url: str) -> ValidationResult:
        """
        Check URL accessibility via HTTP HEAD request.

        Args:
            url: URL to check

        Returns:
            ValidationResult with accessibility status
        """
        result = ValidationResult(
            identifier=url,
            identifier_type="url",
            is_valid=False,
        )

        if not self.validate_urls:
            result.error_message = "URL validation disabled"
            return result

        session = self._get_session()
        if not session:
            result.error_message = "requests library not available"
            return result

        try:
            response = session.head(
                url,
                timeout=self.timeout,
                allow_redirects=True,
            )

            result.status_code = response.status_code
            result.resolved_url = response.url if response.url != url else None

            # Consider 2xx and 3xx as valid (redirects are ok)
            if 200 <= response.status_code < 400:
                result.is_valid = True
            else:
                result.error_message = f"HTTP {response.status_code}"

        except Exception as e:
            result.error_message = str(e)
            logger.warning(f"URL validation failed for {url}: {e}")

        time.sleep(self.delay)
        return result

    def validate_citations(
        self,
        citations: List[Dict[str, Any]],
    ) -> CitationValidationReport:
        """
        Validate a list of citation entries.

        Args:
            citations: List of citation dicts with doi, nct, pmid, url fields

        Returns:
            CitationValidationReport with all results
        """
        report = CitationValidationReport()

        for citation in citations:
            # Validate DOI
            if citation.get("doi"):
                result = self.validate_doi(citation["doi"])
                report.add_result(result)

            # Validate NCT
            if citation.get("nct"):
                result = self.validate_nct(citation["nct"])
                report.add_result(result)

            # Validate PMID
            if citation.get("pmid"):
                result = self.validate_pmid(citation["pmid"])
                report.add_result(result)

            # Validate URL (optional)
            if citation.get("url") and self.validate_urls:
                result = self.validate_url(citation["url"])
                report.add_result(result)

        return report


def validate_citation_file(
    json_path: str,
    config: Optional[Dict[str, Any]] = None,
) -> CitationValidationReport:
    """
    Validate citations from an exported JSON file.

    Args:
        json_path: Path to citations JSON file
        config: Optional configuration

    Returns:
        CitationValidationReport
    """
    import json
    from pathlib import Path

    with open(Path(json_path), "r", encoding="utf-8") as f:
        data = json.load(f)

    citations = data.get("citations", [])
    validator = CitationValidator(config)
    return validator.validate_citations(citations)

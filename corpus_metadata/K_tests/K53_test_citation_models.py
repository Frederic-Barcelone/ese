# corpus_metadata/K_tests/K53_test_citation_models.py
"""
Tests for A_core.A11_citation_models module.

Tests citation model validation, enums, and validation results.
"""

from __future__ import annotations

import uuid

import pytest

from A_core.A11_citation_models import (
    CitationIdentifierType,
    CitationGeneratorType,
    CitationProvenanceMetadata,
    CitationCandidate,
    ExtractedCitation,
    CitationValidation,
    CitationExportEntry,
    CitationValidationSummary,
    CitationExportDocument,
)
from A_core.A01_domain_models import BoundingBox, Coordinate, EvidenceSpan, ValidationStatus


@pytest.fixture
def sample_provenance():
    return CitationProvenanceMetadata(
        run_id="RUN_001",
        pipeline_version="1.0.0",
        doc_fingerprint="abc123",
        generator_name=CitationGeneratorType.REFERENCE_SECTION,
    )


@pytest.fixture
def sample_coordinate():
    return Coordinate(page_num=25, bbox=BoundingBox(coords=(100, 600, 500, 620)))


@pytest.fixture
def sample_evidence(sample_coordinate):
    return EvidenceSpan(
        text="[1] Smith J et al. N Engl J Med. 2024;390:1-10.",
        location=sample_coordinate,
        scope_ref="doc_001",
        start_char_offset=0,
        end_char_offset=47,
    )


class TestCitationIdentifierType:
    """Tests for CitationIdentifierType enum."""

    def test_all_values(self):
        assert CitationIdentifierType.PMID.value == "pmid"
        assert CitationIdentifierType.PMCID.value == "pmcid"
        assert CitationIdentifierType.DOI.value == "doi"
        assert CitationIdentifierType.NCT.value == "nct"
        assert CitationIdentifierType.URL.value == "url"


class TestCitationGeneratorType:
    """Tests for CitationGeneratorType enum."""

    def test_all_values(self):
        assert CitationGeneratorType.REGEX_PATTERN.value == "gen:citation_regex"
        assert CitationGeneratorType.REFERENCE_SECTION.value == "gen:citation_reference"
        assert CitationGeneratorType.INLINE_CITATION.value == "gen:citation_inline"


class TestCitationProvenanceMetadata:
    """Tests for CitationProvenanceMetadata class."""

    def test_create_valid(self):
        prov = CitationProvenanceMetadata(
            run_id="RUN_001",
            pipeline_version="1.0.0",
            doc_fingerprint="abc123",
            generator_name=CitationGeneratorType.REFERENCE_SECTION,
        )
        assert prov.generator_name == CitationGeneratorType.REFERENCE_SECTION


class TestCitationCandidate:
    """Tests for CitationCandidate class."""

    def test_create_with_pmid(self, sample_provenance, sample_coordinate):
        candidate = CitationCandidate(
            doc_id="doc_001",
            pmid="12345678",
            citation_text="Smith J et al. N Engl J Med. 2024;390:1-10.",
            citation_number=1,
            generator_type=CitationGeneratorType.REFERENCE_SECTION,
            identifier_types=[CitationIdentifierType.PMID],
            context_text="[1] Smith J et al...",
            context_location=sample_coordinate,
            provenance=sample_provenance,
        )
        assert candidate.pmid == "12345678"
        assert candidate.citation_number == 1

    def test_create_with_doi(self, sample_provenance, sample_coordinate):
        candidate = CitationCandidate(
            doc_id="doc_001",
            doi="10.1056/NEJMoa2024816",
            citation_text="Smith J. doi:10.1056/NEJMoa2024816",
            generator_type=CitationGeneratorType.REGEX_PATTERN,
            identifier_types=[CitationIdentifierType.DOI],
            context_text="Reference: Smith J. doi:10.1056/NEJMoa2024816",
            context_location=sample_coordinate,
            provenance=sample_provenance,
        )
        assert candidate.doi == "10.1056/NEJMoa2024816"

    def test_create_with_nct(self, sample_provenance, sample_coordinate):
        candidate = CitationCandidate(
            doc_id="doc_001",
            nct="NCT01234567",
            citation_text="ClinicalTrials.gov NCT01234567",
            generator_type=CitationGeneratorType.INLINE_CITATION,
            identifier_types=[CitationIdentifierType.NCT],
            context_text="registered at NCT01234567",
            context_location=sample_coordinate,
            provenance=sample_provenance,
        )
        assert candidate.nct == "NCT01234567"

    def test_create_with_multiple_identifiers(self, sample_provenance, sample_coordinate):
        candidate = CitationCandidate(
            doc_id="doc_001",
            pmid="12345678",
            doi="10.1056/NEJMoa2024816",
            citation_text="Smith J et al. PMID: 12345678, DOI: 10.1056/NEJMoa2024816",
            generator_type=CitationGeneratorType.REFERENCE_SECTION,
            identifier_types=[CitationIdentifierType.PMID, CitationIdentifierType.DOI],
            context_text="Reference",
            context_location=sample_coordinate,
            provenance=sample_provenance,
        )
        assert candidate.pmid == "12345678"
        assert candidate.doi == "10.1056/NEJMoa2024816"
        assert len(candidate.identifier_types) == 2

    def test_default_confidence(self, sample_provenance, sample_coordinate):
        candidate = CitationCandidate(
            doc_id="doc_001",
            pmid="12345678",
            citation_text="test",
            generator_type=CitationGeneratorType.REGEX_PATTERN,
            context_text="test",
            context_location=sample_coordinate,
            provenance=sample_provenance,
        )
        assert candidate.initial_confidence == 0.9  # default for citations


class TestExtractedCitation:
    """Tests for ExtractedCitation class."""

    def test_create_valid(self, sample_provenance, sample_evidence):
        citation = ExtractedCitation(
            candidate_id=uuid.uuid4(),
            doc_id="doc_001",
            pmid="12345678",
            doi="10.1056/NEJMoa2024816",
            citation_text="Smith J et al. N Engl J Med. 2024;390:1-10.",
            citation_number=1,
            primary_evidence=sample_evidence,
            status=ValidationStatus.VALIDATED,
            confidence_score=0.95,
            provenance=sample_provenance,
        )
        assert citation.pmid == "12345678"
        assert citation.status == ValidationStatus.VALIDATED


class TestCitationValidation:
    """Tests for CitationValidation class."""

    def test_valid_citation(self):
        validation = CitationValidation(
            is_valid=True,
            resolved_url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
            title="A Study of Something",
        )
        assert validation.is_valid is True

    def test_invalid_citation(self):
        validation = CitationValidation(
            is_valid=False,
            error="PMID not found",
        )
        assert validation.is_valid is False
        assert validation.error == "PMID not found"

    def test_nct_with_status(self):
        validation = CitationValidation(
            is_valid=True,
            resolved_url="https://clinicaltrials.gov/ct2/show/NCT01234567",
            title="Study Title",
            status="Recruiting",
        )
        assert validation.status == "Recruiting"


class TestCitationExportEntry:
    """Tests for CitationExportEntry class."""

    def test_create_minimal(self):
        entry = CitationExportEntry(
            pmid="12345678",
            citation_text="Smith J et al. N Engl J Med. 2024.",
            confidence=0.95,
        )
        assert entry.pmid == "12345678"

    def test_create_with_validation(self):
        entry = CitationExportEntry(
            pmid="12345678",
            citation_text="Smith J et al. N Engl J Med. 2024.",
            page=25,
            confidence=0.95,
            validation=CitationValidation(
                is_valid=True,
                resolved_url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
            ),
        )
        assert entry.validation is not None
        assert entry.validation.is_valid is True
        assert entry.validation.resolved_url == "https://pubmed.ncbi.nlm.nih.gov/12345678/"


class TestCitationValidationSummary:
    """Tests for CitationValidationSummary class."""

    def test_default_values(self):
        summary = CitationValidationSummary()
        assert summary.total_validated == 0
        assert summary.valid_count == 0
        assert summary.invalid_count == 0
        assert summary.error_count == 0

    def test_with_counts(self):
        summary = CitationValidationSummary(
            total_validated=10,
            valid_count=8,
            invalid_count=1,
            error_count=1,
        )
        assert summary.total_validated == 10


class TestCitationExportDocument:
    """Tests for CitationExportDocument class."""

    def test_create_valid(self):
        doc = CitationExportDocument(
            run_id="RUN_001",
            timestamp="2024-01-15T10:30:00Z",
            document="study.pdf",
            pipeline_version="1.0.0",
            total_detected=25,
            unique_identifiers=20,
            validation_summary=CitationValidationSummary(
                total_validated=20,
                valid_count=18,
                invalid_count=2,
            ),
            citations=[
                CitationExportEntry(
                    pmid="12345678",
                    citation_text="Smith J et al.",
                    confidence=0.95,
                ),
            ],
        )
        assert doc.total_detected == 25
        assert doc.validation_summary is not None
        assert doc.validation_summary.valid_count == 18

    def test_without_validation_summary(self):
        doc = CitationExportDocument(
            run_id="RUN_001",
            timestamp="2024-01-15T10:30:00Z",
            document="study.pdf",
            pipeline_version="1.0.0",
            total_detected=5,
            unique_identifiers=5,
            citations=[],
        )
        assert doc.validation_summary is None

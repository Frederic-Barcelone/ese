# corpus_metadata/K_tests/K52_test_author_models.py
"""
Tests for A_core.A10_author_models module.

Tests author model validation, enums, and provenance.
"""

from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from A_core.A10_author_models import (
    AuthorRoleType,
    AuthorGeneratorType,
    AuthorProvenanceMetadata,
    AuthorCandidate,
    ExtractedAuthor,
    AuthorExportEntry,
    AuthorExportDocument,
)
from A_core.A01_domain_models import BoundingBox, Coordinate, EvidenceSpan, ValidationStatus


@pytest.fixture
def sample_provenance():
    return AuthorProvenanceMetadata(
        run_id="RUN_001",
        pipeline_version="1.0.0",
        doc_fingerprint="abc123",
        generator_name=AuthorGeneratorType.HEADER_PATTERN,
    )


@pytest.fixture
def sample_coordinate():
    return Coordinate(page_num=1, bbox=BoundingBox(coords=(100, 200, 300, 220)))


@pytest.fixture
def sample_evidence(sample_coordinate):
    return EvidenceSpan(
        text="Principal Investigator: Jane Smith, MD",
        location=sample_coordinate,
        scope_ref="doc_001",
        start_char_offset=0,
        end_char_offset=38,
    )


class TestAuthorRoleType:
    """Tests for AuthorRoleType enum."""

    def test_all_values(self):
        assert AuthorRoleType.AUTHOR.value == "author"
        assert AuthorRoleType.PRINCIPAL_INVESTIGATOR.value == "principal_investigator"
        assert AuthorRoleType.CO_INVESTIGATOR.value == "co_investigator"
        assert AuthorRoleType.CORRESPONDING_AUTHOR.value == "corresponding_author"
        assert AuthorRoleType.STEERING_COMMITTEE.value == "steering_committee"
        assert AuthorRoleType.STUDY_CHAIR.value == "study_chair"
        assert AuthorRoleType.DATA_SAFETY_BOARD.value == "data_safety_board"
        assert AuthorRoleType.UNKNOWN.value == "unknown"

    def test_string_conversion(self):
        # str(Enum) includes class name, use .value for just the value
        assert AuthorRoleType.PRINCIPAL_INVESTIGATOR.value == "principal_investigator"


class TestAuthorGeneratorType:
    """Tests for AuthorGeneratorType enum."""

    def test_all_values(self):
        assert AuthorGeneratorType.HEADER_PATTERN.value == "gen:author_header"
        assert AuthorGeneratorType.AFFILIATION_BLOCK.value == "gen:author_affiliation"
        assert AuthorGeneratorType.CONTRIBUTION_SECTION.value == "gen:author_contribution"
        assert AuthorGeneratorType.INVESTIGATOR_LIST.value == "gen:author_investigator"
        assert AuthorGeneratorType.REGEX_PATTERN.value == "gen:author_regex"


class TestAuthorProvenanceMetadata:
    """Tests for AuthorProvenanceMetadata class."""

    def test_create_valid(self):
        prov = AuthorProvenanceMetadata(
            run_id="RUN_001",
            pipeline_version="1.0.0",
            doc_fingerprint="abc123",
            generator_name=AuthorGeneratorType.HEADER_PATTERN,
        )
        assert prov.run_id == "RUN_001"
        assert prov.generator_name == AuthorGeneratorType.HEADER_PATTERN

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            AuthorProvenanceMetadata(  # type: ignore[call-arg]
                run_id="RUN_001",
                # Missing pipeline_version, doc_fingerprint, generator_name
            )


class TestAuthorCandidate:
    """Tests for AuthorCandidate class."""

    def test_create_minimal(self, sample_provenance, sample_coordinate):
        candidate = AuthorCandidate(
            doc_id="doc_001",
            full_name="Jane Smith, MD",
            generator_type=AuthorGeneratorType.HEADER_PATTERN,
            context_text="Principal Investigator: Jane Smith, MD",
            context_location=sample_coordinate,
            provenance=sample_provenance,
        )
        assert candidate.full_name == "Jane Smith, MD"
        assert candidate.role == AuthorRoleType.UNKNOWN  # default
        assert candidate.initial_confidence == 0.8  # default

    def test_create_full(self, sample_provenance, sample_coordinate):
        candidate = AuthorCandidate(
            doc_id="doc_001",
            full_name="Jane Smith, MD, PhD",
            role=AuthorRoleType.PRINCIPAL_INVESTIGATOR,
            affiliation="Harvard Medical School",
            email="jsmith@hms.harvard.edu",
            orcid="0000-0001-2345-6789",
            generator_type=AuthorGeneratorType.HEADER_PATTERN,
            context_text="PI: Jane Smith",
            context_location=sample_coordinate,
            initial_confidence=0.95,
            provenance=sample_provenance,
        )
        assert candidate.role == AuthorRoleType.PRINCIPAL_INVESTIGATOR
        assert candidate.affiliation == "Harvard Medical School"
        assert candidate.orcid == "0000-0001-2345-6789"

    def test_auto_uuid(self, sample_provenance, sample_coordinate):
        candidate = AuthorCandidate(
            doc_id="doc_001",
            full_name="Test Author",
            generator_type=AuthorGeneratorType.HEADER_PATTERN,
            context_text="test",
            context_location=sample_coordinate,
            provenance=sample_provenance,
        )
        assert isinstance(candidate.id, uuid.UUID)
        assert len(str(candidate.id)) == 36

    def test_confidence_bounds(self, sample_provenance, sample_coordinate):
        with pytest.raises(ValidationError):
            AuthorCandidate(
                doc_id="doc_001",
                full_name="Test",
                generator_type=AuthorGeneratorType.HEADER_PATTERN,
                context_text="test",
                context_location=sample_coordinate,
                initial_confidence=1.5,  # > 1.0
                provenance=sample_provenance,
            )


class TestExtractedAuthor:
    """Tests for ExtractedAuthor class."""

    def test_create_valid(self, sample_provenance, sample_evidence):
        candidate_id = uuid.uuid4()
        author = ExtractedAuthor(
            candidate_id=candidate_id,
            doc_id="doc_001",
            full_name="Jane Smith, MD",
            role=AuthorRoleType.PRINCIPAL_INVESTIGATOR,
            primary_evidence=sample_evidence,
            status=ValidationStatus.VALIDATED,
            confidence_score=0.95,
            provenance=sample_provenance,
        )
        assert author.candidate_id == candidate_id
        assert author.status == ValidationStatus.VALIDATED

    def test_with_supporting_evidence(self, sample_provenance, sample_evidence):
        author = ExtractedAuthor(
            candidate_id=uuid.uuid4(),
            doc_id="doc_001",
            full_name="Jane Smith",
            role=AuthorRoleType.AUTHOR,
            primary_evidence=sample_evidence,
            supporting_evidence=[sample_evidence],
            status=ValidationStatus.VALIDATED,
            confidence_score=0.9,
            provenance=sample_provenance,
        )
        assert len(author.supporting_evidence) == 1


class TestAuthorExportEntry:
    """Tests for AuthorExportEntry class."""

    def test_create_minimal(self):
        entry = AuthorExportEntry(
            full_name="Jane Smith",
            role="principal_investigator",
            confidence=0.95,
        )
        assert entry.full_name == "Jane Smith"
        assert entry.affiliation is None

    def test_create_full(self):
        entry = AuthorExportEntry(
            full_name="Jane Smith, MD",
            role="principal_investigator",
            affiliation="Harvard Medical School",
            email="jsmith@hms.edu",
            orcid="0000-0001-2345-6789",
            context="From PI section",
            page=1,
            confidence=0.95,
        )
        assert entry.orcid == "0000-0001-2345-6789"


class TestAuthorExportDocument:
    """Tests for AuthorExportDocument class."""

    def test_create_valid(self):
        doc = AuthorExportDocument(
            run_id="RUN_001",
            timestamp="2024-01-15T10:30:00Z",
            document="study.pdf",
            pipeline_version="1.0.0",
            total_detected=5,
            unique_authors=4,
            authors=[
                AuthorExportEntry(
                    full_name="Jane Smith",
                    role="principal_investigator",
                    confidence=0.95,
                ),
            ],
        )
        assert doc.total_detected == 5
        assert len(doc.authors) == 1

    def test_empty_authors_list(self):
        doc = AuthorExportDocument(
            run_id="RUN_001",
            timestamp="2024-01-15T10:30:00Z",
            document="study.pdf",
            pipeline_version="1.0.0",
            total_detected=0,
            unique_authors=0,
            authors=[],
        )
        assert doc.total_detected == 0

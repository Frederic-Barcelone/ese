# corpus_metadata/K_tests/K34_test_validation_logger.py
"""
Tests for D_validation.D03_validation_logger module.

Tests validation logging and statistics tracking.
"""

from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path

import pytest

from D_validation.D03_validation_logger import ValidationLogger
from A_core.A01_domain_models import (
    Candidate,
    ExtractedEntity,
    ValidationStatus,
    FieldType,
    GeneratorType,
    ProvenanceMetadata,
    EvidenceSpan,
    Coordinate,
)


@pytest.fixture
def temp_log_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_candidate():
    return Candidate(
        id=uuid.uuid4(),
        doc_id="test_doc.pdf",
        short_form="TNF",
        long_form="Tumor Necrosis Factor",
        field_type=FieldType.DEFINITION_PAIR,
        generator_type=GeneratorType.SYNTAX_PATTERN,
        initial_confidence=0.8,
        context_text="TNF (Tumor Necrosis Factor) was measured",
        context_location=Coordinate(page_num=1),
        provenance=ProvenanceMetadata(
            run_id="TEST",
            pipeline_version="1.0",
            doc_fingerprint="abc123",
            generator_name=GeneratorType.SYNTAX_PATTERN,
        ),
    )


@pytest.fixture
def sample_entity(sample_candidate):
    return ExtractedEntity(
        candidate_id=sample_candidate.id,
        doc_id=sample_candidate.doc_id,
        short_form="TNF",
        long_form="Tumor Necrosis Factor",
        field_type=FieldType.DEFINITION_PAIR,
        primary_evidence=EvidenceSpan(
            text="TNF (Tumor Necrosis Factor)",
            location=Coordinate(page_num=1),
            scope_ref="abc",
            start_char_offset=0,
            end_char_offset=27,
        ),
        supporting_evidence=[],
        status=ValidationStatus.VALIDATED,
        confidence_score=0.95,
        provenance=sample_candidate.provenance,
    )


class TestValidationLogger:
    """Tests for ValidationLogger class."""

    def test_init_creates_directory(self, temp_log_dir):
        logger = ValidationLogger(log_dir=temp_log_dir, run_id="TEST_001")
        assert logger.log_dir.exists()
        assert logger.run_id == "TEST_001"

    def test_init_generates_run_id(self, temp_log_dir):
        logger = ValidationLogger(log_dir=temp_log_dir)
        assert logger.run_id.startswith("VAL_")

    def test_initial_stats(self, temp_log_dir):
        logger = ValidationLogger(log_dir=temp_log_dir)
        assert logger.stats["total"] == 0
        assert logger.stats["validated"] == 0
        assert logger.stats["rejected"] == 0
        assert logger.stats["ambiguous"] == 0
        assert logger.stats["errors"] == 0

    def test_log_validation_updates_stats(
        self, temp_log_dir, sample_candidate, sample_entity
    ):
        logger = ValidationLogger(log_dir=temp_log_dir)
        logger.log_validation(sample_candidate, sample_entity)

        assert logger.stats["total"] == 1
        assert logger.stats["validated"] == 1

    def test_log_rejected_updates_stats(self, temp_log_dir, sample_candidate):
        logger = ValidationLogger(log_dir=temp_log_dir)

        rejected_entity = ExtractedEntity(
            candidate_id=sample_candidate.id,
            doc_id=sample_candidate.doc_id,
            short_form="TNF",
            long_form="Wrong Expansion",
            field_type=FieldType.DEFINITION_PAIR,
            primary_evidence=EvidenceSpan(
                text="test",
                location=Coordinate(page_num=1),
                scope_ref="abc",
                start_char_offset=0,
                end_char_offset=4,
            ),
            supporting_evidence=[],
            status=ValidationStatus.REJECTED,
            confidence_score=0.2,
            rejection_reason="Invalid expansion",
            provenance=sample_candidate.provenance,
        )

        logger.log_validation(sample_candidate, rejected_entity)
        assert logger.stats["rejected"] == 1

    def test_log_error_updates_stats(self, temp_log_dir, sample_candidate):
        logger = ValidationLogger(log_dir=temp_log_dir)
        logger.log_error(sample_candidate, "API call failed", "api_error")

        assert logger.stats["errors"] == 1
        assert logger.stats["total"] == 1

    def test_log_creates_file(self, temp_log_dir, sample_candidate, sample_entity):
        logger = ValidationLogger(log_dir=temp_log_dir, run_id="TEST_LOG")
        logger.log_validation(sample_candidate, sample_entity)

        log_file = Path(temp_log_dir) / "TEST_LOG.jsonl"
        assert log_file.exists()

    def test_write_summary(self, temp_log_dir, sample_candidate, sample_entity):
        logger = ValidationLogger(log_dir=temp_log_dir, run_id="TEST_SUM")
        logger.log_validation(sample_candidate, sample_entity)

        summary_path = logger.write_summary()
        assert summary_path.exists()

        with open(summary_path) as f:
            summary = json.load(f)

        assert summary["run_id"] == "TEST_SUM"
        assert summary["stats"]["total"] == 1
        assert summary["stats"]["validated"] == 1


class TestValidationLoggerRates:
    """Tests for rate calculations."""

    def test_validation_rate(self, temp_log_dir):
        logger = ValidationLogger(log_dir=temp_log_dir)
        logger.stats = {"total": 100, "validated": 80, "rejected": 15, "ambiguous": 5, "errors": 0}

        summary_path = logger.write_summary()
        with open(summary_path) as f:
            summary = json.load(f)

        assert summary["validation_rate"] == 0.8
        assert summary["rejection_rate"] == 0.15

    def test_zero_total_rates(self, temp_log_dir):
        logger = ValidationLogger(log_dir=temp_log_dir)
        summary_path = logger.write_summary()

        with open(summary_path) as f:
            summary = json.load(f)

        assert summary["validation_rate"] == 0.0
        assert summary["rejection_rate"] == 0.0

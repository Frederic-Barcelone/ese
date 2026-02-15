# corpus_metadata/K_tests/K35_test_quote_verifier.py
"""
Tests for D_validation.D04_quote_verifier module.

Tests quote verification, numerical verification, and extraction verification.
"""

from __future__ import annotations

import pytest

from Z_utils.Z14_quote_verifier import (
    QuoteVerificationResult,
    NumericalVerificationResult,
    FieldVerificationResult,
    QuoteVerifier,
    NumericalVerifier,
    ExtractionVerifier,
    verify_quote,
    verify_number,
)


class TestQuoteVerificationResult:
    """Tests for QuoteVerificationResult dataclass."""

    def test_verified_result(self):
        result = QuoteVerificationResult(
            verified=True,
            position=(10, 25),
            match_ratio=1.0,
            matched_text="patients were randomized",
            is_exact_match=True,
        )
        assert result.verified
        assert result.position == (10, 25)
        assert result.is_exact_match

    def test_unverified_result(self):
        result = QuoteVerificationResult(verified=False, match_ratio=0.5)
        assert not result.verified
        assert result.match_ratio == 0.5


class TestNumericalVerificationResult:
    """Tests for NumericalVerificationResult dataclass."""

    def test_verified_number(self):
        result = NumericalVerificationResult(
            verified=True,
            positions=[(100, 103)],
            matched_formats=["350"],
        )
        assert result.verified
        assert len(result.positions) == 1

    def test_multiple_positions(self):
        result = NumericalVerificationResult(
            verified=True,
            positions=[(10, 13), (50, 53), (100, 103)],
            matched_formats=["350", "350", "350"],
        )
        assert len(result.positions) == 3


class TestFieldVerificationResult:
    """Tests for FieldVerificationResult dataclass."""

    def test_is_verified_all_pass(self):
        result = FieldVerificationResult(
            field_name="test",
            quote_result=QuoteVerificationResult(verified=True),
            numerical_results={
                "count": NumericalVerificationResult(verified=True),
            },
        )
        assert result.is_verified

    def test_is_verified_quote_fails(self):
        result = FieldVerificationResult(
            field_name="test",
            quote_result=QuoteVerificationResult(verified=False),
        )
        assert not result.is_verified

    def test_confidence_penalty_no_failures(self):
        result = FieldVerificationResult(field_name="test")
        assert result.confidence_penalty == 1.0

    def test_confidence_penalty_quote_fails(self):
        result = FieldVerificationResult(
            field_name="test",
            quote_result=QuoteVerificationResult(verified=False),
        )
        assert result.confidence_penalty == 0.5

    def test_confidence_penalty_number_fails(self):
        result = FieldVerificationResult(
            field_name="test",
            numerical_results={
                "count": NumericalVerificationResult(verified=False),
            },
        )
        assert result.confidence_penalty == 0.7


class TestQuoteVerifier:
    """Tests for QuoteVerifier class."""

    @pytest.fixture
    def verifier(self):
        return QuoteVerifier(fuzzy_threshold=0.90)

    def test_exact_match(self, verifier):
        context = "The patients were randomized to receive treatment."
        quote = "patients were randomized"
        result = verifier.verify(quote, context)

        assert result.verified
        assert result.is_exact_match
        assert result.match_ratio == 1.0

    def test_case_insensitive_match(self, verifier):
        context = "The PATIENTS were randomized"
        quote = "patients were randomized"
        result = verifier.verify(quote, context, case_sensitive=False)

        assert result.verified

    def test_whitespace_normalization(self, verifier):
        context = "The patients   were  randomized"
        quote = "patients were randomized"
        result = verifier.verify(quote, context)

        assert result.verified

    def test_fuzzy_match(self, verifier):
        context = "The patients were randomised to receive treatment."
        quote = "patients were randomized to receive"
        result = verifier.verify(quote, context)

        # Should match with fuzzy threshold
        assert result.match_ratio >= 0.85

    def test_no_match(self, verifier):
        context = "Completely unrelated text about something else."
        quote = "patients were randomized"
        result = verifier.verify(quote, context)

        assert not result.verified

    def test_empty_quote(self, verifier):
        result = verifier.verify("", "some context")
        assert not result.verified

    def test_empty_context(self, verifier):
        result = verifier.verify("quote", "")
        assert not result.verified

    def test_whitespace_only_quote(self, verifier):
        result = verifier.verify("   ", "some context")
        assert not result.verified


class TestNumericalVerifier:
    """Tests for NumericalVerifier class."""

    @pytest.fixture
    def verifier(self):
        return NumericalVerifier()

    def test_exact_integer_match(self, verifier):
        context = "A total of 350 patients were enrolled."
        result = verifier.verify(350, context)

        assert result.verified
        assert "350" in result.matched_formats

    def test_comma_separated_number(self, verifier):
        context = "The study included 1,234 participants."
        result = verifier.verify(1234, context)

        assert result.verified

    def test_decimal_match(self, verifier):
        context = "Mean age was 45.5 years."
        result = verifier.verify(45.5, context)

        assert result.verified

    def test_string_input(self, verifier):
        context = "Dose was 500 mg daily."
        result = verifier.verify("500", context)

        assert result.verified

    def test_string_with_prefix(self, verifier):
        context = "Age >= 18 years required."
        result = verifier.verify(">=18", context)

        assert result.verified

    def test_number_not_found(self, verifier):
        context = "No numbers here."
        result = verifier.verify(999, context)

        assert not result.verified

    def test_with_tolerance(self, verifier):
        context = "Approximately 100 patients."
        result = verifier.verify(99, context, tolerance=0.1)

        assert result.verified  # 99 is within 10% of 100

    def test_verify_multiple(self, verifier):
        context = "Study enrolled 350 patients with mean age 55.2 years."
        values = {"patients": 350, "age": 55.2}

        results = verifier.verify_multiple(values, context)

        assert results["patients"].verified
        assert results["age"].verified


class TestExtractionVerifier:
    """Tests for ExtractionVerifier class."""

    @pytest.fixture
    def verifier(self):
        return ExtractionVerifier(fuzzy_threshold=0.90)

    def test_verify_field_with_quote(self, verifier):
        context = "Patients were enrolled in the study."
        result = verifier.verify_field(
            "enrollment",
            context,
            quote="Patients were enrolled",
        )

        assert result.is_verified

    def test_verify_field_with_numbers(self, verifier):
        context = "Study enrolled 350 patients over 24 months."
        result = verifier.verify_field(
            "enrollment",
            context,
            numerical_values={"count": 350, "duration": 24},
        )

        assert result.is_verified

    def test_verify_extraction(self, verifier):
        context = "A total of 500 patients were randomized."
        extraction = {
            "quote": "patients were randomized",
            "patient_count": 500,
        }

        results = verifier.verify_extraction(
            extraction,
            context,
            quote_fields=["quote"],
            numerical_fields=["patient_count"],
        )

        assert "quote" in results
        assert "patient_count" in results

    def test_calculate_confidence_penalty(self, verifier):
        results = {
            "field1": FieldVerificationResult(
                field_name="field1",
                quote_result=QuoteVerificationResult(verified=True),
            ),
            "field2": FieldVerificationResult(
                field_name="field2",
                quote_result=QuoteVerificationResult(verified=False),
            ),
        }

        penalty = verifier.calculate_confidence_penalty(results)
        assert penalty == 0.5  # One quote failed


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_verify_quote_found(self):
        assert verify_quote("patients enrolled", "The patients enrolled in the study.")

    def test_verify_quote_not_found(self):
        assert not verify_quote("something else", "The patients enrolled in the study.")

    def test_verify_number_found(self):
        assert verify_number(350, "Enrolled 350 patients.")

    def test_verify_number_not_found(self):
        assert not verify_number(999, "Enrolled 350 patients.")

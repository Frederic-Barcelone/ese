# corpus_metadata/K_tests/K33_test_llm_engine.py
"""
Tests for D_validation.D02_llm_engine module.

Tests LLM engine, Claude client, and verification result handling.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from D_validation.D02_llm_engine import (
    ClaudeClient,
    LLMEngine,
    VerificationResult,
)
from A_core.A01_domain_models import ValidationStatus


class TestVerificationResult:
    """Tests for VerificationResult model."""

    def test_valid_result(self):
        result = VerificationResult(
            status=ValidationStatus.VALIDATED,
            confidence=0.95,
            evidence="TNF-alpha was measured",
            reason="Explicit definition found",
            corrected_long_form=None,
        )
        assert result.status == ValidationStatus.VALIDATED
        assert result.confidence == 0.95

    def test_confidence_bounds(self):
        # Valid range
        result = VerificationResult(
            status=ValidationStatus.VALIDATED,
            confidence=0.0,
        )
        assert result.confidence == 0.0

        result = VerificationResult(
            status=ValidationStatus.VALIDATED,
            confidence=1.0,
        )
        assert result.confidence == 1.0

    def test_rejected_with_reason(self):
        result = VerificationResult(
            status=ValidationStatus.REJECTED,
            confidence=0.9,
            reason="Not a valid abbreviation",
        )
        assert result.status == ValidationStatus.REJECTED
        assert "abbreviation" in result.reason.lower()


class TestClaudeClientJsonParsing:
    """Tests for ClaudeClient JSON extraction methods."""

    @pytest.fixture
    def mock_client(self):
        """Create a ClaudeClient with mocked anthropic."""
        with patch("D_validation.D02_llm_engine.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value = MagicMock()
            client = ClaudeClient(api_key="test-key")
            return client

    def test_extract_json_object(self, mock_client):
        text = '{"status": "VALIDATED", "confidence": 0.9}'
        result = mock_client._extract_json_object(text)
        assert result["status"] == "VALIDATED"

    def test_extract_json_from_markdown(self, mock_client):
        text = '```json\n{"status": "VALIDATED", "confidence": 0.9}\n```'
        result = mock_client._extract_json_any(text)
        assert result["status"] == "VALIDATED"

    def test_extract_json_array(self, mock_client):
        text = '[{"id": "1", "status": "VALIDATED"}]'
        result = mock_client._extract_json_any(text)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_extract_json_with_wrapper(self, mock_client):
        text = '{"results": [{"id": "1", "status": "VALIDATED"}]}'
        result = mock_client._extract_json_any(text)
        # _extract_json_any extracts the inner array from the wrapper
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["status"] == "VALIDATED"

    def test_extract_json_fallback(self, mock_client):
        text = "This is not valid JSON"
        result = mock_client._extract_json_object(text)
        assert result["status"] == "AMBIGUOUS"

    def test_detect_media_type_jpeg(self, mock_client):
        # JPEG starts with /9j/
        assert mock_client._detect_media_type("/9j/abc") == "image/jpeg"

    def test_detect_media_type_png(self, mock_client):
        # Default to PNG
        assert mock_client._detect_media_type("iVBOR") == "image/png"

    def test_find_balanced_braces(self, mock_client):
        text = 'prefix {"key": "value"} suffix'
        result = mock_client._find_balanced(text, "{", "}")
        assert result == '{"key": "value"}'

    def test_find_balanced_nested(self, mock_client):
        text = '{"outer": {"inner": 1}}'
        result = mock_client._find_balanced(text, "{", "}")
        assert result == '{"outer": {"inner": 1}}'


class TestLLMEngineHelpers:
    """Tests for LLMEngine helper methods."""

    @pytest.fixture
    def mock_engine(self):
        """Create an LLMEngine with mocked client."""
        mock_client = MagicMock()
        engine = LLMEngine(
            llm_client=mock_client,
            model="claude-test",
            run_id="TEST_RUN",
        )
        return engine

    def test_infer_offsets_found(self, mock_engine):
        context = "The TNF level was measured"
        start, end = mock_engine._infer_offsets(context, "TNF", None)
        assert start == 4
        assert end == 7

    def test_infer_offsets_with_lf(self, mock_engine):
        context = "Tumor Necrosis Factor (TNF) was measured"
        start, end = mock_engine._infer_offsets(context, "TNF", "Tumor Necrosis Factor")
        assert start == 0  # LF starts at 0
        assert end > 20  # Includes LF

    def test_infer_offsets_not_found(self, mock_engine):
        context = "Some unrelated text"
        start, end = mock_engine._infer_offsets(context, "XYZ", None)
        assert start == 0
        assert end == len(context)

    def test_select_task_definition(self, mock_engine):
        from A_core.A01_domain_models import FieldType
        from D_validation.D01_prompt_registry import PromptTask

        task = mock_engine._select_task(FieldType.DEFINITION_PAIR)
        assert task == PromptTask.VERIFY_DEFINITION_PAIR

    def test_select_task_short_form_only(self, mock_engine):
        from A_core.A01_domain_models import FieldType
        from D_validation.D01_prompt_registry import PromptTask

        task = mock_engine._select_task(FieldType.SHORT_FORM_ONLY)
        assert task == PromptTask.VERIFY_SHORT_FORM_ONLY


class TestLLMEngineCache:
    """Tests for LLMEngine caching."""

    @pytest.fixture
    def mock_engine(self):
        mock_client = MagicMock()
        engine = LLMEngine(llm_client=mock_client, model="claude-test")
        return engine

    def test_cache_empty_initially(self, mock_engine):
        assert len(mock_engine._validation_cache) == 0

    def test_cache_key_format(self, mock_engine):
        # Cache uses (SF_UPPER, lf_lower_or_none)
        mock_engine._validation_cache[("TNF", "tumor necrosis factor")] = {
            "status": "VALIDATED",
            "confidence": 0.9,
        }
        assert ("TNF", "tumor necrosis factor") in mock_engine._validation_cache


class TestClaudeClientInit:
    """Tests for ClaudeClient initialization."""

    def test_missing_api_key_raises(self):
        with patch("D_validation.D02_llm_engine.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value = MagicMock()
            with patch.dict("os.environ", {}, clear=True):
                with pytest.raises(ValueError, match="API key"):
                    ClaudeClient()

    def test_api_key_from_param(self):
        with patch("D_validation.D02_llm_engine.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value = MagicMock()
            client = ClaudeClient(api_key="test-key-123")
            assert client.api_key == "test-key-123"

    def test_default_model(self):
        with patch("D_validation.D02_llm_engine.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value = MagicMock()
            client = ClaudeClient(api_key="test-key")
            assert "claude" in client.default_model.lower()

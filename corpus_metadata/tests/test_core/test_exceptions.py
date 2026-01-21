# corpus_metadata/tests/test_core/test_exceptions.py
"""Tests for A_core/A12_exceptions.py - Exception hierarchy."""

import pytest

from A_core.A12_exceptions import (
    ESEPipelineError,
    ConfigurationError,
    ParsingError,
    ExtractionError,
    EnrichmentError,
    APIError,
    RateLimitError,
    ValidationError,
    CacheError,
    EvaluationError,
)


class TestESEPipelineError:
    """Tests for the base exception class."""

    def test_basic_message(self):
        """Test exception with just a message."""
        exc = ESEPipelineError("Something went wrong")
        assert str(exc) == "Something went wrong"
        assert exc.message == "Something went wrong"
        assert exc.context == {}

    def test_with_context(self):
        """Test exception with context dictionary."""
        exc = ESEPipelineError("Error occurred", context={"file": "test.pdf", "line": 42})
        assert "file=test.pdf" in str(exc)
        assert "line=42" in str(exc)
        assert exc.context == {"file": "test.pdf", "line": 42}

    def test_inheritance(self):
        """Test that all exceptions inherit from base."""
        exc = ConfigurationError("Invalid config")
        assert isinstance(exc, ESEPipelineError)
        assert isinstance(exc, Exception)


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_basic(self):
        """Test basic configuration error."""
        exc = ConfigurationError("Missing API key")
        assert exc.message == "Missing API key"

    def test_with_key_info(self):
        """Test with configuration key information."""
        exc = ConfigurationError(
            "Invalid value",
            config_key="timeout_seconds",
            expected_type="int",
            actual_value="abc",
        )
        assert exc.config_key == "timeout_seconds"
        assert exc.expected_type == "int"
        assert exc.actual_value == "abc"
        assert "key=timeout_seconds" in str(exc)


class TestParsingError:
    """Tests for ParsingError."""

    def test_with_file_info(self):
        """Test with file path and page number."""
        exc = ParsingError(
            "Failed to extract text",
            file_path="/path/to/doc.pdf",
            page_number=5,
        )
        assert exc.file_path == "/path/to/doc.pdf"
        assert exc.page_number == 5
        assert "file=/path/to/doc.pdf" in str(exc)
        assert "page=5" in str(exc)


class TestExtractionError:
    """Tests for ExtractionError."""

    def test_with_extractor_info(self):
        """Test with extractor context."""
        exc = ExtractionError(
            "NER model failed",
            extractor_name="biomedical_ner",
            entity_type="Disease",
            input_text="This is a very long text...",
        )
        assert exc.extractor_name == "biomedical_ner"
        assert exc.entity_type == "Disease"
        assert "extractor=biomedical_ner" in str(exc)

    def test_input_truncation(self):
        """Test that long input text is truncated in context."""
        long_text = "x" * 200
        exc = ExtractionError("Failed", input_text=long_text)
        # Input should be truncated to 100 chars + "..."
        assert len(exc.context.get("input", "")) <= 103


class TestEnrichmentError:
    """Tests for EnrichmentError."""

    def test_with_entity_info(self):
        """Test with entity context."""
        exc = EnrichmentError(
            "PubTator lookup failed",
            enricher_name="pubtator",
            entity_id="disease_001",
            entity_type="Disease",
        )
        assert exc.enricher_name == "pubtator"
        assert exc.entity_id == "disease_001"


class TestAPIError:
    """Tests for APIError."""

    def test_with_status_code(self):
        """Test with HTTP status code."""
        exc = APIError(
            "Request failed",
            status_code=404,
            api_name="PubTator",
            endpoint="/autocomplete",
        )
        assert exc.status_code == 404
        assert exc.api_name == "PubTator"
        assert "status_code=404" in str(exc)

    def test_response_truncation(self):
        """Test that long response bodies are truncated."""
        long_response = "x" * 300
        exc = APIError("Error", response_body=long_response)
        assert len(exc.context.get("response", "")) <= 203

    def test_inheritance_from_enrichment(self):
        """Test that APIError inherits from EnrichmentError."""
        exc = APIError("API failed", api_name="test")
        assert isinstance(exc, EnrichmentError)
        assert isinstance(exc, ESEPipelineError)


class TestRateLimitError:
    """Tests for RateLimitError."""

    def test_with_retry_after(self):
        """Test with retry-after information."""
        exc = RateLimitError(
            "Rate limited by API",
            api_name="ClinicalTrials",
            retry_after=60,
        )
        assert exc.status_code == 429  # Always 429 for rate limits
        assert exc.retry_after == 60
        assert "retry_after=60" in str(exc)

    def test_inheritance(self):
        """Test inheritance chain."""
        exc = RateLimitError("Limited")
        assert isinstance(exc, APIError)
        assert isinstance(exc, EnrichmentError)
        assert isinstance(exc, ESEPipelineError)


class TestValidationError:
    """Tests for ValidationError."""

    def test_with_field_info(self):
        """Test with validation field information."""
        exc = ValidationError(
            "Confidence too low",
            entity_id="ent_001",
            field_name="confidence",
            expected_value=">0.5",
            actual_value="0.3",
        )
        assert exc.field_name == "confidence"
        assert exc.expected_value == ">0.5"
        assert exc.actual_value == "0.3"


class TestCacheError:
    """Tests for CacheError."""

    def test_with_operation_info(self):
        """Test with cache operation details."""
        exc = CacheError(
            "Write failed",
            cache_key="autocomplete_abc123",
            operation="set",
        )
        assert exc.cache_key == "autocomplete_abc123"
        assert exc.operation == "set"


class TestEvaluationError:
    """Tests for EvaluationError."""

    def test_with_metric_info(self):
        """Test with evaluation metric details."""
        exc = EvaluationError(
            "Invalid gold data format",
            metric_name="f1_score",
            file_path="/data/gold.json",
        )
        assert exc.metric_name == "f1_score"
        assert exc.file_path == "/data/gold.json"


class TestExceptionCatching:
    """Test exception catching patterns."""

    def test_catch_all_pipeline_errors(self):
        """Test catching all pipeline errors with base class."""
        errors = [
            ConfigurationError("Config error"),
            ParsingError("Parse error"),
            ExtractionError("Extract error"),
            EnrichmentError("Enrich error"),
            APIError("API error"),
            RateLimitError("Rate limit"),
            ValidationError("Validation error"),
            CacheError("Cache error"),
            EvaluationError("Eval error"),
        ]

        for error in errors:
            with pytest.raises(ESEPipelineError):
                raise error

    def test_catch_api_errors_specifically(self):
        """Test catching only API-related errors."""
        with pytest.raises(APIError):
            raise RateLimitError("Limited")

        with pytest.raises(EnrichmentError):
            raise APIError("Failed")

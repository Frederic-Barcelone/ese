# corpus_metadata/A_core/A12_exceptions.py
"""
Exception hierarchy for the ESE pipeline.

Provides a structured exception hierarchy that replaces generic exceptions
with specific, meaningful error types. This enables:
- Better error handling and recovery
- More informative error messages
- Easier debugging and logging
- Type-safe exception catching

Hierarchy:
    ESEPipelineError (base)
    ├── ConfigurationError     # Invalid config, missing keys
    ├── ParsingError           # PDF/document parsing failures
    ├── ExtractionError        # Entity extraction failures
    ├── EnrichmentError        # Enrichment process failures
    │   └── APIError           # External API failures (from Z01_api_client)
    │       └── RateLimitError # HTTP 429 rate limiting
    ├── ValidationError        # Entity validation failures
    ├── CacheError             # Caching operation failures
    └── EvaluationError        # Evaluation/scoring failures

Usage:
    from A_core.A12_exceptions import (
        ExtractionError,
        EnrichmentError,
        ConfigurationError,
    )

    try:
        result = enricher.enrich(entity)
    except EnrichmentError as e:
        logger.error(f"Enrichment failed: {e.message}")
        if e.entity_id:
            logger.error(f"Entity: {e.entity_id}")
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class ESEPipelineError(Exception):
    """
    Base exception for all ESE pipeline errors.

    All custom exceptions inherit from this class to enable
    catching all pipeline errors with a single except clause.

    Attributes:
        message: Human-readable error description.
        context: Additional context data for debugging.
    """

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the exception.

        Args:
            message: Human-readable error description.
            context: Optional dictionary with additional context.
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def __str__(self) -> str:
        """Return error message with optional context."""
        if self.context:
            ctx_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} [{ctx_str}]"
        return self.message


class ConfigurationError(ESEPipelineError):
    """
    Raised when configuration is invalid or incomplete.

    Examples:
        - Missing required configuration key
        - Invalid configuration value type
        - Incompatible configuration options
    """

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        expected_type: Optional[str] = None,
        actual_value: Optional[Any] = None,
    ):
        """
        Initialize configuration error.

        Args:
            message: Error description.
            config_key: The problematic configuration key.
            expected_type: Expected type or format.
            actual_value: The actual invalid value.
        """
        context = {}
        if config_key:
            context["key"] = config_key
        if expected_type:
            context["expected"] = expected_type
        if actual_value is not None:
            context["actual"] = repr(actual_value)

        super().__init__(message, context)
        self.config_key = config_key
        self.expected_type = expected_type
        self.actual_value = actual_value


class ParsingError(ESEPipelineError):
    """
    Raised when document parsing fails.

    Examples:
        - Corrupted PDF file
        - Unsupported file format
        - Encoding issues
    """

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        page_number: Optional[int] = None,
    ):
        """
        Initialize parsing error.

        Args:
            message: Error description.
            file_path: Path to the problematic file.
            page_number: Page number where error occurred.
        """
        context = {}
        if file_path:
            context["file"] = file_path
        if page_number is not None:
            context["page"] = page_number

        super().__init__(message, context)
        self.file_path = file_path
        self.page_number = page_number


class ExtractionError(ESEPipelineError):
    """
    Raised when entity extraction fails.

    Examples:
        - NER model failure
        - Regex pattern error
        - Invalid input text
    """

    def __init__(
        self,
        message: str,
        extractor_name: Optional[str] = None,
        entity_type: Optional[str] = None,
        input_text: Optional[str] = None,
    ):
        """
        Initialize extraction error.

        Args:
            message: Error description.
            extractor_name: Name of the extractor that failed.
            entity_type: Type of entity being extracted.
            input_text: Truncated input text (for debugging).
        """
        context = {}
        if extractor_name:
            context["extractor"] = extractor_name
        if entity_type:
            context["entity_type"] = entity_type
        if input_text:
            # Truncate for logging
            context["input"] = input_text[:100] + "..." if len(input_text) > 100 else input_text

        super().__init__(message, context)
        self.extractor_name = extractor_name
        self.entity_type = entity_type
        self.input_text = input_text


class EnrichmentError(ESEPipelineError):
    """
    Raised when enrichment processing fails.

    Examples:
        - Failed to look up entity in database
        - Invalid enrichment response
        - Enrichment timeout
    """

    def __init__(
        self,
        message: str,
        enricher_name: Optional[str] = None,
        entity_id: Optional[str] = None,
        entity_type: Optional[str] = None,
    ):
        """
        Initialize enrichment error.

        Args:
            message: Error description.
            enricher_name: Name of the enricher that failed.
            entity_id: ID of the entity being enriched.
            entity_type: Type of the entity.
        """
        context = {}
        if enricher_name:
            context["enricher"] = enricher_name
        if entity_id:
            context["entity_id"] = entity_id
        if entity_type:
            context["entity_type"] = entity_type

        super().__init__(message, context)
        self.enricher_name = enricher_name
        self.entity_id = entity_id
        self.entity_type = entity_type


class APIError(EnrichmentError):
    """
    Raised when an external API request fails.

    This is a subclass of EnrichmentError since most API calls
    happen during enrichment. Can also be raised standalone.

    Examples:
        - HTTP 4xx/5xx responses
        - Connection timeout
        - Invalid JSON response
    """

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        api_name: Optional[str] = None,
        endpoint: Optional[str] = None,
    ):
        """
        Initialize API error.

        Args:
            message: Error description.
            status_code: HTTP status code.
            response_body: Truncated response body.
            api_name: Name of the API service.
            endpoint: API endpoint that failed.
        """
        super().__init__(message, enricher_name=api_name)

        if status_code:
            self.context["status_code"] = status_code
        if endpoint:
            self.context["endpoint"] = endpoint
        if response_body:
            self.context["response"] = (
                response_body[:200] + "..."
                if len(response_body) > 200
                else response_body
            )

        self.status_code = status_code
        self.response_body = response_body
        self.api_name = api_name
        self.endpoint = endpoint


class RateLimitError(APIError):
    """
    Raised when rate limited by an external API (HTTP 429).

    Indicates the client should wait before retrying.
    """

    def __init__(
        self,
        message: str,
        api_name: Optional[str] = None,
        retry_after: Optional[int] = None,
    ):
        """
        Initialize rate limit error.

        Args:
            message: Error description.
            api_name: Name of the API service.
            retry_after: Seconds to wait before retrying.
        """
        super().__init__(message, status_code=429, api_name=api_name)
        if retry_after:
            self.context["retry_after"] = retry_after
        self.retry_after = retry_after


class ValidationError(ESEPipelineError):
    """
    Raised when entity validation fails.

    Examples:
        - Entity doesn't meet confidence threshold
        - Invalid entity format
        - Failed LLM verification
    """

    def __init__(
        self,
        message: str,
        entity_id: Optional[str] = None,
        field_name: Optional[str] = None,
        expected_value: Optional[str] = None,
        actual_value: Optional[str] = None,
    ):
        """
        Initialize validation error.

        Args:
            message: Error description.
            entity_id: ID of the entity that failed validation.
            field_name: Name of the invalid field.
            expected_value: Expected value or format.
            actual_value: Actual invalid value.
        """
        context = {}
        if entity_id:
            context["entity_id"] = entity_id
        if field_name:
            context["field"] = field_name
        if expected_value:
            context["expected"] = expected_value
        if actual_value:
            context["actual"] = actual_value

        super().__init__(message, context)
        self.entity_id = entity_id
        self.field_name = field_name
        self.expected_value = expected_value
        self.actual_value = actual_value


class CacheError(ESEPipelineError):
    """
    Raised when a caching operation fails.

    Examples:
        - Cache read/write failure
        - Cache corruption
        - Permission denied
    """

    def __init__(
        self,
        message: str,
        cache_key: Optional[str] = None,
        operation: Optional[str] = None,
    ):
        """
        Initialize cache error.

        Args:
            message: Error description.
            cache_key: The cache key involved.
            operation: The operation that failed (get, set, delete).
        """
        context = {}
        if cache_key:
            context["key"] = cache_key
        if operation:
            context["operation"] = operation

        super().__init__(message, context)
        self.cache_key = cache_key
        self.operation = operation


class EvaluationError(ESEPipelineError):
    """
    Raised when evaluation or scoring fails.

    Examples:
        - Invalid gold data format
        - Missing required fields for scoring
        - Incompatible prediction format
    """

    def __init__(
        self,
        message: str,
        metric_name: Optional[str] = None,
        file_path: Optional[str] = None,
    ):
        """
        Initialize evaluation error.

        Args:
            message: Error description.
            metric_name: Name of the metric that failed.
            file_path: Path to the problematic data file.
        """
        context = {}
        if metric_name:
            context["metric"] = metric_name
        if file_path:
            context["file"] = file_path

        super().__init__(message, context)
        self.metric_name = metric_name
        self.file_path = file_path


# Re-export commonly used exceptions for convenience
__all__ = [
    "ESEPipelineError",
    "ConfigurationError",
    "ParsingError",
    "ExtractionError",
    "EnrichmentError",
    "APIError",
    "RateLimitError",
    "ValidationError",
    "CacheError",
    "EvaluationError",
]

# corpus_metadata/Z_utils/Z01_api_client.py
"""
Shared API client utilities for external service integrations.

Provides reusable components for HTTP clients with:
- Thread-safe disk caching with TTL
- Request rate limiting
- Abstract base class for API clients

This module extracts common functionality from E04_pubtator_enricher and
E06_nct_enricher to eliminate code duplication (~200 lines).

Usage:
    from Z_utils.Z01_api_client import BaseAPIClient, DiskCache, RateLimiter

    class MyAPIClient(BaseAPIClient):
        def __init__(self, config=None):
            super().__init__(
                config=config,
                service_name="myservice",
                default_rate_limit=5,
                default_cache_ttl_hours=24,
            )

        def fetch_data(self, query: str) -> dict:
            cache_key = self.cache.make_key("query", query)
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

            self.rate_limiter.wait()
            result = self._request("GET", "/endpoint", params={"q": query})
            self.cache.set(cache_key, result)
            return result
"""

from __future__ import annotations

import functools
import hashlib
import json
import random
import threading
import time
from abc import ABC
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Type, TypeVar, Union

import requests

from A_core.A00_logging import get_logger

logger = get_logger(__name__)

# Type variable for generic retry decorator
F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# RETRY WITH EXPONENTIAL BACKOFF
# =============================================================================


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
    ),
    retryable_status_codes: Tuple[int, ...] = (429, 500, 502, 503, 504),
) -> Callable[[F], F]:
    """
    Decorator that adds retry with exponential backoff to a function.

    Retries the decorated function on specific exceptions or HTTP status codes,
    with exponentially increasing delays between attempts plus jitter to avoid
    thundering herd problems.

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay in seconds (default: 30.0)
        retryable_exceptions: Tuple of exception types to retry on
        retryable_status_codes: HTTP status codes that trigger retry

    Returns:
        Decorated function with retry logic

    Usage:
        @retry_with_backoff(max_retries=3, base_delay=1.0)
        def fetch_data(url: str) -> dict:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()

    Example with custom exceptions:
        @retry_with_backoff(
            max_retries=5,
            retryable_exceptions=(APIError, TimeoutError),
        )
        def call_external_api() -> dict:
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Optional[Exception] = None

            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)

                    # Check for retryable HTTP response (if result is Response)
                    if isinstance(result, requests.Response):
                        if result.status_code in retryable_status_codes:
                            if attempt == max_retries:
                                result.raise_for_status()
                            delay = _calculate_delay(attempt, base_delay, max_delay)
                            logger.warning(
                                f"Retry {attempt + 1}/{max_retries} for "
                                f"{func.__name__} after HTTP {result.status_code}, "
                                f"waiting {delay:.1f}s"
                            )
                            time.sleep(delay)
                            continue

                    return result

                except retryable_exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.error(
                            f"Max retries ({max_retries}) exceeded for "
                            f"{func.__name__}: {e}"
                        )
                        raise

                    delay = _calculate_delay(attempt, base_delay, max_delay)
                    logger.warning(
                        f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                        f"after {type(e).__name__}: {e}, waiting {delay:.1f}s"
                    )
                    time.sleep(delay)

            # Should not reach here, but raise last exception if we do
            if last_exception:
                raise last_exception
            return None

        return wrapper  # type: ignore[return-value]
    return decorator


def _calculate_delay(attempt: int, base_delay: float, max_delay: float) -> float:
    """
    Calculate delay with exponential backoff and jitter.

    Uses exponential backoff (2^attempt) with a random jitter factor
    to prevent thundering herd problems.

    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Base delay in seconds
        max_delay: Maximum delay cap in seconds

    Returns:
        Delay in seconds
    """
    # Exponential backoff: base_delay * 2^attempt
    exponential = base_delay * (2 ** attempt)

    # Add jitter (0-50% of exponential delay)
    jitter = random.uniform(0, exponential * 0.5)

    # Cap at max_delay
    return min(exponential + jitter, max_delay)


class RetryConfig:
    """
    Configuration for retry behavior.

    Use this class to create reusable retry configurations.

    Example:
        api_retry = RetryConfig(max_retries=5, base_delay=2.0)

        @api_retry.decorator
        def call_api():
            ...
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        retryable_exceptions: Tuple[Type[Exception], ...] = (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
        ),
        retryable_status_codes: Tuple[int, ...] = (429, 500, 502, 503, 504),
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.retryable_exceptions = retryable_exceptions
        self.retryable_status_codes = retryable_status_codes

    @property
    def decorator(self) -> Callable[[F], F]:
        """Get a decorator with this configuration."""
        return retry_with_backoff(
            max_retries=self.max_retries,
            base_delay=self.base_delay,
            max_delay=self.max_delay,
            retryable_exceptions=self.retryable_exceptions,
            retryable_status_codes=self.retryable_status_codes,
        )


# Pre-configured retry policies for common use cases
DEFAULT_RETRY = RetryConfig()
API_RETRY = RetryConfig(max_retries=3, base_delay=1.0, max_delay=60.0)
AGGRESSIVE_RETRY = RetryConfig(max_retries=5, base_delay=0.5, max_delay=30.0)


# =============================================================================
# EXCEPTIONS
# =============================================================================


class CacheError(Exception):
    """Raised when a cache operation fails."""
    pass


class APIError(Exception):
    """Raised when an API request fails."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class RateLimitError(APIError):
    """Raised when rate limited by the API (HTTP 429)."""
    pass


class DiskCache:
    """
    Thread-safe disk-based cache with TTL support.

    Stores JSON-serializable values on disk with automatic expiration.
    Safe for concurrent access from multiple threads.

    Attributes:
        cache_dir: Directory for cache files.
        ttl_seconds: Time-to-live for cached values.
        enabled: Whether caching is active.
    """

    def __init__(
        self,
        cache_dir: Union[str, Path],
        ttl_seconds: int = 86400,
        enabled: bool = True,
    ):
        """
        Initialize the disk cache.

        Args:
            cache_dir: Directory to store cache files.
            ttl_seconds: Time-to-live in seconds (default: 24 hours).
            enabled: Whether caching is enabled.
        """
        self.cache_dir = Path(cache_dir)
        self.ttl_seconds = ttl_seconds
        self.enabled = enabled
        self._lock = threading.Lock()

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def make_key(self, prefix: str, value: str) -> str:
        """
        Generate a cache key from prefix and value.

        Args:
            prefix: Key prefix (e.g., "autocomplete", "search").
            value: Value to hash (e.g., search term).

        Returns:
            Cache key string.
        """
        normalized = value.lower().strip()
        hash_val = hashlib.md5(normalized.encode()).hexdigest()[:12]
        return f"{prefix}_{hash_val}"

    def _cache_path(self, key: str) -> Path:
        """Get file path for a cache key."""
        return self.cache_dir / f"{key}.json"

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve value from cache if exists and not expired.

        Args:
            key: Cache key.

        Returns:
            Cached value, or None if not found/expired.
        """
        if not self.enabled:
            return None

        path = self._cache_path(key)

        with self._lock:
            if not path.exists():
                return None

            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                timestamp = data.get("_cached_at", 0)

                if time.time() - timestamp > self.ttl_seconds:
                    path.unlink(missing_ok=True)
                    return None

                return data.get("result")

            except (json.JSONDecodeError, OSError) as e:
                logger.debug(f"Cache read error for {key}: {e}")
                return None

    def set(self, key: str, value: Any) -> bool:
        """
        Store value in cache.

        Args:
            key: Cache key.
            value: JSON-serializable value.

        Returns:
            True if successfully cached, False otherwise.
        """
        if not self.enabled:
            return False

        path = self._cache_path(key)

        with self._lock:
            try:
                data = {"_cached_at": time.time(), "result": value}
                path.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                return True

            except (TypeError, OSError) as e:
                logger.debug(f"Cache write error for {key}: {e}")
                return False

    def delete(self, key: str) -> bool:
        """
        Delete a cached value.

        Args:
            key: Cache key.

        Returns:
            True if deleted, False if not found.
        """
        if not self.enabled:
            return False

        path = self._cache_path(key)

        with self._lock:
            if path.exists():
                path.unlink()
                return True
            return False

    def clear(self) -> int:
        """
        Clear all cached values.

        Returns:
            Number of files deleted.
        """
        if not self.enabled:
            return 0

        count = 0
        with self._lock:
            for path in self.cache_dir.glob("*.json"):
                try:
                    path.unlink()
                    count += 1
                except OSError:
                    pass
        return count


class RateLimiter:
    """
    Thread-safe rate limiter for API requests.

    Enforces a maximum request rate by sleeping between requests.

    Attributes:
        requests_per_second: Maximum requests per second.
    """

    def __init__(self, requests_per_second: float = 1.0):
        """
        Initialize the rate limiter.

        Args:
            requests_per_second: Maximum requests per second (0 = unlimited).
        """
        self.requests_per_second = requests_per_second
        self._last_request_time = 0.0
        self._lock = threading.Lock()

    def wait(self) -> float:
        """
        Wait if necessary to maintain rate limit.

        Returns:
            Seconds waited (0 if no wait needed).
        """
        if self.requests_per_second <= 0:
            return 0.0

        with self._lock:
            min_interval = 1.0 / self.requests_per_second
            elapsed = time.time() - self._last_request_time

            if elapsed < min_interval:
                wait_time = min_interval - elapsed
                time.sleep(wait_time)
            else:
                wait_time = 0.0

            self._last_request_time = time.time()
            return wait_time


class BaseAPIClient(ABC):
    """
    Abstract base class for HTTP API clients.

    Provides common functionality for API clients including:
    - Configurable timeouts
    - Integrated rate limiting
    - Disk-based caching
    - Connection pooling via requests.Session
    - Standardized error handling

    Subclasses must implement their specific API methods.

    Attributes:
        base_url: Base URL for API requests.
        timeout: Request timeout in seconds.
        cache: DiskCache instance.
        rate_limiter: RateLimiter instance.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        service_name: str = "api",
        default_base_url: str = "",
        default_rate_limit: float = 1.0,
        default_cache_ttl_hours: int = 24,
        default_cache_dir: Optional[str] = None,
    ):
        """
        Initialize the API client.

        Args:
            config: Configuration dictionary with optional keys:
                - base_url: API base URL
                - timeout_seconds: Request timeout
                - rate_limit_per_second: Max requests per second
                - cache.enabled: Enable disk caching
                - cache.directory: Cache directory path
                - cache.ttl_hours: Cache TTL in hours
            service_name: Service identifier for logging/cache.
            default_base_url: Default base URL if not in config.
            default_rate_limit: Default rate limit if not in config.
            default_cache_ttl_hours: Default cache TTL if not in config.
            default_cache_dir: Default cache directory.
        """
        config = config or {}
        self.service_name = service_name

        # URL and timeout
        self.base_url = config.get("base_url", default_base_url).rstrip("/")
        self.timeout = config.get("timeout_seconds", 30)

        # Rate limiting
        rate_limit = config.get("rate_limit_per_second", default_rate_limit)
        self.rate_limiter = RateLimiter(rate_limit)

        # Caching
        cache_cfg = config.get("cache", {})
        cache_enabled = cache_cfg.get("enabled", True)
        cache_dir = cache_cfg.get(
            "directory",
            default_cache_dir or f"cache/{service_name}",
        )
        cache_ttl = cache_cfg.get("ttl_hours", default_cache_ttl_hours) * 3600

        self.cache = DiskCache(
            cache_dir=cache_dir,
            ttl_seconds=cache_ttl,
            enabled=cache_enabled,
        )

        # HTTP session for connection pooling
        self._session = requests.Session()
        self._session.headers["User-Agent"] = f"ESE-Pipeline/1.0 ({service_name})"

        logger.debug(
            f"{service_name} client initialized: "
            f"rate_limit={rate_limit}/s, cache_ttl={cache_ttl}s"
        )

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        retry_on_rate_limit: bool = True,
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to the API.

        Args:
            method: HTTP method (GET, POST, etc.).
            endpoint: API endpoint path (appended to base_url).
            params: Query parameters.
            json_data: JSON request body.
            headers: Additional headers.
            retry_on_rate_limit: Retry once on HTTP 429.

        Returns:
            Parsed JSON response.

        Raises:
            APIError: On request failure.
            RateLimitError: On HTTP 429 (after retry if enabled).
        """
        url = f"{self.base_url}{endpoint}" if endpoint.startswith("/") else endpoint

        try:
            self.rate_limiter.wait()

            response = self._session.request(
                method=method,
                url=url,
                params=params,
                json=json_data,
                headers=headers,
                timeout=self.timeout,
            )

            # Handle rate limiting
            if response.status_code == 429:
                if retry_on_rate_limit:
                    logger.warning(
                        f"{self.service_name} rate limited, waiting 60s..."
                    )
                    time.sleep(60)
                    return self._request(
                        method, endpoint, params, json_data, headers,
                        retry_on_rate_limit=False,
                    )
                raise RateLimitError(
                    f"Rate limited by {self.service_name}",
                    status_code=429,
                )

            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout as e:
            logger.warning(f"{self.service_name} timeout: {e}")
            raise APIError(f"Request timeout: {e}") from e

        except requests.exceptions.RequestException as e:
            logger.warning(f"{self.service_name} request failed: {e}")
            status = getattr(e.response, "status_code", None) if hasattr(e, "response") else None
            raise APIError(f"Request failed: {e}", status_code=status) from e

        except json.JSONDecodeError as e:
            logger.warning(f"{self.service_name} invalid JSON response: {e}")
            raise APIError(f"Invalid JSON response: {e}") from e

    def close(self) -> None:
        """Close the HTTP session."""
        self._session.close()

    def __enter__(self) -> "BaseAPIClient":
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()

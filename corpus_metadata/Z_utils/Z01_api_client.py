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

import hashlib
import json
import threading
import time
from abc import ABC
from pathlib import Path
from typing import Any, Dict, Optional, Union

import requests

from A_core.A00_logging import get_logger

logger = get_logger(__name__)


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

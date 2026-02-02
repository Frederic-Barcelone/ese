# corpus_metadata/tests/test_core/test_api_client.py
"""Tests for Z_utils/Z01_api_client.py - Shared API client utilities."""

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from Z_utils.Z01_api_client import (
    DiskCache,
    RateLimiter,
    BaseAPIClient,
    APIError,
)


class TestDiskCache:
    """Tests for DiskCache class."""

    def test_cache_disabled(self, temp_cache_dir: Path):
        """Test that disabled cache returns None."""
        cache = DiskCache(cache_dir=temp_cache_dir, enabled=False)
        cache.set("key", {"data": "value"})
        assert cache.get("key") is None

    def test_cache_set_and_get(self, temp_cache_dir: Path):
        """Test basic set and get operations."""
        cache = DiskCache(cache_dir=temp_cache_dir, ttl_seconds=3600)
        cache.set("test_key", {"result": "success"})
        result = cache.get("test_key")
        assert result == {"result": "success"}

    def test_cache_miss(self, temp_cache_dir: Path):
        """Test cache miss returns None."""
        cache = DiskCache(cache_dir=temp_cache_dir)
        assert cache.get("nonexistent_key") is None

    def test_cache_expiration(self, temp_cache_dir: Path):
        """Test that expired entries are not returned."""
        cache = DiskCache(cache_dir=temp_cache_dir, ttl_seconds=1)
        cache.set("expiring_key", {"data": "value"})

        # Should be available immediately
        assert cache.get("expiring_key") is not None

        # Wait for expiration
        time.sleep(1.5)
        assert cache.get("expiring_key") is None

    def test_make_key(self, temp_cache_dir: Path):
        """Test cache key generation."""
        cache = DiskCache(cache_dir=temp_cache_dir)
        key1 = cache.make_key("autocomplete", "diabetes")
        key2 = cache.make_key("autocomplete", "DIABETES")  # Case insensitive
        key3 = cache.make_key("search", "diabetes")  # Different prefix

        assert key1 == key2  # Same term, different case
        assert key1 != key3  # Different prefix

    def test_cache_delete(self, temp_cache_dir: Path):
        """Test cache deletion."""
        cache = DiskCache(cache_dir=temp_cache_dir)
        cache.set("to_delete", {"data": "value"})
        assert cache.get("to_delete") is not None

        assert cache.delete("to_delete") is True
        assert cache.get("to_delete") is None

    def test_cache_clear(self, temp_cache_dir: Path):
        """Test clearing all cache entries."""
        cache = DiskCache(cache_dir=temp_cache_dir)
        cache.set("key1", {"a": 1})
        cache.set("key2", {"b": 2})
        cache.set("key3", {"c": 3})

        count = cache.clear()
        assert count == 3
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.get("key3") is None

    def test_cache_creates_directory(self):
        """Test that cache creates directory if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "nested" / "cache"
            _cache = DiskCache(cache_dir=cache_dir)
            assert cache_dir.exists()
            assert _cache is not None  # Verify cache was created


class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_no_rate_limit(self):
        """Test that rate limit of 0 means no limiting."""
        limiter = RateLimiter(requests_per_second=0)
        start = time.time()
        for _ in range(5):
            limiter.wait()
        elapsed = time.time() - start
        assert elapsed < 0.1  # Should be near-instant

    def test_rate_limiting(self):
        """Test that rate limiting enforces delays."""
        limiter = RateLimiter(requests_per_second=10)  # 100ms between requests

        # Make 3 requests
        start = time.time()
        for _ in range(3):
            limiter.wait()
        elapsed = time.time() - start

        # Should take at least 200ms for 3 requests at 10/sec
        assert elapsed >= 0.15  # Allow some tolerance

    def test_wait_returns_delay(self):
        """Test that wait() returns the delay time."""
        limiter = RateLimiter(requests_per_second=5)  # 200ms between requests

        # First request: no wait
        delay1 = limiter.wait()
        assert delay1 == 0.0

        # Immediate second request: should wait
        delay2 = limiter.wait()
        assert delay2 > 0


class TestBaseAPIClient:
    """Tests for BaseAPIClient abstract class."""

    def test_can_instantiate_base_class(self):
        """Test that BaseAPIClient can be instantiated (no abstract methods)."""
        # BaseAPIClient is a concrete class that can be instantiated directly.
        # It provides common functionality; subclasses add specific API methods.
        client = BaseAPIClient(
            config={"base_url": "https://api.example.com"},
            service_name="test",
        )
        assert client.base_url == "https://api.example.com"
        client.close()

    def test_concrete_implementation(self, temp_cache_dir: Path):
        """Test concrete implementation of BaseAPIClient."""

        class TestClient(BaseAPIClient):
            def fetch(self, query: str):
                return self._request("GET", "/test", params={"q": query})

        config = {
            "base_url": "https://api.example.com",
            "timeout_seconds": 10,
            "rate_limit_per_second": 5,
            "cache": {
                "enabled": True,
                "directory": str(temp_cache_dir),
                "ttl_hours": 1,
            },
        }

        client = TestClient(
            config=config,
            service_name="test_service",
        )

        assert client.base_url == "https://api.example.com"
        assert client.timeout == 10
        assert client.cache.enabled is True

    def test_request_success(self):
        """Test successful API request."""

        class TestClient(BaseAPIClient):
            def fetch(self, query: str):
                return self._request("GET", "/test", params={"q": query})

        client = TestClient(
            config={"base_url": "https://api.example.com"},
            service_name="test",
        )

        with patch.object(client._session, "request") as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"result": "success"}
            mock_response.raise_for_status = MagicMock()
            mock_request.return_value = mock_response

            result = client.fetch("test")
            assert result == {"result": "success"}

    def test_request_rate_limit_retry(self):
        """Test that 429 responses trigger retry."""

        class TestClient(BaseAPIClient):
            def fetch(self, query: str):
                return self._request("GET", "/test", params={"q": query})

        client = TestClient(
            config={"base_url": "https://api.example.com"},
            service_name="test",
        )

        call_count = [0]

        def mock_request(*args, **kwargs):
            call_count[0] += 1
            mock_response = MagicMock()
            if call_count[0] == 1:
                mock_response.status_code = 429
            else:
                mock_response.status_code = 200
                mock_response.json.return_value = {"result": "success"}
                mock_response.raise_for_status = MagicMock()
            return mock_response

        with patch.object(client._session, "request", side_effect=mock_request):
            with patch("time.sleep"):  # Skip actual sleep
                result = client.fetch("test")
                assert result == {"result": "success"}
                assert call_count[0] == 2  # Initial + retry

    def test_request_timeout_error(self):
        """Test that timeouts raise APIError."""
        import requests

        class TestClient(BaseAPIClient):
            def fetch(self, query: str):
                return self._request("GET", "/test", params={"q": query})

        client = TestClient(
            config={"base_url": "https://api.example.com"},
            service_name="test",
        )

        with patch.object(
            client._session,
            "request",
            side_effect=requests.exceptions.Timeout("Connection timed out"),
        ):
            with pytest.raises(APIError) as exc_info:
                client.fetch("test")
            assert "timeout" in str(exc_info.value).lower()

    def test_context_manager(self, temp_cache_dir: Path):
        """Test using client as context manager."""

        class TestClient(BaseAPIClient):
            pass

        with TestClient(
            config={"base_url": "https://api.example.com"},
            service_name="test",
        ) as client:
            assert client is not None

        # Session should be closed after exiting context
        assert client._session is not None  # Object still exists

# corpus_metadata/K_tests/K51_test_provenance.py
"""
Tests for A_core.A03_provenance module.

Tests hashing, fingerprinting, and run ID generation functions.
"""

from __future__ import annotations

import re


from A_core.A03_provenance import (
    get_git_revision_hash,
    hash_bytes,
    hash_string,
    hash_short,
    compute_prompt_hash,
    compute_prompt_bundle_hash,
    compute_context_hash,
    compute_doc_fingerprint,
    generate_run_id,
)


class TestHashBytes:
    """Tests for hash_bytes function."""

    def test_empty_bytes(self):
        result = hash_bytes(b"")
        # SHA256 of empty bytes is a known constant
        assert result == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_simple_bytes(self):
        result = hash_bytes(b"hello")
        # SHA256 produces 64-char hex string
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_deterministic(self):
        data = b"test data"
        assert hash_bytes(data) == hash_bytes(data)

    def test_different_input_different_hash(self):
        assert hash_bytes(b"hello") != hash_bytes(b"world")


class TestHashString:
    """Tests for hash_string function."""

    def test_empty_string(self):
        result = hash_string("")
        # Same as hash_bytes(b"")
        assert result == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_simple_string(self):
        result = hash_string("hello")
        assert len(result) == 64

    def test_unicode_string(self):
        result = hash_string("hello \u00e4\u00f6\u00fc")
        assert len(result) == 64

    def test_deterministic(self):
        text = "some test text"
        assert hash_string(text) == hash_string(text)


class TestHashShort:
    """Tests for hash_short function."""

    def test_default_length(self):
        full_hash = "abcdef1234567890abcdef1234567890"
        result = hash_short(full_hash)
        assert result == "abcdef123456"
        assert len(result) == 12

    def test_custom_length(self):
        full_hash = "abcdef1234567890"
        result = hash_short(full_hash, n=6)
        assert result == "abcdef"

    def test_empty_hash(self):
        result = hash_short("")
        assert result == ""

    def test_none_hash(self):
        result = hash_short(None)  # type: ignore[arg-type]
        assert result == ""


class TestComputePromptHash:
    """Tests for compute_prompt_hash function."""

    def test_basic_prompt_hash(self):
        result = compute_prompt_hash(
            system_prompt="You are a helpful assistant.",
            user_template="Extract entities from: {text}",
            parameters={"text": "sample"}
        )
        assert len(result) == 64

    def test_deterministic(self):
        result1 = compute_prompt_hash("sys", "user", {"key": "value"})
        result2 = compute_prompt_hash("sys", "user", {"key": "value"})
        assert result1 == result2

    def test_different_params_different_hash(self):
        hash1 = compute_prompt_hash("sys", "user", {"a": 1})
        hash2 = compute_prompt_hash("sys", "user", {"a": 2})
        assert hash1 != hash2

    def test_param_order_independent(self):
        """Sorted keys should produce same hash regardless of insertion order."""
        hash1 = compute_prompt_hash("sys", "user", {"a": 1, "b": 2})
        hash2 = compute_prompt_hash("sys", "user", {"b": 2, "a": 1})
        assert hash1 == hash2


class TestComputePromptBundleHash:
    """Tests for compute_prompt_bundle_hash function."""

    def test_with_schema(self):
        result = compute_prompt_bundle_hash(
            system_prompt="sys",
            user_template="user",
            schema={"type": "object", "properties": {}},
            parameters={},
        )
        assert len(result) == 64

    def test_without_schema(self):
        result = compute_prompt_bundle_hash(
            system_prompt="sys",
            user_template="user",
            schema=None,
            parameters={},
        )
        assert len(result) == 64

    def test_schema_affects_hash(self):
        hash1 = compute_prompt_bundle_hash("sys", "user", {"type": "string"}, {})
        hash2 = compute_prompt_bundle_hash("sys", "user", {"type": "number"}, {})
        assert hash1 != hash2


class TestComputeContextHash:
    """Tests for compute_context_hash function."""

    def test_basic_context(self):
        result = compute_context_hash("Some context text for verification.")
        assert len(result) == 64

    def test_empty_context(self):
        result = compute_context_hash("")
        assert len(result) == 64
        assert result == hash_string("")

    def test_none_context(self):
        result = compute_context_hash(None)  # type: ignore[arg-type]
        assert result == hash_string("")


class TestComputeDocFingerprint:
    """Tests for compute_doc_fingerprint function."""

    def test_basic_fingerprint(self):
        result = compute_doc_fingerprint(b"%PDF-1.4 fake pdf content")
        assert len(result) == 64

    def test_empty_bytes(self):
        result = compute_doc_fingerprint(b"")
        assert result == hash_bytes(b"")

    def test_none_bytes(self):
        result = compute_doc_fingerprint(None)  # type: ignore[arg-type]
        assert result == hash_bytes(b"")


class TestGenerateRunId:
    """Tests for generate_run_id function."""

    def test_default_prefix(self):
        result = generate_run_id()
        assert result.startswith("RUN_")

    def test_custom_prefix(self):
        result = generate_run_id(prefix="BATCH")
        assert result.startswith("BATCH_")

    def test_format(self):
        result = generate_run_id()
        # Format: PREFIX_YYYYMMDD_HHMMSS_uuid12
        pattern = r"^RUN_\d{8}_\d{6}_[a-f0-9]{12}$"
        assert re.match(pattern, result), f"Run ID format mismatch: {result}"

    def test_unique(self):
        ids = [generate_run_id() for _ in range(10)]
        assert len(set(ids)) == 10  # All unique


class TestGetGitRevisionHash:
    """Tests for get_git_revision_hash function."""

    def test_returns_string(self):
        result = get_git_revision_hash()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_commit_or_unknown(self):
        result = get_git_revision_hash()
        # Should be a 40-char hex string or "unknown-version"
        is_commit = len(result) == 40 and all(c in "0123456789abcdef" for c in result)
        is_unknown = result == "unknown-version"
        assert is_commit or is_unknown

    def test_timeout_respected(self):
        # Very short timeout should still return something
        result = get_git_revision_hash(timeout_sec=0.001)
        assert isinstance(result, str)
        assert len(result) > 0

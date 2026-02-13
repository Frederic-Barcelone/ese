# corpus_metadata/K_tests/K32_test_prompt_registry.py
"""
Tests for D_validation.D01_prompt_registry module.

Tests prompt registry, task enums, and prompt bundle generation.
"""

from __future__ import annotations

import pytest

from D_validation.D01_prompt_registry import (
    PromptTask,
    PromptBundle,
    PromptRegistry,
)


class TestPromptTask:
    """Tests for PromptTask enum."""

    def test_task_values(self):
        assert PromptTask.VERIFY_DEFINITION_PAIR.value == "verify_definition_pair"
        assert PromptTask.VERIFY_SHORT_FORM_ONLY.value == "verify_short_form_only"
        assert PromptTask.VERIFY_BATCH.value == "verify_batch"
        assert PromptTask.FAST_REJECT.value == "fast_reject"

    def test_disease_tasks(self):
        assert PromptTask.VERIFY_DISEASE.value == "verify_disease"
        assert PromptTask.VERIFY_DISEASE_BATCH.value == "verify_disease_batch"

    def test_author_citation_tasks(self):
        assert PromptTask.VERIFY_AUTHOR_BATCH.value == "verify_author_batch"
        assert PromptTask.VERIFY_CITATION_BATCH.value == "verify_citation_batch"


class TestPromptBundle:
    """Tests for PromptBundle model."""

    def test_bundle_attributes(self):
        bundle = PromptBundle(
            task=PromptTask.VERIFY_BATCH,
            version="v1.0",
            system_prompt="You are a validator.",
            user_template="Validate: {candidates}",
            output_schema=None,
            prompt_bundle_hash="abc123",
        )
        assert bundle.task == PromptTask.VERIFY_BATCH
        assert bundle.version == "v1.0"
        assert "validator" in bundle.system_prompt
        assert "{candidates}" in bundle.user_template

    def test_bundle_immutable(self):
        bundle = PromptBundle(
            task=PromptTask.VERIFY_BATCH,
            version="v1.0",
            system_prompt="test",
            user_template="test",
            prompt_bundle_hash="abc",
        )
        with pytest.raises(Exception):  # Frozen model
            bundle.version = "v2.0"


class TestPromptRegistry:
    """Tests for PromptRegistry class."""

    def test_get_bundle_latest(self):
        bundle = PromptRegistry.get_bundle(PromptTask.VERIFY_BATCH, version="latest")
        assert bundle.task == PromptTask.VERIFY_BATCH
        assert bundle.version == "v2.0"  # Latest version

    def test_get_bundle_specific_version(self):
        bundle = PromptRegistry.get_bundle(
            PromptTask.VERIFY_DEFINITION_PAIR, version="v1.0"
        )
        assert bundle.version == "v1.0"
        assert "clinical document QA auditor" in bundle.system_prompt

    def test_get_bundle_v12(self):
        bundle = PromptRegistry.get_bundle(
            PromptTask.VERIFY_DEFINITION_PAIR, version="v1.2"
        )
        assert bundle.version == "v1.2"
        # "lean toward VALIDATED" is in the system prompt, not user template
        assert "lean toward VALIDATED" in bundle.system_prompt

    def test_bundle_has_hash(self):
        bundle = PromptRegistry.get_bundle(PromptTask.VERIFY_BATCH)
        assert bundle.prompt_bundle_hash
        assert len(bundle.prompt_bundle_hash) > 10

    def test_invalid_version_raises(self):
        with pytest.raises(ValueError):
            PromptRegistry.get_bundle(PromptTask.VERIFY_BATCH, version="v99.0")

    def test_disease_batch_bundle(self):
        bundle = PromptRegistry.get_bundle(PromptTask.VERIFY_DISEASE_BATCH)
        assert "disease" in bundle.system_prompt.lower()
        assert "{candidates}" in bundle.user_template

    def test_fast_reject_bundle(self):
        bundle = PromptRegistry.get_bundle(PromptTask.FAST_REJECT)
        assert "REJECT" in bundle.user_template
        assert "REVIEW" in bundle.user_template

    def test_author_batch_bundle(self):
        bundle = PromptRegistry.get_bundle(PromptTask.VERIFY_AUTHOR_BATCH)
        assert "author" in bundle.system_prompt.lower()

    def test_citation_batch_bundle(self):
        bundle = PromptRegistry.get_bundle(PromptTask.VERIFY_CITATION_BATCH)
        assert "citation" in bundle.system_prompt.lower()


class TestPromptTemplates:
    """Tests for prompt template content."""

    def test_batch_v2_has_output_contract(self):
        bundle = PromptRegistry.get_bundle(PromptTask.VERIFY_BATCH, version="v2.0")
        assert "OUTPUT CONTRACT" in bundle.user_template
        assert "expected_count" in bundle.user_template

    def test_definition_pair_has_json_schema(self):
        bundle = PromptRegistry.get_bundle(PromptTask.VERIFY_DEFINITION_PAIR)
        assert "status" in bundle.user_template
        assert "VALIDATED" in bundle.user_template
        assert "REJECTED" in bundle.user_template
        assert "AMBIGUOUS" in bundle.user_template

    def test_short_form_only_no_guess(self):
        bundle = PromptRegistry.get_bundle(PromptTask.VERIFY_SHORT_FORM_ONLY)
        assert "Do NOT guess" in bundle.system_prompt or "Do NOT invent" in bundle.user_template


class TestLatestVersions:
    """Tests for _LATEST version mapping."""

    def test_all_tasks_have_latest(self):
        for task in PromptTask:
            assert task in PromptRegistry._LATEST
            version = PromptRegistry._LATEST[task]
            # Should be able to get bundle with this version
            bundle = PromptRegistry.get_bundle(task, version=version)
            assert bundle is not None

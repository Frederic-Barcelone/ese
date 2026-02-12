# corpus_metadata/K_tests/K33_test_llm_engine.py
"""
Tests for D_validation.D02_llm_engine module.

Tests LLM engine, Claude client, and verification result handling.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from unittest.mock import MagicMock, patch

from D_validation.D02_llm_engine import (
    ClaudeClient,
    LLMEngine,
    VerificationResult,
    MODEL_PRICING,
)
from A_core.A01_domain_models import ValidationStatus
from Z_utils.Z13_llm_tracking import CallType, resolve_model_tier


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


class TestModelTierRouting:
    """Tests for model tier routing — ensures Haiku call types use Haiku, not Sonnet."""

    # Use CallType registry as single source of truth (stays in sync automatically)
    HAIKU_CALL_TYPES = sorted(CallType.HAIKU_CALL_TYPES)
    SONNET_CALL_TYPES = sorted(CallType.SONNET_CALL_TYPES)

    def test_no_config_path_means_empty_model_tiers(self):
        """ClaudeClient with config_path=None has no model tier routing."""
        with patch("D_validation.D02_llm_engine.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value = MagicMock()
            client = ClaudeClient(api_key="test-key", config_path=None)
            assert client._model_tiers == {}

    def test_config_path_loads_model_tiers(self):
        """ClaudeClient with real config.yaml path loads model_tiers."""
        from pathlib import Path
        config_path = str(
            Path(__file__).parent.parent / "G_config" / "config.yaml"
        )
        with patch("D_validation.D02_llm_engine.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value = MagicMock()
            client = ClaudeClient(api_key="test-key", config_path=config_path)
            assert len(client._model_tiers) > 0, (
                "model_tiers should be populated from config.yaml"
            )

    def test_haiku_call_types_resolve_to_haiku(self):
        """All Haiku-designated call types must resolve to a Haiku model."""
        from pathlib import Path
        config_path = str(
            Path(__file__).parent.parent / "G_config" / "config.yaml"
        )
        with patch("D_validation.D02_llm_engine.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value = MagicMock()
            client = ClaudeClient(api_key="test-key", config_path=config_path)

            for call_type in self.HAIKU_CALL_TYPES:
                resolved = client.resolve_model(call_type)
                assert "haiku" in resolved.lower(), (
                    f"call_type '{call_type}' should route to Haiku, "
                    f"got '{resolved}'"
                )

    def test_sonnet_call_types_resolve_to_sonnet(self):
        """All Sonnet-designated call types must resolve to a Sonnet model."""
        from pathlib import Path
        config_path = str(
            Path(__file__).parent.parent / "G_config" / "config.yaml"
        )
        with patch("D_validation.D02_llm_engine.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value = MagicMock()
            client = ClaudeClient(api_key="test-key", config_path=config_path)

            for call_type in self.SONNET_CALL_TYPES:
                resolved = client.resolve_model(call_type)
                assert "sonnet" in resolved.lower(), (
                    f"call_type '{call_type}' should route to Sonnet, "
                    f"got '{resolved}'"
                )

    def test_factory_creates_client_with_model_tiers(self):
        """ComponentFactory.create_claude_client() must produce a client with model_tiers."""
        from pathlib import Path
        from H_pipeline.H01_component_factory import ComponentFactory
        import yaml

        config_path = Path(__file__).parent.parent / "G_config" / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        with patch("D_validation.D02_llm_engine.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value = MagicMock()
            factory = ComponentFactory(
                config=config,
                run_id="TEST",
                pipeline_version="0.8",
                log_dir=Path("/tmp"),
                api_key="test-key",
            )
            client = factory.create_claude_client()
            assert client is not None
            assert len(client._model_tiers) > 0, (
                "ComponentFactory must pass config_path so model_tiers are loaded"
            )
            # Verify a Haiku call type actually routes to Haiku
            resolved = client.resolve_model("fast_reject")
            assert "haiku" in resolved.lower(), (
                f"fast_reject should route to Haiku via factory client, got '{resolved}'"
            )

    def test_call_claude_uses_tier_model(self):
        """_call_claude must use the tier-mapped model, not the default."""
        from pathlib import Path
        config_path = str(
            Path(__file__).parent.parent / "G_config" / "config.yaml"
        )
        with patch("D_validation.D02_llm_engine.anthropic") as mock_anthropic:
            mock_messages = MagicMock()
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text='{"result": "ok"}')]
            mock_response.usage = MagicMock(
                input_tokens=10, output_tokens=5,
                cache_read_input_tokens=0, cache_creation_input_tokens=0,
            )
            mock_messages.create.return_value = mock_response
            mock_anthropic.Anthropic.return_value = MagicMock(
                messages=mock_messages
            )

            client = ClaudeClient(api_key="test-key", config_path=config_path)
            # Call with a Haiku call_type
            client._call_claude(
                "system", "user", None, None, None, 1.0,
                call_type="fast_reject",
            )
            # Verify the actual model passed to the API
            call_args = mock_messages.create.call_args
            actual_model = call_args.kwargs.get("model", call_args[1].get("model", ""))
            assert "haiku" in actual_model.lower(), (
                f"API call for fast_reject should use Haiku model, got '{actual_model}'"
            )

    def test_model_ids_are_valid(self):
        """All model IDs in config.yaml must be from the known valid set."""
        from pathlib import Path
        import yaml

        # Known valid Anthropic model IDs (update when new models are released)
        VALID_MODEL_IDS = {
            # Haiku
            "claude-3-5-haiku-20241022",
            "claude-haiku-4-5-20251001",
            # Sonnet
            "claude-sonnet-4-20250514",
            "claude-sonnet-4-5-20250929",
            # Opus
            "claude-opus-4-20250514",
        }

        config_path = Path(__file__).parent.parent / "G_config" / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        model_tiers = config.get("api", {}).get("claude", {}).get("model_tiers", {})
        assert len(model_tiers) > 0, "model_tiers should not be empty"

        for call_type, model_id in model_tiers.items():
            assert model_id in VALID_MODEL_IDS, (
                f"model_tiers['{call_type}'] = '{model_id}' is not a valid model ID. "
                f"Valid IDs: {sorted(VALID_MODEL_IDS)}"
            )

    def test_haiku_call_does_not_pass_both_temperature_and_top_p(self):
        """Haiku 4.5 rejects temperature + top_p together. Verify we don't send both."""
        from pathlib import Path
        config_path = str(
            Path(__file__).parent.parent / "G_config" / "config.yaml"
        )
        with patch("D_validation.D02_llm_engine.anthropic") as mock_anthropic:
            mock_messages = MagicMock()
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text='{"result": "ok"}')]
            mock_response.usage = MagicMock(
                input_tokens=10, output_tokens=5,
                cache_read_input_tokens=0, cache_creation_input_tokens=0,
            )
            mock_messages.create.return_value = mock_response
            mock_anthropic.Anthropic.return_value = MagicMock(
                messages=mock_messages
            )

            client = ClaudeClient(api_key="test-key", config_path=config_path)
            # Call with default top_p=1.0 (the normal case)
            client._call_claude(
                "system", "user", None, None, None, 1.0,
                call_type="fast_reject",
            )
            call_kwargs = mock_messages.create.call_args.kwargs
            has_temp = "temperature" in call_kwargs
            has_top_p = "top_p" in call_kwargs
            assert not (has_temp and has_top_p), (
                f"API call must not pass both temperature and top_p. "
                f"Got temperature={has_temp}, top_p={has_top_p}"
            )

    def test_all_call_types_in_config(self):
        """Every CallType.ALL_CALL_TYPES member must have an entry in config.yaml model_tiers."""
        config_path = Path(__file__).parent.parent / "G_config" / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        model_tiers = config.get("api", {}).get("claude", {}).get("model_tiers", {})
        missing = CallType.ALL_CALL_TYPES - set(model_tiers.keys())
        assert not missing, (
            f"config.yaml model_tiers is missing entries for: {sorted(missing)}"
        )

    def test_config_has_no_unknown_call_types(self):
        """Every key in config.yaml model_tiers must be a known CallType (catches typos)."""
        config_path = Path(__file__).parent.parent / "G_config" / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        model_tiers = config.get("api", {}).get("claude", {}).get("model_tiers", {})
        unknown = set(model_tiers.keys()) - CallType.ALL_CALL_TYPES
        assert not unknown, (
            f"config.yaml model_tiers has unknown call types (typo?): {sorted(unknown)}"
        )

    def test_all_code_call_types_are_registered(self):
        """Scan all .py files for call_type=... patterns and verify each is in CallType."""
        corpus_dir = Path(__file__).parent.parent
        call_type_pattern = re.compile(r'call_type\s*=\s*["\']([a-z_]+)["\']')
        unregistered = set()

        for py_file in corpus_dir.rglob("*.py"):
            if "K_tests" in str(py_file):
                continue
            text = py_file.read_text(encoding="utf-8", errors="ignore")
            for match in call_type_pattern.finditer(text):
                ct = match.group(1)
                if ct not in CallType.ALL_CALL_TYPES and ct not in CallType._DEFAULT_CALL_TYPES:
                    unregistered.add(ct)

        assert not unregistered, (
            f"call_type values found in code but not in CallType registry: {sorted(unregistered)}"
        )

    def test_default_call_types_trigger_warning(self):
        """Calling _call_claude with a default call_type (e.g. 'json') should log a warning."""
        config_path = str(
            Path(__file__).parent.parent / "G_config" / "config.yaml"
        )
        with patch("D_validation.D02_llm_engine.anthropic") as mock_anthropic:
            mock_messages = MagicMock()
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text='{"result": "ok"}')]
            mock_response.usage = MagicMock(
                input_tokens=10, output_tokens=5,
                cache_read_input_tokens=0, cache_creation_input_tokens=0,
            )
            mock_messages.create.return_value = mock_response
            mock_anthropic.Anthropic.return_value = MagicMock(
                messages=mock_messages
            )

            client = ClaudeClient(api_key="test-key", config_path=config_path)
            with patch("D_validation.D02_llm_engine.logger") as mock_logger:
                client._call_claude(
                    "system", "user", None, None, None, 1.0,
                    call_type="json",
                )
                mock_logger.warning.assert_called()
                warning_msg = str(mock_logger.warning.call_args)
                assert "json" in warning_msg and "default placeholder" in warning_msg

    def test_empty_model_tiers_triggers_warning(self):
        """ClaudeClient(config_path=None) should log a warning about empty tiers."""
        with patch("D_validation.D02_llm_engine.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value = MagicMock()
            with patch("D_validation.D02_llm_engine.logger") as mock_logger:
                ClaudeClient(api_key="test-key", config_path=None)
                mock_logger.warning.assert_called()
                warning_msg = str(mock_logger.warning.call_args)
                assert "empty" in warning_msg.lower()

    def test_vision_call_uses_tier_model(self):
        """complete_vision_json with call_type='vlm_visual_enrichment' should use Haiku."""
        config_path = str(
            Path(__file__).parent.parent / "G_config" / "config.yaml"
        )
        with patch("D_validation.D02_llm_engine.anthropic") as mock_anthropic:
            mock_messages = MagicMock()
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text='{"result": "ok"}')]
            mock_response.usage = MagicMock(
                input_tokens=10, output_tokens=5,
                cache_read_input_tokens=0, cache_creation_input_tokens=0,
            )
            mock_messages.create.return_value = mock_response
            mock_anthropic.Anthropic.return_value = MagicMock(
                messages=mock_messages
            )

            client = ClaudeClient(api_key="test-key", config_path=config_path)
            # Tiny valid PNG (1x1 pixel)
            import base64
            tiny_png = base64.b64encode(
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
                b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
                b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
                b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
            ).decode()
            client.complete_vision_json(
                image_base64=tiny_png,
                prompt="test",
                call_type="vlm_visual_enrichment",
            )
            call_args = mock_messages.create.call_args
            actual_model = call_args.kwargs.get("model", "")
            assert "haiku" in actual_model.lower(), (
                f"Vision call for vlm_visual_enrichment should use Haiku, got '{actual_model}'"
            )

    def test_resolve_model_tier_standalone(self):
        """Test Z13.resolve_model_tier() resolves correctly for known call types."""
        import Z_utils.Z13_llm_tracking as z13
        # Reset cache to force reload
        z13._model_tier_cache = None

        for ct in CallType.HAIKU_CALL_TYPES:
            resolved = resolve_model_tier(ct)
            assert "haiku" in resolved.lower(), (
                f"resolve_model_tier('{ct}') should return Haiku model, got '{resolved}'"
            )

        for ct in CallType.SONNET_CALL_TYPES:
            resolved = resolve_model_tier(ct)
            assert "sonnet" in resolved.lower(), (
                f"resolve_model_tier('{ct}') should return Sonnet model, got '{resolved}'"
            )

    def test_model_pricing_covers_all_configured_models(self):
        """Every model ID in config.yaml model_tiers must have an entry in MODEL_PRICING."""
        config_path = Path(__file__).parent.parent / "G_config" / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        model_tiers = config.get("api", {}).get("claude", {}).get("model_tiers", {})
        configured_models = set(model_tiers.values())
        missing = configured_models - set(MODEL_PRICING.keys())
        assert not missing, (
            f"MODEL_PRICING is missing entries for configured models: {sorted(missing)}"
        )

    def test_author_extraction_routes_to_haiku(self):
        """author_extraction must be in CallType.HAIKU_CALL_TYPES and route to Haiku."""
        assert "author_extraction" in CallType.HAIKU_CALL_TYPES
        config_path = str(
            Path(__file__).parent.parent / "G_config" / "config.yaml"
        )
        with patch("D_validation.D02_llm_engine.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value = MagicMock()
            client = ClaudeClient(api_key="test-key", config_path=config_path)
            resolved = client.resolve_model("author_extraction")
            assert "haiku" in resolved.lower(), (
                f"author_extraction should route to Haiku, got '{resolved}'"
            )

    def test_haiku_tier_cost_lower_than_sonnet(self):
        """All Haiku models in MODEL_PRICING must be cheaper than all Sonnet models."""
        haiku_models = {m for m in MODEL_PRICING if "haiku" in m.lower()}
        sonnet_models = {m for m in MODEL_PRICING if "sonnet" in m.lower()}
        assert haiku_models, "No Haiku models found in MODEL_PRICING"
        assert sonnet_models, "No Sonnet models found in MODEL_PRICING"

        max_haiku_input = max(MODEL_PRICING[m][0] for m in haiku_models)
        min_sonnet_input = min(MODEL_PRICING[m][0] for m in sonnet_models)
        assert max_haiku_input < min_sonnet_input, (
            f"Haiku input price ({max_haiku_input}) should be less than "
            f"Sonnet input price ({min_sonnet_input})"
        )

        max_haiku_output = max(MODEL_PRICING[m][1] for m in haiku_models)
        min_sonnet_output = min(MODEL_PRICING[m][1] for m in sonnet_models)
        assert max_haiku_output < min_sonnet_output, (
            f"Haiku output price ({max_haiku_output}) should be less than "
            f"Sonnet output price ({min_sonnet_output})"
        )

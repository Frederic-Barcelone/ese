"""
LLM engine for candidate validation using Claude API.

This module provides the core validation engine that uses Claude to verify
extracted candidates. Implements a protocol-based design allowing different
LLM backends, with built-in retry logic, rate limiting, and structured output parsing.

Key Components:
    - LLMClient: Protocol (interface) for LLM clients
    - ClaudeClient: Anthropic Claude API client implementation
    - LLMEngine: Main verifier that uses any LLMClient for validation
    - VerificationResult: Pydantic schema for structured validation responses
    - Batch validation support with configurable concurrency

Example:
    >>> from D_validation.D02_llm_engine import LLMEngine, ClaudeClient
    >>> client = ClaudeClient(model="claude-sonnet-4-20250514")
    >>> engine = LLMEngine(client=client)
    >>> results = engine.verify_batch(candidates, doc_context)
    >>> for result in results:
    ...     print(f"{result.candidate_id}: {result.status}")
    abbr_001: VALIDATED

Dependencies:
    - A_core.A01_domain_models: Candidate, ExtractedEntity, ValidationStatus
    - A_core.A03_provenance: Provenance tracking utilities
    - D_validation.D01_prompt_registry: PromptRegistry, PromptTask
    - Z_utils.Z04_image_utils: Image compression for vision tasks
    - anthropic: Claude API client (optional)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

import yaml
from pydantic import BaseModel, Field, ValidationError

from Z_utils.Z13_llm_tracking import (
    MODEL_PRICING,
    LLMUsageRecord,
    LLMUsageTracker,
    calc_record_cost,
    get_usage_tracker,
    record_api_usage,
    resolve_model_tier,
)
from A_core.A01_domain_models import (
    Candidate,
    ExtractedEntity,
    EvidenceSpan,
    FieldType,
    LLMParameters,
    ProvenanceMetadata,
    ValidationStatus,
)
from A_core.A03_provenance import (
    generate_run_id,
    get_git_revision_hash,
    hash_string,
)
from D_validation.D01_prompt_registry import (
    PromptRegistry,
    PromptTask,
)
from Z_utils.Z04_image_utils import (
    MAX_VISION_IMAGE_SIZE_BYTES,
    compress_image_for_vision,
    get_image_size_bytes,
)

logger = logging.getLogger(__name__)

# Optional anthropic import
try:
    import anthropic
    from anthropic import APIConnectionError as AnthropicConnectionError
    from anthropic import APIError as AnthropicAPIError
    from anthropic import APIStatusError as AnthropicStatusError
    from anthropic import APITimeoutError as AnthropicTimeoutError
    from anthropic import RateLimitError as AnthropicRateLimitError
except ImportError:
    anthropic = None  # type: ignore
    AnthropicAPIError = Exception  # type: ignore
    AnthropicConnectionError = Exception  # type: ignore
    AnthropicTimeoutError = Exception  # type: ignore
    AnthropicRateLimitError = Exception  # type: ignore
    AnthropicStatusError = Exception  # type: ignore


# =============================================================================
# PROTOCOL
# =============================================================================


class LLMClient(Protocol):
    """
    Vendor-agnostic interface for LLM clients.
    Implementations should return a dict already parsed from JSON.
    """

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        seed: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        call_type: str = "json",
    ) -> Dict[str, Any]: ...

    def complete_json_any(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        seed: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        call_type: str = "json_any",
    ) -> Any: ...


# =============================================================================
# CLAUDE CLIENT
# =============================================================================


class ClaudeClient:
    """
    Anthropic Claude client implementing LLMClient protocol.

    Reads config from:
      1. Explicit parameters
      2. config.yaml (if config_path provided)
      3. Environment variables (ANTHROPIC_API_KEY)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        config_path: Optional[str] = None,
        usage_tracker: Optional[LLMUsageTracker] = None,
    ):
        if anthropic is None:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            )

        # Load config from YAML if provided
        cfg = self._load_config(config_path) if config_path else {}

        # Resolve API key: param > config > env
        self.api_key = (
            api_key or cfg.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY env var, "
                "pass api_key param, or configure in config.yaml"
            )

        # Resolve model params: param > config > defaults
        self.default_model = model or cfg.get("model", "claude-sonnet-4-20250514")
        self.default_max_tokens = max_tokens or cfg.get("max_tokens", 1024)
        self.default_temperature = (
            temperature if temperature is not None else cfg.get("temperature", 0.0)
        )

        # Initialize client
        self.client = anthropic.Anthropic(api_key=self.api_key)

        # Token usage tracker
        self.usage_tracker = usage_tracker or get_usage_tracker()

        # Model tier routing (call_type -> model)
        self._model_tiers: Dict[str, str] = cfg.get("model_tiers", {})

    @property
    def messages(self):
        """Provide access to the messages API for direct usage."""
        return self.client.messages

    def resolve_model(self, call_type: str) -> str:
        """Resolve the model to use for a given call type based on config tiers.

        Falls back to self.default_model if no tier mapping exists.
        """
        return self._model_tiers.get(call_type, self.default_model)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load Claude config from YAML file."""
        path = Path(config_path)
        if not path.exists():
            return {}

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            claude_cfg = data.get("api", {}).get("claude", {})
            val_cfg = claude_cfg.get("validation", {}) or claude_cfg.get("fast", {})

            return {
                "api_key": claude_cfg.get("api_key"),
                "model": val_cfg.get("model"),
                "max_tokens": val_cfg.get("max_tokens"),
                "temperature": val_cfg.get("temperature"),
                "model_tiers": claude_cfg.get("model_tiers", {}),
            }
        except (OSError, IOError, yaml.YAMLError, KeyError, TypeError) as e:
            logger.warning("Failed to load config from %s: %s", config_path, e)
            return {}

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        seed: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        call_type: str = "json",
    ) -> Dict[str, Any]:
        """Call Claude and return parsed JSON object response."""
        raw_text = self._call_claude(
            system_prompt, user_prompt, model, temperature, max_tokens, top_p,
            call_type=call_type,
        )
        return self._extract_json_object(raw_text)

    def complete_json_any(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        seed: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        call_type: str = "json_any",
    ) -> Any:
        """Call Claude and return parsed JSON (object or array)."""
        raw_text = self._call_claude(
            system_prompt, user_prompt, model, temperature, max_tokens, top_p,
            call_type=call_type,
        )
        return self._extract_json_any(raw_text)

    def complete_vision_json(
        self,
        *,
        image_base64: str,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.0,
        auto_compress: bool = True,
        max_image_size: int = MAX_VISION_IMAGE_SIZE_BYTES,
        call_type: str = "vision",
    ) -> Optional[Dict[str, Any]]:
        """
        Call Claude Vision API with an image and return parsed JSON response.
        Automatically handles images exceeding the 5MB limit by compressing them.
        """
        # Model tier routing: if call_type has a tier mapping, use that model
        use_model = self._model_tiers.get(call_type, model or self.default_model)

        # Check image size and compress if needed
        image_size = get_image_size_bytes(image_base64)
        if image_size > max_image_size:
            if auto_compress:
                logger.info(
                    "Image exceeds %.1fMB limit (%.1fMB), compressing...",
                    max_image_size / 1024 / 1024,
                    image_size / 1024 / 1024,
                )
                compressed, info = compress_image_for_vision(
                    image_base64, max_size_bytes=max_image_size
                )
                if compressed:
                    image_base64 = compressed
                    logger.info(
                        "Compressed: %.1fMB -> %.1fMB (ratio: %.1fx)",
                        info["original_size"] / 1024 / 1024,
                        info["final_size"] / 1024 / 1024,
                        info["compression_ratio"],
                    )
                else:
                    logger.warning("Could not compress image: %s", info.get("error"))
                    return None
            else:
                logger.warning("Image exceeds %.1fMB limit, skipping", max_image_size / 1024 / 1024)
                return None

        # Detect media type
        media_type = self._detect_media_type(image_base64)

        message = self.client.messages.create(
            model=use_model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {  # type: ignore[list-item]
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_base64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )

        # Track token usage
        self._record_usage(message, use_model, call_type)

        raw_text = self._extract_text_from_message(message)
        return self._extract_json_any(raw_text)

    def _call_claude(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
        top_p: float,
        call_type: str = "unknown",
    ) -> str:
        """Make Claude API call and return raw text response."""
        # Model tier routing: if call_type has a tier mapping, use that model
        use_model = self._model_tiers.get(call_type, model or self.default_model)
        use_max_tokens = max_tokens or self.default_max_tokens
        use_temperature = temperature if temperature is not None else self.default_temperature

        # Build system message with prompt caching
        system_messages: Any = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]

        # Build API kwargs — Haiku 4.5 rejects temperature + top_p together
        api_kwargs: Dict[str, Any] = {
            "model": use_model,
            "max_tokens": use_max_tokens,
            "temperature": use_temperature,
            "system": system_messages,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if top_p != 1.0:
            api_kwargs.pop("temperature", None)
            api_kwargs["top_p"] = top_p

        message = self.client.messages.create(**api_kwargs)

        # Track token usage
        self._record_usage(message, use_model, call_type)

        return self._extract_text_from_message(message)

    def _record_usage(self, message, model: str, call_type: str) -> None:
        """Record token usage from a Claude API response."""
        if hasattr(message, "usage") and message.usage:
            usage = message.usage
            cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
            cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
            self.usage_tracker.record(
                model=model,
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
                call_type=call_type,
                cache_read_tokens=cache_read,
                cache_creation_tokens=cache_creation,
            )

    def _extract_text_from_message(self, message) -> str:
        """Extract text content from Claude message response."""
        raw_text = ""
        for block in message.content:
            if hasattr(block, "text"):
                raw_text += block.text
        return raw_text

    def _detect_media_type(self, image_base64: str) -> str:
        """Detect image media type from base64 header."""
        if image_base64.startswith("/9j/"):
            return "image/jpeg"
        elif image_base64.startswith("R0lGOD"):
            return "image/gif"
        elif image_base64.startswith("UklGR"):
            return "image/webp"
        return "image/png"

    def _find_balanced(self, text: str, open_ch: str, close_ch: str) -> Optional[str]:
        """Find balanced brackets/braces and return the matched substring."""
        start = text.find(open_ch)
        if start == -1:
            return None
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

    def _extract_json_any(self, text: str) -> Any:
        """Extract JSON from Claude's response - handles both objects and arrays."""
        text = (text or "").strip()

        # Try markdown code block
        code_block_match = re.search(
            r"```(?:json)?\s*([\[\{][\s\S]*?[\]\}])\s*```", text, re.DOTALL
        )
        if code_block_match:
            try:
                return json.loads(code_block_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try balanced array first, then object
        for open_ch, close_ch in [("[", "]"), ("{", "}")]:
            matched = self._find_balanced(text, open_ch, close_ch)
            if matched:
                try:
                    return json.loads(matched)
                except json.JSONDecodeError:
                    pass

        # Try entire response
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.debug("JSON parse failed. Preview: %r", text[:200])
            return None

    def _extract_json_object(self, text: str) -> Dict[str, Any]:
        """Extract JSON object from Claude's response with fallback."""
        parsed = self._extract_json_any(text)

        if isinstance(parsed, dict) and "status" in parsed:
            return parsed

        return {
            "status": "AMBIGUOUS",
            "confidence": 0.0,
            "evidence": "",
            "reason": f"Failed to parse JSON: {(text or '')[:200]}",
            "corrected_long_form": None,
        }


# =============================================================================
# VERIFICATION RESULT SCHEMA
# =============================================================================


class VerificationResult(BaseModel):
    status: ValidationStatus
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: str = ""
    reason: str = ""
    corrected_long_form: Optional[str] = None


# =============================================================================
# LLM ENGINE
# =============================================================================


class LLMEngine:
    """
    Abbreviation verifier using LLM.

    Input: Candidate (short_form required, long_form optional)
    Output: ExtractedEntity with EvidenceSpan + immutable provenance trail.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        model: str = "claude-sonnet-4-20250514",
        prompt_version: str = "latest",
        temperature: float = 0.0,
        max_tokens: int = 450,
        top_p: float = 1.0,
        seed: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
        pipeline_version: Optional[str] = None,
    ):
        self.client = llm_client
        self.model = model
        self.prompt_version = prompt_version
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.top_p = float(top_p)
        self.seed = seed
        self.response_format = response_format or {"type": "json_object"}
        self.run_id = run_id or generate_run_id("RUN")
        self.pipeline_version = pipeline_version or get_git_revision_hash()

        # Cache for LLM responses: (sf_upper, lf_lower) -> validation result
        self._validation_cache: Dict[Tuple[str, Optional[str]], Dict[str, Any]] = {}

    def verify_candidate(self, candidate: Candidate) -> ExtractedEntity:
        """Verify a single candidate using the LLM."""
        context = (candidate.context_text or "").strip()
        if not context:
            return self._entity_ambiguous(
                candidate,
                reason="No context_text provided; cannot verify.",
                flags=["no_context"],
                raw_llm=None,
            )

        task = self._select_task(candidate.field_type)
        prompt_version = self._select_prompt_version(candidate)

        llm_params = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "seed": self.seed,
            "response_format": self.response_format,
        }
        bundle = PromptRegistry.get_bundle(task, version=prompt_version, llm_parameters=llm_params)
        context_hash = hash_string(context)
        provenance_text = self._build_provenance_context(candidate)

        user_prompt = bundle.user_template.format(
            context=context,
            sf=candidate.short_form,
            lf=candidate.long_form or "",
            provenance=provenance_text,
        )

        try:
            raw = self.client.complete_json(
                system_prompt=bundle.system_prompt,
                user_prompt=user_prompt,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                seed=self.seed,
                response_format=self.response_format,
                call_type="abbreviation_single_validation",
            )
        except (AnthropicRateLimitError, AnthropicConnectionError, AnthropicStatusError, AnthropicAPIError, Exception) as e:
            return self._handle_llm_error(candidate, e, context_hash, bundle)

        try:
            result = VerificationResult.model_validate(raw)
        except ValidationError as ve:
            return self._entity_ambiguous(
                candidate,
                reason=f"LLM response schema invalid: {ve}",
                flags=["llm_schema_error"],
                raw_llm=raw if isinstance(raw, dict) else {"raw": str(raw)},
                context_hash=context_hash,
                prompt_bundle_hash=bundle.prompt_bundle_hash,
                rule_version=f"{task.value}:{bundle.version}",
            )

        return self._build_entity_from_result(candidate, result, raw, bundle, context_hash, task)

    def verify_candidates_batch(
        self,
        candidates: List[Candidate],
        batch_size: int = 15,
        delay_ms: float = 100,
    ) -> List[ExtractedEntity]:
        """Verify multiple candidates in batches using a single LLM call per batch."""
        if not candidates:
            return []

        results: List[ExtractedEntity] = []
        uncached_candidates: List[Tuple[int, Candidate]] = []
        cached_results: Dict[int, ExtractedEntity] = {}

        for i, c in enumerate(candidates):
            cache_key = (c.short_form.upper(), (c.long_form or "").lower() or None)
            if cache_key in self._validation_cache:
                cached_results[i] = self._build_entity_from_cache(c, self._validation_cache[cache_key])
            else:
                uncached_candidates.append((i, c))

        uncached_results: Dict[int, ExtractedEntity] = {}
        batch_candidates = [c for _, c in uncached_candidates]
        batch_indices = [i for i, _ in uncached_candidates]

        for batch_start in range(0, len(batch_candidates), batch_size):
            if batch_start > 0 and delay_ms > 0:
                time.sleep(delay_ms / 1000)
            batch = batch_candidates[batch_start : batch_start + batch_size]
            batch_idx = batch_indices[batch_start : batch_start + batch_size]
            batch_results = self._verify_batch(batch)

            for idx, candidate, entity in zip(batch_idx, batch, batch_results):
                cache_key = (candidate.short_form.upper(), (candidate.long_form or "").lower() or None)
                self._validation_cache[cache_key] = {
                    "status": entity.status.value,
                    "confidence": entity.confidence_score,
                    "rejection_reason": entity.rejection_reason,
                }
                uncached_results[idx] = entity

        for i in range(len(candidates)):
            result_entity = cached_results.get(i) or uncached_results.get(i)
            if result_entity is not None:
                results.append(result_entity)

        return results

    def fast_reject_batch(
        self,
        candidates: List[Candidate],
        haiku_model: str = "claude-3-5-haiku-20241022",
        batch_size: int = 20,
    ) -> Tuple[List[Candidate], List[ExtractedEntity]]:
        """Use Haiku to fast-reject obvious non-abbreviations."""
        if not candidates:
            return [], []

        needs_review: List[Candidate] = []
        rejected: List[ExtractedEntity] = []

        for batch_start in range(0, len(candidates), batch_size):
            batch = candidates[batch_start : batch_start + batch_size]
            batch_needs_review, batch_rejected = self._fast_reject_single_batch(batch, haiku_model)
            needs_review.extend(batch_needs_review)
            rejected.extend(batch_rejected)

        return needs_review, rejected

    # -------------------------------------------------------------------------
    # PRIVATE HELPERS
    # -------------------------------------------------------------------------

    def _handle_llm_error(
        self, candidate: Candidate, error: Exception, context_hash: str, bundle
    ) -> ExtractedEntity:
        """Handle LLM API errors uniformly."""
        error_type = type(error).__name__
        if isinstance(error, AnthropicRateLimitError):
            error_type = "rate_limit"
        elif isinstance(error, AnthropicConnectionError):
            error_type = "connection"
        elif isinstance(error, AnthropicStatusError):
            error_type = "api_status"

        return self._entity_ambiguous(
            candidate,
            reason=f"LLM call failed: {error}",
            flags=[f"llm_{error_type}_error"],
            raw_llm={"error": str(error), "error_type": error_type},
            context_hash=context_hash,
            prompt_bundle_hash=bundle.prompt_bundle_hash,
            rule_version=f"{bundle.task.value}:{bundle.version}",
        )

    def _build_entity_from_result(
        self,
        candidate: Candidate,
        result: VerificationResult,
        raw: Dict[str, Any],
        bundle,
        context_hash: str,
        task: PromptTask,
    ) -> ExtractedEntity:
        """Build ExtractedEntity from verification result."""
        context = (candidate.context_text or "").strip()
        final_lf = candidate.long_form
        final_status = result.status

        if candidate.field_type in (FieldType.DEFINITION_PAIR, FieldType.GLOSSARY_ENTRY):
            if result.corrected_long_form and result.corrected_long_form.strip():
                final_lf = result.corrected_long_form.strip()
                if result.status == ValidationStatus.REJECTED:
                    final_status = ValidationStatus.VALIDATED

        ev_text = (result.evidence or context).strip()
        start_off, end_off = self._infer_offsets(context, candidate.short_form, final_lf)

        primary = EvidenceSpan(
            text=ev_text,
            location=candidate.context_location,
            scope_ref=context_hash,
            start_char_offset=start_off,
            end_char_offset=end_off,
        )

        prov = self._updated_provenance(
            candidate.provenance,
            prompt_bundle_hash=bundle.prompt_bundle_hash,
            context_hash=context_hash,
            rule_version=f"{task.value}:{bundle.version}",
        )

        out_lf = None if candidate.field_type == FieldType.SHORT_FORM_ONLY else final_lf
        rejection_reason = result.reason if final_status == ValidationStatus.REJECTED else None

        return ExtractedEntity(
            candidate_id=candidate.id,
            doc_id=candidate.doc_id,
            field_type=candidate.field_type,
            short_form=candidate.short_form.strip(),
            long_form=(out_lf.strip() if isinstance(out_lf, str) else None),
            primary_evidence=primary,
            supporting_evidence=[],
            status=final_status,
            confidence_score=float(result.confidence),
            rejection_reason=rejection_reason,
            validation_flags=[],
            provenance=prov,
            raw_llm_response=raw,
        )

    def _build_entity_from_cache(self, candidate: Candidate, cached: Dict[str, Any]) -> ExtractedEntity:
        """Build an ExtractedEntity from cached validation result."""
        status_str = cached.get("status", "AMBIGUOUS")
        status = ValidationStatus(status_str) if status_str in [s.value for s in ValidationStatus] else ValidationStatus.AMBIGUOUS

        prov = ProvenanceMetadata(
            run_id=self.run_id,
            pipeline_version=self.pipeline_version,
            doc_fingerprint=candidate.provenance.doc_fingerprint if candidate.provenance else "unknown",
            generator_name=candidate.generator_type,
            rule_version="cached",
        )

        context = candidate.context_text or ""
        start_off, end_off = self._infer_offsets(context, candidate.short_form, candidate.long_form)
        primary = EvidenceSpan(
            text=context.strip(),
            location=candidate.context_location,
            scope_ref="cached",
            start_char_offset=start_off,
            end_char_offset=end_off,
        )

        return ExtractedEntity(
            candidate_id=candidate.id,
            doc_id=candidate.doc_id,
            short_form=candidate.short_form,
            long_form=candidate.long_form,
            field_type=candidate.field_type,
            primary_evidence=primary,
            supporting_evidence=[],
            status=status,
            confidence_score=cached.get("confidence", 0.5),
            rejection_reason=cached.get("rejection_reason"),
            validation_flags=["from_cache"],
            provenance=prov,
            raw_llm_response={"cached": True},
        )

    def _verify_batch(self, batch: List[Candidate]) -> List[ExtractedEntity]:
        """Verify a single batch of candidates with one LLM call."""
        if not batch:
            return []

        candidate_lines, id_to_candidate = self._build_batch_prompt_lines(batch)
        candidates_text = "\n\n".join(candidate_lines)
        batch_max_tokens = max(self.max_tokens * 2, len(batch) * 100 + 200)

        llm_params = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": batch_max_tokens,
            "top_p": self.top_p,
            "seed": self.seed,
            "response_format": self.response_format,
        }
        bundle = PromptRegistry.get_bundle(PromptTask.VERIFY_BATCH, version="latest", llm_parameters=llm_params)
        user_prompt = bundle.user_template.format(candidates=candidates_text, count=len(batch))

        try:
            raw = self._call_client_json_any(
                bundle.system_prompt, user_prompt, batch_max_tokens,
                call_type="abbreviation_batch_validation",
            )
        except (AnthropicRateLimitError, AnthropicConnectionError, AnthropicTimeoutError, AnthropicStatusError, AnthropicAPIError) as e:
            logger.warning("Batch LLM error, falling back: %s", e)
            return [self.verify_candidate(c) for c in batch]
        except Exception as e:
            logger.error("Batch LLM unexpected error (%s): %s", type(e).__name__, e)
            return [self.verify_candidate(c) for c in batch]

        return self._parse_batch_response(batch, raw)

    def _build_batch_prompt_lines(self, batch: List[Candidate]) -> Tuple[List[str], Dict[str, Candidate]]:
        """Build prompt lines for batch validation."""
        candidate_lines = []
        id_to_candidate: Dict[str, Candidate] = {}

        for c in batch:
            cid = str(c.id)
            id_to_candidate[cid] = c
            context = (c.context_text or "")[:400]

            source = "UNKNOWN"
            if c.generator_type:
                gen_name = c.generator_type.value
                if "syntax" in gen_name.lower():
                    source = "SYNTAX_PATTERN"
                elif "lexicon" in gen_name.lower():
                    source = "LEXICON_MATCH"
                elif "glossary" in gen_name.lower():
                    source = "GLOSSARY_TABLE"
                elif "layout" in gen_name.lower() or "table" in gen_name.lower():
                    source = "TABLE_LAYOUT"

            has_explicit_pair = source == "SYNTAX_PATTERN"
            ctx_lower = context.lower()
            sf_in_ctx = c.short_form.lower() in ctx_lower
            lf_in_ctx = (c.long_form or "").lower() in ctx_lower if c.long_form else False
            ctx_pair = sf_in_ctx and lf_in_ctx
            lexicon_src = c.provenance.lexicon_source if c.provenance and c.provenance.lexicon_source else ""

            candidate_lines.append(
                f"- id: {cid}\n"
                f'  sf: "{c.short_form}"\n'
                f'  lf: "{c.long_form or "(none)"}"\n'
                f"  source: {source}\n"
                f"  has_explicit_pair: {str(has_explicit_pair).lower()}\n"
                f"  ctx_pair: {str(ctx_pair).lower()}\n"
                f'  lexicon: "{lexicon_src}"\n'
                f'  context: "{context}"'
            )

        return candidate_lines, id_to_candidate

    def _call_client_json_any(
        self, system_prompt: str, user_prompt: str, max_tokens: int,
        call_type: str = "batch_validation",
    ) -> Any:
        """Call LLM client with complete_json_any if available."""
        if hasattr(self.client, "complete_json_any"):
            return self.client.complete_json_any(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=self.model,
                temperature=self.temperature,
                max_tokens=max_tokens,
                top_p=self.top_p,
                seed=self.seed,
                response_format=self.response_format,
                call_type=call_type,
            )
        return self.client.complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=self.model,
            temperature=self.temperature,
            max_tokens=max_tokens,
            top_p=self.top_p,
            seed=self.seed,
            response_format=self.response_format,
            call_type=call_type,
        )

    def _parse_batch_response(self, batch: List[Candidate], raw: Any) -> List[ExtractedEntity]:
        """Parse batch LLM response into individual ExtractedEntity results."""
        expected_count = len(batch)

        if raw is None:
            logger.warning("Batch parse failed (raw=None), falling back")
            return [self.verify_candidate(c) for c in batch]

        response_list: List[Dict] = []
        if isinstance(raw, dict):
            if "results" in raw and isinstance(raw["results"], list):
                response_list = raw["results"]
            else:
                for key in ["validations", "items", "responses"]:
                    if key in raw and isinstance(raw[key], list):
                        response_list = raw[key]
                        break
        elif isinstance(raw, list):
            response_list = raw

        if len(response_list) != expected_count:
            logger.warning("Batch parse failed (got %d/%d), falling back", len(response_list), expected_count)
            return [self.verify_candidate(c) for c in batch]

        id_to_response: Dict[str, Dict] = {}
        for resp in response_list:
            if isinstance(resp, dict):
                resp_id = str(resp.get("id", ""))
                if resp_id:
                    id_to_response[resp_id] = resp

        results: List[ExtractedEntity] = []
        for idx, candidate in enumerate(batch):
            cid = str(candidate.id)
            candidate_resp = id_to_response.get(cid)

            if candidate_resp:
                results.append(self._build_entity_from_batch_response(candidate, candidate_resp, raw))
            elif idx < len(response_list) and isinstance(response_list[idx], dict):
                results.append(self._build_entity_from_batch_response(candidate, response_list[idx], raw))
            else:
                results.append(self._entity_ambiguous(candidate, reason="Missing response in batch", flags=["batch_missing"], raw_llm=raw))

        return results

    def _build_entity_from_batch_response(self, candidate: Candidate, resp: Dict, raw_batch: Any) -> ExtractedEntity:
        """Build ExtractedEntity from a single batch response item."""
        status_str = str(resp.get("status", "AMBIGUOUS")).upper()
        try:
            status = ValidationStatus(status_str)
        except ValueError:
            status = ValidationStatus.AMBIGUOUS

        confidence = float(resp.get("confidence", 0.5))
        reason = resp.get("reason", "")
        corrected_lf = resp.get("corrected_long_form")

        final_lf = candidate.long_form
        if corrected_lf and isinstance(corrected_lf, str) and corrected_lf.strip():
            final_lf = corrected_lf.strip()
            if status == ValidationStatus.REJECTED:
                status = ValidationStatus.VALIDATED

        context = (candidate.context_text or "").strip()
        context_hash = hash_string(context) if context else "no_context"
        start_off, end_off = self._infer_offsets(context, candidate.short_form, final_lf)

        primary = EvidenceSpan(
            text=context or "",
            location=candidate.context_location,
            scope_ref=context_hash,
            start_char_offset=start_off,
            end_char_offset=end_off,
        )

        prov = candidate.provenance.model_copy(
            update={"pipeline_version": self.pipeline_version, "run_id": self.run_id, "rule_version": "batch_validation:v2.0"}
        )

        rejection_reason = reason if status == ValidationStatus.REJECTED else None

        return ExtractedEntity(
            candidate_id=candidate.id,
            doc_id=candidate.doc_id,
            field_type=candidate.field_type,
            short_form=candidate.short_form.strip(),
            long_form=(final_lf.strip() if final_lf else None),
            primary_evidence=primary,
            supporting_evidence=[],
            status=status,
            confidence_score=confidence,
            rejection_reason=rejection_reason,
            validation_flags=["batch_validated"],
            provenance=prov,
            raw_llm_response={"batch_response": raw_batch, "item_response": resp},
        )

    def _fast_reject_single_batch(self, batch: List[Candidate], haiku_model: str) -> Tuple[List[Candidate], List[ExtractedEntity]]:
        """Process a single batch with Haiku fast-reject."""
        if not batch:
            return [], []

        id_to_candidate: Dict[str, Candidate] = {}
        candidate_lines = []

        for c in batch:
            cid = str(c.id)
            id_to_candidate[cid] = c
            context = (c.context_text or "")[:200]
            candidate_lines.append(
                f"- id: {cid}\n"
                f'  sf: "{c.short_form}"\n'
                f'  lf: "{c.long_form or "(none)"}"\n'
                f'  context: "{context}"'
            )

        candidates_text = "\n\n".join(candidate_lines)
        bundle = PromptRegistry.get_bundle(PromptTask.FAST_REJECT, version="latest")
        user_prompt = bundle.user_template.format(candidates=candidates_text)

        try:
            raw = self._call_client_json_any(
                bundle.system_prompt, user_prompt, self.max_tokens,
                call_type="fast_reject",
            )
        except (AnthropicRateLimitError, AnthropicConnectionError, AnthropicTimeoutError, AnthropicStatusError, AnthropicAPIError, Exception) as e:
            logger.warning("Haiku error, sending all to Sonnet: %s", e)
            return batch, []

        needs_review: List[Candidate] = []
        rejected: List[ExtractedEntity] = []

        results_list = raw.get("results", []) if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])
        id_to_result: Dict[str, Dict] = {str(r.get("id", "")): r for r in results_list if isinstance(r, dict) and "id" in r}

        for c in batch:
            result = id_to_result.get(str(c.id), {})
            decision = str(result.get("decision", "REVIEW")).upper()
            confidence = float(result.get("confidence", 0.5))
            reason = result.get("reason", "")

            if decision == "REJECT" and confidence >= 0.85:
                rejected.append(self._entity_fast_rejected(c, confidence, reason))
            else:
                needs_review.append(c)

        return needs_review, rejected

    def _entity_fast_rejected(self, candidate: Candidate, confidence: float, reason: str) -> ExtractedEntity:
        """Create a rejected entity from Haiku fast-reject."""
        context = (candidate.context_text or "").strip()
        ctx_hash = hash_string(context) if context else "no_context"

        primary = EvidenceSpan(
            text=context,
            location=candidate.context_location,
            scope_ref=ctx_hash,
            start_char_offset=0,
            end_char_offset=len(context),
        )

        prov = candidate.provenance.model_copy(
            update={"pipeline_version": self.pipeline_version, "run_id": self.run_id, "rule_version": "fast_reject:v1.0"}
        )

        return ExtractedEntity(
            candidate_id=candidate.id,
            doc_id=candidate.doc_id,
            field_type=candidate.field_type,
            short_form=candidate.short_form.strip(),
            long_form=(candidate.long_form.strip() if candidate.long_form else None),
            primary_evidence=primary,
            supporting_evidence=[],
            status=ValidationStatus.REJECTED,
            confidence_score=confidence,
            rejection_reason=f"Fast-reject: {reason}",
            validation_flags=["haiku_rejected"],
            provenance=prov,
            raw_llm_response={"haiku_decision": "REJECT", "reason": reason},
        )

    def _select_task(self, field_type: FieldType) -> PromptTask:
        if field_type == FieldType.SHORT_FORM_ONLY:
            return PromptTask.VERIFY_SHORT_FORM_ONLY
        return PromptTask.VERIFY_DEFINITION_PAIR

    def _select_prompt_version(self, candidate: Candidate) -> str:
        """Select prompt version based on candidate source."""
        if candidate.field_type == FieldType.SHORT_FORM_ONLY:
            return "v1.0"

        has_lexicon = False
        if candidate.generator_type and candidate.generator_type.value == "gen:lexicon_match":
            has_lexicon = True
        if candidate.provenance and (candidate.provenance.lexicon_source or candidate.provenance.lexicon_ids):
            has_lexicon = True

        return "v1.2" if has_lexicon else "v1.1"

    def _build_provenance_context(self, candidate: Candidate) -> str:
        """Build provenance context string for Claude validation."""
        parts = []
        prov = candidate.provenance

        if prov.lexicon_source:
            parts.append(f"Lexicon source: {prov.lexicon_source}")
        if prov.lexicon_ids:
            ids_str = ", ".join(f"{lid.source}:{lid.id}" for lid in prov.lexicon_ids)
            parts.append(f"External IDs: {ids_str}")
        if candidate.generator_type:
            gen_name = candidate.generator_type.value
            if gen_name == "gen:lexicon_match":
                parts.append("Source: Medical terminology lexicon")
            elif gen_name == "gen:syntax_pattern":
                parts.append("Source: Explicit definition pattern in document")

        return ("Provenance: " + "; ".join(parts) + "\n") if parts else ""

    def _updated_provenance(
        self,
        existing: ProvenanceMetadata,
        *,
        prompt_bundle_hash: str,
        context_hash: str,
        rule_version: str,
    ) -> ProvenanceMetadata:
        llm_cfg = LLMParameters(
            model_name=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            seed=self.seed,
            response_format=self.response_format,
        )

        return existing.model_copy(
            update={
                "pipeline_version": self.pipeline_version or existing.pipeline_version,
                "run_id": self.run_id or existing.run_id,
                "prompt_bundle_hash": prompt_bundle_hash,
                "context_hash": context_hash,
                "llm_config": llm_cfg,
                "rule_version": rule_version,
            }
        )

    def _entity_ambiguous(
        self,
        candidate: Candidate,
        *,
        reason: str,
        flags: List[str],
        raw_llm: Any = None,
        context_hash: Optional[str] = None,
        prompt_bundle_hash: Optional[str] = None,
        rule_version: Optional[str] = None,
    ) -> ExtractedEntity:
        context = (candidate.context_text or "").strip()
        ctx_hash = context_hash or (hash_string(context) if context else "no_context")

        primary = EvidenceSpan(
            text=context or "",
            location=candidate.context_location,
            scope_ref=ctx_hash,
            start_char_offset=0,
            end_char_offset=max(0, len(context)),
        )

        prov = candidate.provenance
        if prompt_bundle_hash or rule_version or context_hash:
            prov = prov.model_copy(
                update={
                    "pipeline_version": self.pipeline_version or prov.pipeline_version,
                    "run_id": self.run_id or prov.run_id,
                    "prompt_bundle_hash": prompt_bundle_hash,
                    "context_hash": ctx_hash,
                    "llm_config": LLMParameters(
                        model_name=self.model,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
                        seed=self.seed,
                        response_format=self.response_format,
                    ),
                    "rule_version": rule_version,
                }
            )

        lf = None if candidate.field_type == FieldType.SHORT_FORM_ONLY else candidate.long_form

        return ExtractedEntity(
            candidate_id=candidate.id,
            doc_id=candidate.doc_id,
            field_type=candidate.field_type,
            short_form=candidate.short_form.strip(),
            long_form=(lf.strip() if isinstance(lf, str) else None),
            primary_evidence=primary,
            supporting_evidence=[],
            status=ValidationStatus.AMBIGUOUS,
            confidence_score=0.0,
            rejection_reason=reason,
            validation_flags=flags,
            provenance=prov,
            raw_llm_response=raw_llm,
        )

    def _infer_offsets(self, context: str, sf: str, lf: Optional[str]) -> Tuple[int, int]:
        """Best-effort evidence offsets inside context_text."""
        ctx = context or ""
        if not ctx:
            return (0, 0)

        sf_idx = ctx.lower().find((sf or "").lower())
        if sf_idx == -1:
            return (0, len(ctx))

        start, end = sf_idx, sf_idx + len(sf)

        if lf:
            lf_idx = ctx.lower().find(lf.lower())
            if lf_idx != -1:
                start = min(start, lf_idx)
                end = max(end, lf_idx + len(lf))

        return (start, end)


__all__ = [
    "ClaudeClient",
    "LLMClient",
    "LLMEngine",
    "LLMUsageRecord",
    "LLMUsageTracker",
    "MODEL_PRICING",
    "VerificationResult",
    "calc_record_cost",
    "get_usage_tracker",
    "record_api_usage",
    "resolve_model_tier",
]

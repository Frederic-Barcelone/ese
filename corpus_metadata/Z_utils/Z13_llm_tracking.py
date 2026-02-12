"""
LLM usage tracking, cost calculation, and model tier routing.

Infrastructure utilities for tracking token usage across all API calls,
calculating estimated costs, and resolving which model to use for each
call type based on config.yaml tiers.

Key Components:
    - MODEL_PRICING: Per-model token pricing
    - calc_record_cost: Calculate cost for a single API call
    - LLMUsageRecord: Single API call usage record
    - LLMUsageTracker: Accumulator for token usage across a pipeline run
    - get_usage_tracker: Access the global usage tracker singleton
    - resolve_model_tier: Resolve model from config.yaml for a call type
    - record_api_usage: Record usage from raw anthropic API responses

Example:
    >>> from Z_utils.Z13_llm_tracking import get_usage_tracker, calc_record_cost
    >>> tracker = get_usage_tracker()
    >>> tracker.record("claude-sonnet-4-20250514", 1000, 200, "validation")
    >>> print(f"Cost: ${tracker.estimated_cost():.4f}")

Dependencies:
    - yaml: Config loading for model tiers
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


class CallType:
    """Registry of all known LLM call types with expected model tier.

    Every LLM call site must use one of these constants. Adding a new call site
    requires adding the constant here AND a corresponding entry in config.yaml
    model_tiers.
    """

    # Haiku tier — simple classification / extraction tasks
    ABBREVIATION_BATCH_VALIDATION = "abbreviation_batch_validation"
    ABBREVIATION_SINGLE_VALIDATION = "abbreviation_single_validation"
    FAST_REJECT = "fast_reject"
    DOCUMENT_CLASSIFICATION = "document_classification"
    IMAGE_CLASSIFICATION = "image_classification"
    SF_ONLY_EXTRACTION = "sf_only_extraction"
    LAYOUT_ANALYSIS = "layout_analysis"
    VLM_VISUAL_ENRICHMENT = "vlm_visual_enrichment"
    DESCRIPTION_EXTRACTION = "description_extraction"
    OCR_TEXT_FALLBACK = "ocr_text_fallback"
    AUTHOR_EXTRACTION = "author_extraction"

    # Sonnet tier — complex reasoning / extraction tasks
    FEASIBILITY_EXTRACTION = "feasibility_extraction"
    RECOMMENDATION_EXTRACTION = "recommendation_extraction"
    RECOMMENDATION_VLM = "recommendation_vlm"
    VLM_TABLE_EXTRACTION = "vlm_table_extraction"
    FLOWCHART_ANALYSIS = "flowchart_analysis"
    CHART_ANALYSIS = "chart_analysis"
    VLM_DETECTION = "vlm_detection"

    HAIKU_CALL_TYPES = {
        ABBREVIATION_BATCH_VALIDATION,
        ABBREVIATION_SINGLE_VALIDATION,
        FAST_REJECT,
        DOCUMENT_CLASSIFICATION,
        IMAGE_CLASSIFICATION,
        SF_ONLY_EXTRACTION,
        LAYOUT_ANALYSIS,
        VLM_VISUAL_ENRICHMENT,
        DESCRIPTION_EXTRACTION,
        OCR_TEXT_FALLBACK,
        AUTHOR_EXTRACTION,
    }

    SONNET_CALL_TYPES = {
        FEASIBILITY_EXTRACTION,
        RECOMMENDATION_EXTRACTION,
        RECOMMENDATION_VLM,
        VLM_TABLE_EXTRACTION,
        FLOWCHART_ANALYSIS,
        CHART_ANALYSIS,
        VLM_DETECTION,
    }

    ALL_CALL_TYPES = HAIKU_CALL_TYPES | SONNET_CALL_TYPES

    # Default call_type values that indicate a caller forgot to pass a proper one
    _DEFAULT_CALL_TYPES = {"json", "json_any", "vision", "unknown"}


# Model pricing per million tokens (input_cost, output_cost)
MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-sonnet-4-5-20250929": (3.0, 15.0),
    "claude-3-5-haiku-20241022": (0.80, 4.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
}


def calc_record_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> float:
    """Calculate estimated cost in USD for a single API call."""
    input_price, output_price = MODEL_PRICING.get(model, (3.0, 15.0))
    base_input = input_tokens - cache_read_tokens - cache_creation_tokens
    cache_read_cost = cache_read_tokens * (input_price * 0.1) / 1_000_000
    cache_create_cost = cache_creation_tokens * (input_price * 1.25) / 1_000_000
    input_cost = max(0, base_input) * input_price / 1_000_000
    output_cost = output_tokens * output_price / 1_000_000
    return input_cost + cache_read_cost + cache_create_cost + output_cost


@dataclass
class LLMUsageRecord:
    """Single API call usage record."""
    model: str
    input_tokens: int
    output_tokens: int
    call_type: str
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


@dataclass
class LLMUsageTracker:
    """Accumulates token usage across all API calls in a pipeline run."""
    records: List[LLMUsageRecord] = field(default_factory=list)

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        call_type: str = "unknown",
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
    ) -> None:
        self.records.append(LLMUsageRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            call_type=call_type,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
        ))

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.records)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.records)

    @property
    def total_cache_read_tokens(self) -> int:
        return sum(r.cache_read_tokens for r in self.records)

    @property
    def total_calls(self) -> int:
        return len(self.records)

    def estimated_cost(self) -> float:
        """Calculate estimated cost in USD based on model pricing."""
        return sum(
            calc_record_cost(r.model, r.input_tokens, r.output_tokens,
                             r.cache_read_tokens, r.cache_creation_tokens)
            for r in self.records
        )

    def summary_by_model(self) -> Dict[str, Dict[str, Any]]:
        """Summarize usage grouped by model."""
        by_model: Dict[str, Dict[str, Any]] = {}
        for r in self.records:
            if r.model not in by_model:
                by_model[r.model] = {
                    "calls": 0, "input_tokens": 0, "output_tokens": 0,
                    "cache_read_tokens": 0, "cost": 0.0,
                }
            entry = by_model[r.model]
            entry["calls"] += 1
            entry["input_tokens"] += r.input_tokens
            entry["output_tokens"] += r.output_tokens
            entry["cache_read_tokens"] += r.cache_read_tokens
            entry["cost"] += calc_record_cost(
                r.model, r.input_tokens, r.output_tokens,
                r.cache_read_tokens, r.cache_creation_tokens,
            )
        return by_model

    def summary_by_call_type(self) -> Dict[str, Dict[str, Any]]:
        """Summarize usage grouped by call type."""
        by_type: Dict[str, Dict[str, Any]] = {}
        for r in self.records:
            if r.call_type not in by_type:
                by_type[r.call_type] = {
                    "calls": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0,
                }
            entry = by_type[r.call_type]
            entry["calls"] += 1
            entry["input_tokens"] += r.input_tokens
            entry["output_tokens"] += r.output_tokens
            entry["cost"] += calc_record_cost(
                r.model, r.input_tokens, r.output_tokens,
                r.cache_read_tokens, r.cache_creation_tokens,
            )
        return by_type

    def reset(self) -> None:
        self.records.clear()


# Global usage tracker (shared across all ClaudeClient instances)
_global_usage_tracker = LLMUsageTracker()


def get_usage_tracker() -> LLMUsageTracker:
    """Get the global LLM usage tracker."""
    return _global_usage_tracker


_model_tier_cache: Optional[Dict[str, str]] = None


def resolve_model_tier(call_type: str, default_model: str = "claude-sonnet-4-20250514") -> str:
    """Resolve the model to use for a given call type from config.yaml tiers.

    Loads model_tiers from config once and caches. Falls back to default_model
    if no tier mapping exists for the call_type.
    """
    global _model_tier_cache
    if _model_tier_cache is None:
        try:
            config_path = Path(__file__).parent.parent / "G_config" / "config.yaml"
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                tiers = data.get("api", {}).get("claude", {}).get("model_tiers", {})
                _model_tier_cache = tiers if isinstance(tiers, dict) else {}
            else:
                _model_tier_cache = {}
        except (OSError, yaml.YAMLError, TypeError, KeyError) as e:
            logger.warning("Failed to load model_tiers from config: %s", e)
            _model_tier_cache = {}

        if not _model_tier_cache:
            logger.warning(
                "model_tiers config is empty — all calls will use default model '%s'. "
                "This may route Haiku-tier tasks to an expensive model.",
                default_model,
            )

    if call_type in CallType._DEFAULT_CALL_TYPES:
        logger.warning(
            "resolve_model_tier called with default call_type '%s' — "
            "caller should pass an explicit CallType constant.",
            call_type,
        )
    elif call_type not in _model_tier_cache:
        logger.warning(
            "call_type '%s' not in model_tiers config, falling back to default model '%s'. "
            "This may route to an expensive model unintentionally.",
            call_type, default_model,
        )

    return _model_tier_cache.get(call_type, default_model)


def record_api_usage(message, model: str, call_type: str) -> None:
    """Record token usage from a raw anthropic API response into the global tracker.

    Use this for direct anthropic.Anthropic().messages.create() calls
    outside of ClaudeClient (e.g., in B_parsing, C_generators).
    """
    if message is None or not hasattr(message, "usage") or not message.usage:
        logger.debug("No usage data on API response for call_type=%s", call_type)
        return
    tracker = _global_usage_tracker
    usage = message.usage
    tracker.record(
        model=model,
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
        call_type=call_type,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
    )


__all__ = [
    "CallType",
    "MODEL_PRICING",
    "LLMUsageRecord",
    "LLMUsageTracker",
    "calc_record_cost",
    "get_usage_tracker",
    "record_api_usage",
    "resolve_model_tier",
]

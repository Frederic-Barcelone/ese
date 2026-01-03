# corpus_metadata/corpus_abbreviations/D_validation/D02_llm_engine.py
"""
LLM Engine for abbreviation validation.

Contains:
  - LLMClient: Protocol (interface)
  - ClaudeClient: Anthropic Claude implementation
  - LLMEngine: Verifier that uses any LLMClient
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Tuple

import yaml
from pydantic import BaseModel, Field, ValidationError

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

# Optional anthropic import
try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore


# -----------------------------------------------------------------------------
# Protocol
# -----------------------------------------------------------------------------

class LLMClient(Protocol):
    """
    Vendor-agnostic interface.
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
    ) -> Dict[str, Any]:
        ...


# -----------------------------------------------------------------------------
# Claude Client
# -----------------------------------------------------------------------------

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
    ):
        if anthropic is None:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            )

        # Load config from YAML if provided
        cfg = self._load_config(config_path) if config_path else {}

        # Resolve API key: param > config > env
        self.api_key = (
            api_key
            or cfg.get("api_key")
            or os.environ.get("ANTHROPIC_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY env var, "
                "pass api_key param, or configure in config.yaml"
            )

        # Resolve model params: param > config > defaults
        self.default_model = model or cfg.get("model", "claude-sonnet-4-20250514")
        self.default_max_tokens = max_tokens or cfg.get("max_tokens", 1024)
        self.default_temperature = temperature if temperature is not None else cfg.get("temperature", 0.0)

        # Initialize client
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load Claude config from YAML file."""
        path = Path(config_path)
        if not path.exists():
            return {}

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            # Extract claude validation config
            # Expected structure: api.claude.validation.{model, max_tokens, temperature}
            claude_cfg = data.get("api", {}).get("claude", {})
            
            # Try validation config first, then fast config
            val_cfg = claude_cfg.get("validation", {})
            if not val_cfg:
                val_cfg = claude_cfg.get("fast", {})

            return {
                "api_key": claude_cfg.get("api_key"),
                "model": val_cfg.get("model"),
                "max_tokens": val_cfg.get("max_tokens"),
                "temperature": val_cfg.get("temperature"),
            }
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")
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
        seed: Optional[int] = None,  # Claude doesn't support seed
        response_format: Optional[Dict[str, Any]] = None,  # Ignored, we parse JSON
    ) -> Dict[str, Any]:
        """
        Call Claude and return parsed JSON response.
        """
        use_model = model or self.default_model
        use_max_tokens = max_tokens or self.default_max_tokens
        use_temperature = temperature if temperature is not None else self.default_temperature

        # Call Claude API
        message = self.client.messages.create(
            model=use_model,
            max_tokens=use_max_tokens,
            temperature=use_temperature,
            top_p=top_p,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        # Extract text content
        raw_text = ""
        for block in message.content:
            if hasattr(block, "text"):
                raw_text += block.text

        # Parse JSON from response
        return self._extract_json(raw_text)

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """
        Extract JSON from Claude's response.
        Handles markdown code blocks and raw JSON.
        """
        text = (text or "").strip()

        parsed = None

        # Try markdown code block: ```json {...} ```
        code_block_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.DOTALL)
        if code_block_match:
            try:
                parsed = json.loads(code_block_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find JSON object (handles nested braces)
        if parsed is None:
            # Find first { and match to its closing }
            start = text.find("{")
            if start != -1:
                depth = 0
                end = start
                for i, ch in enumerate(text[start:], start):
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                if depth == 0:
                    try:
                        parsed = json.loads(text[start:end])
                    except json.JSONDecodeError:
                        pass

        # Try entire response
        if parsed is None:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                pass

        # Validate that parsed dict has required "status" key
        if isinstance(parsed, dict) and "status" in parsed:
            return parsed

        # Fallback: return AMBIGUOUS result
        return {
            "status": "AMBIGUOUS",
            "confidence": 0.0,
            "evidence": "",
            "reason": f"Failed to parse JSON from response: {text[:200]}",
            "corrected_long_form": None,
        }


# -----------------------------------------------------------------------------
# Verification Result Schema
# -----------------------------------------------------------------------------

class VerificationResult(BaseModel):
    status: ValidationStatus
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: str = ""
    reason: str = ""
    corrected_long_form: Optional[str] = None


# -----------------------------------------------------------------------------
# LLM Engine
# -----------------------------------------------------------------------------

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

    def verify_candidate(self, candidate: Candidate) -> ExtractedEntity:
        """Verify a single candidate using the LLM."""
        # 1) Basic guardrails
        context = (candidate.context_text or "").strip()
        if not context:
            return self._entity_ambiguous(
                candidate,
                reason="No context_text provided; cannot verify.",
                flags=["no_context"],
                raw_llm=None,
            )

        task = self._select_task(candidate.field_type)

        # 2) Prompt bundle + deterministic hashes
        llm_params = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "seed": self.seed,
            "response_format": self.response_format,
        }
        bundle = PromptRegistry.get_bundle(
            task, version=self.prompt_version, llm_parameters=llm_params
        )
        context_hash = hash_string(context)

        # Build provenance context for v1.1+ prompts
        provenance_text = self._build_provenance_context(candidate)

        user_prompt = bundle.user_template.format(
            context=context,
            sf=candidate.short_form,
            lf=candidate.long_form or "",
            provenance=provenance_text,
        )

        # 3) Call LLM
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
            )
        except Exception as e:
            return self._entity_ambiguous(
                candidate,
                reason=f"LLM call failed: {e}",
                flags=["llm_call_error"],
                raw_llm={"error": str(e)},
                context_hash=context_hash,
                prompt_bundle_hash=bundle.prompt_bundle_hash,
                rule_version=f"{task.value}:{bundle.version}",
            )

        # 4) Validate response schema
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

        # 5) Decide final LF (only for definition/glossary)
        final_lf = candidate.long_form
        if candidate.field_type in (FieldType.DEFINITION_PAIR, FieldType.GLOSSARY_ENTRY):
            if result.corrected_long_form:
                cl = result.corrected_long_form.strip()
                if cl:
                    final_lf = cl

        # 6) Build evidence span
        ev_text = (result.evidence or context).strip()
        start_off, end_off = self._infer_offsets(
            context=context,
            sf=candidate.short_form,
            lf=final_lf,
        )

        primary = EvidenceSpan(
            text=ev_text,
            location=candidate.context_location,
            scope_ref=context_hash,
            start_char_offset=start_off,
            end_char_offset=end_off,
        )

        # 7) Update provenance
        prov = self._updated_provenance(
            candidate.provenance,
            prompt_bundle_hash=bundle.prompt_bundle_hash,
            context_hash=context_hash,
            rule_version=f"{task.value}:{bundle.version}",
        )

        # 8) SHORT_FORM_ONLY: never create LF
        out_lf: Optional[str] = final_lf
        if candidate.field_type == FieldType.SHORT_FORM_ONLY:
            out_lf = None

        # 9) Build entity
        rejection_reason = None
        if result.status == ValidationStatus.REJECTED:
            rejection_reason = result.reason or "Rejected by verifier"

        return ExtractedEntity(
            candidate_id=candidate.id,
            doc_id=candidate.doc_id,
            field_type=candidate.field_type,
            short_form=candidate.short_form.strip(),
            long_form=(out_lf.strip() if isinstance(out_lf, str) else None),
            primary_evidence=primary,
            supporting_evidence=[],
            status=result.status,
            confidence_score=float(result.confidence),
            rejection_reason=rejection_reason,
            validation_flags=[],
            provenance=prov,
            raw_llm_response=raw,
        )

    # -------------------------
    # Helpers
    # -------------------------

    def _select_task(self, field_type: FieldType) -> PromptTask:
        if field_type == FieldType.SHORT_FORM_ONLY:
            return PromptTask.VERIFY_SHORT_FORM_ONLY
        return PromptTask.VERIFY_DEFINITION_PAIR

    def _build_provenance_context(self, candidate: Candidate) -> str:
        """
        Build provenance context string for Claude validation.
        Includes lexicon source and external IDs if available.
        """
        parts = []
        prov = candidate.provenance

        # Add lexicon source
        if prov.lexicon_source:
            parts.append(f"Lexicon source: {prov.lexicon_source}")

        # Add external IDs (CUI, etc.)
        if prov.lexicon_ids:
            ids_str = ", ".join(f"{lid.source}:{lid.id}" for lid in prov.lexicon_ids)
            parts.append(f"External IDs: {ids_str}")

        # Add generator type
        if candidate.generator_type:
            gen_name = candidate.generator_type.value
            if gen_name == "gen:lexicon_match":
                parts.append("Source: Medical terminology lexicon")
            elif gen_name == "gen:syntax_pattern":
                parts.append("Source: Explicit definition pattern in document")

        if parts:
            return "Provenance: " + "; ".join(parts) + "\n"
        return ""

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
        flags: list[str],
        raw_llm: Optional[Dict[str, Any]],
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

        lf = candidate.long_form
        if candidate.field_type == FieldType.SHORT_FORM_ONLY:
            lf = None

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

    def _infer_offsets(
        self, *, context: str, sf: str, lf: Optional[str]
    ) -> Tuple[int, int]:
        """
        Best-effort evidence offsets inside context_text.
        """
        ctx = context or ""
        if not ctx:
            return (0, 0)

        sf_idx = ctx.lower().find((sf or "").lower())
        if sf_idx == -1:
            return (0, len(ctx))

        start = sf_idx
        end = sf_idx + len(sf)

        if lf:
            lf_idx = ctx.lower().find(lf.lower())
            if lf_idx != -1:
                start = min(start, lf_idx)
                end = max(end, lf_idx + len(lf))

        return (start, end)
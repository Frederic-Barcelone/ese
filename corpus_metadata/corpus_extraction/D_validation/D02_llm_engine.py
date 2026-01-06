# corpus_metadata/corpus_extraction/D_validation/D02_llm_engine.py
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
    ) -> Any: ...


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
        use_temperature = (
            temperature if temperature is not None else self.default_temperature
        )

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
    ):
        """
        Call Claude and return parsed JSON response - handles both objects and arrays.
        """
        use_model = model or self.default_model
        use_max_tokens = max_tokens or self.default_max_tokens
        use_temperature = (
            temperature if temperature is not None else self.default_temperature
        )

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

        # Parse JSON from response (handles arrays and objects)
        return self._extract_json_any(raw_text)

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

    def _extract_json_any(self, text: str):
        """
        Extract JSON from Claude's response - handles both objects and arrays.
        Returns parsed JSON (dict or list) or None if parsing fails.
        """
        text = (text or "").strip()

        # Try markdown code block: ```json [...] ``` or ```json {...} ```
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
            return None

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """
        Extract JSON object from Claude's response.
        Returns dict with "status" key or fallback AMBIGUOUS result.
        """
        parsed = self._extract_json_any(text)

        # Validate that parsed dict has required "status" key
        if isinstance(parsed, dict) and "status" in parsed:
            return parsed

        # Fallback: return AMBIGUOUS result
        return {
            "status": "AMBIGUOUS",
            "confidence": 0.0,
            "evidence": "",
            "reason": f"Failed to parse JSON from response: {(text or '')[:200]}",
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
        # Gate v1.2 (permissive) to lexicon-backed candidates only
        # Use v1.1 (stricter) for syntax/pattern candidates to reduce FP
        prompt_version = self._select_prompt_version(candidate)

        llm_params = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "seed": self.seed,
            "response_format": self.response_format,
        }
        bundle = PromptRegistry.get_bundle(
            task, version=prompt_version, llm_parameters=llm_params
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
        final_status = result.status  # May be upgraded if corrected_long_form provided
        if candidate.field_type in (
            FieldType.DEFINITION_PAIR,
            FieldType.GLOSSARY_ENTRY,
        ):
            if result.corrected_long_form:
                cl = result.corrected_long_form.strip()
                if cl:
                    final_lf = cl
                    # KEY FIX: If LLM provides corrected_long_form, upgrade REJECTED to VALIDATED
                    if result.status == ValidationStatus.REJECTED:
                        final_status = ValidationStatus.VALIDATED

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
        if final_status == ValidationStatus.REJECTED:
            rejection_reason = result.reason or "Rejected by verifier"

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

    def verify_candidates_batch(
        self,
        candidates: list[Candidate],
        batch_size: int = 15,
    ) -> list[ExtractedEntity]:
        """
        Verify multiple candidates in batches using a single LLM call per batch.

        This is ~10x faster than individual calls while maintaining accuracy.
        Candidates are grouped into batches of `batch_size` and validated together.

        Args:
            candidates: List of candidates to validate
            batch_size: Number of candidates per LLM call (default 15)

        Returns:
            List of ExtractedEntity results in the same order as input
        """
        if not candidates:
            return []

        results: list[ExtractedEntity] = []

        # Process in batches
        for batch_start in range(0, len(candidates), batch_size):
            batch = candidates[batch_start : batch_start + batch_size]
            batch_results = self._verify_batch(batch)
            results.extend(batch_results)

        return results

    def _verify_batch(self, batch: list[Candidate]) -> list[ExtractedEntity]:
        """Verify a single batch of candidates with one LLM call."""
        if not batch:
            return []

        # Build batch prompt with v2.0 format: id, has_explicit_pair, source
        candidate_lines = []
        id_to_candidate: dict[str, Candidate] = {}

        for c in batch:
            cid = str(c.id)
            id_to_candidate[cid] = c

            context = (c.context_text or "")[:400]  # More context for better accuracy

            # Determine source from generator_type
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

            # has_explicit_pair: True if SYNTAX_PATTERN (detected LF(SF) pattern)
            has_explicit_pair = source == "SYNTAX_PATTERN"

            # ctx_pair: True if both SF and LF appear in context (case-insensitive)
            ctx_lower = context.lower()
            sf_in_ctx = c.short_form.lower() in ctx_lower
            lf_in_ctx = (
                (c.long_form or "").lower() in ctx_lower if c.long_form else False
            )
            ctx_pair = sf_in_ctx and lf_in_ctx

            # Get lexicon source if available (for trusted source validation)
            lexicon_src = ""
            if c.provenance and c.provenance.lexicon_source:
                lexicon_src = c.provenance.lexicon_source

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

        candidates_text = "\n\n".join(candidate_lines)

        # Get batch prompt bundle
        llm_params = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens * 2,  # More tokens for batch response
            "top_p": self.top_p,
            "seed": self.seed,
            "response_format": self.response_format,
        }
        bundle = PromptRegistry.get_bundle(
            PromptTask.VERIFY_BATCH, version="latest", llm_parameters=llm_params
        )

        user_prompt = bundle.user_template.format(
            candidates=candidates_text,
            count=len(batch),
        )

        # Call LLM - use complete_json_any to handle array responses
        try:
            # Check if client supports complete_json_any (handles arrays)
            if hasattr(self.client, "complete_json_any"):
                raw = self.client.complete_json_any(
                    system_prompt=bundle.system_prompt,
                    user_prompt=user_prompt,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens * 2,
                    top_p=self.top_p,
                    seed=self.seed,
                    response_format=self.response_format,
                )
            else:
                # Fallback for clients without complete_json_any
                raw = self.client.complete_json(
                    system_prompt=bundle.system_prompt,
                    user_prompt=user_prompt,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens * 2,
                    top_p=self.top_p,
                    seed=self.seed,
                    response_format=self.response_format,
                )
        except Exception as e:
            # On error, fall back to individual validation
            print(f"  [WARN] Batch LLM error, falling back: {e}")
            return [self.verify_candidate(c) for c in batch]

        # Parse batch response
        return self._parse_batch_response(batch, raw)

    def _parse_batch_response(
        self, batch: list[Candidate], raw
    ) -> list[ExtractedEntity]:
        """
        Parse batch LLM response into individual ExtractedEntity results.

        v2.0 format expects:
        {
            "expected_count": N,
            "results": [
                {"id": "<candidate_id>", "status": "...", "confidence": 0.X, "reason": "..."},
                ...
            ]
        }

        Uses id-based matching for robustness.
        Falls back to individual validation if parsing fails.
        """
        expected_count = len(batch)

        # Handle None (parsing failed)
        if raw is None:
            print("  [WARN] Batch parse failed (raw=None), falling back")
            return [self.verify_candidate(c) for c in batch]

        # Extract results list from response
        response_list: list[dict] = []

        if isinstance(raw, dict):
            # v2.0 format: {expected_count, results: [...]}
            if "results" in raw and isinstance(raw["results"], list):
                response_list = raw["results"]

                # Hard validation: check expected_count matches
                resp_expected = raw.get("expected_count", len(response_list))
                if resp_expected != expected_count:
                    print(
                        f"  [WARN] Batch count mismatch (expected {expected_count}, got {resp_expected})"
                    )
            else:
                # Legacy fallback: try other common keys
                for key in ["validations", "items", "responses"]:
                    if key in raw and isinstance(raw[key], list):
                        response_list = raw[key]
                        break
        elif isinstance(raw, list):
            # Direct array (legacy v1.0 format)
            response_list = raw

        # Hard validation: must have exactly expected_count results
        if len(response_list) != expected_count:
            print(
                f"  [WARN] Batch parse failed (got {len(response_list)}/{expected_count}), falling back"
            )
            raw_type = type(raw).__name__
            raw_keys = list(raw.keys())[:5] if isinstance(raw, dict) else None
            print(f"    Raw type: {raw_type}, keys: {raw_keys}")
            return [self.verify_candidate(c) for c in batch]

        # Map responses to candidates by id
        id_to_response: dict[str, dict] = {}
        for resp in response_list:
            if isinstance(resp, dict):
                resp_id = str(resp.get("id", ""))
                if resp_id:
                    id_to_response[resp_id] = resp

        # Build results in original batch order
        results: list[ExtractedEntity] = []
        missing_ids: list[str] = []

        for candidate in batch:
            cid = str(candidate.id)
            resp = id_to_response.get(cid)

            if resp:
                results.append(
                    self._build_entity_from_batch_response(candidate, resp, raw)
                )
            else:
                missing_ids.append(cid)
                # Try index-based fallback (for compatibility)
                idx = batch.index(candidate)
                if idx < len(response_list) and isinstance(response_list[idx], dict):
                    results.append(
                        self._build_entity_from_batch_response(
                            candidate, response_list[idx], raw
                        )
                    )
                else:
                    results.append(
                        self._entity_ambiguous(
                            candidate,
                            reason="Missing response in batch validation",
                            flags=["batch_missing"],
                            raw_llm=raw,
                        )
                    )

        if missing_ids:
            print(
                f"  [WARN] {len(missing_ids)} ids not found in response, used index fallback"
            )

        return results

    def _build_entity_from_batch_response(
        self, candidate: Candidate, resp: dict, raw_batch: Any
    ) -> ExtractedEntity:
        """Build ExtractedEntity from a single batch response item."""
        # Parse status
        status_str = str(resp.get("status", "AMBIGUOUS")).upper()
        try:
            status = ValidationStatus(status_str)
        except ValueError:
            status = ValidationStatus.AMBIGUOUS

        confidence = float(resp.get("confidence", 0.5))
        reason = resp.get("reason", "")
        corrected_lf = resp.get("corrected_long_form")

        # Use corrected LF if provided
        final_lf = candidate.long_form
        if corrected_lf and isinstance(corrected_lf, str) and corrected_lf.strip():
            final_lf = corrected_lf.strip()
            # KEY FIX: If LLM provides corrected_long_form, upgrade REJECTED to VALIDATED
            # This handles cases where lexicon has wrong LF but LLM found correct expansion
            if status == ValidationStatus.REJECTED:
                status = ValidationStatus.VALIDATED

        context = (candidate.context_text or "").strip()
        context_hash = hash_string(context) if context else "no_context"

        # Build evidence span
        start_off, end_off = self._infer_offsets(
            context=context, sf=candidate.short_form, lf=final_lf
        )
        primary = EvidenceSpan(
            text=context or "",
            location=candidate.context_location,
            scope_ref=context_hash,
            start_char_offset=start_off,
            end_char_offset=end_off,
        )

        # Build provenance
        prov = candidate.provenance.model_copy(
            update={
                "pipeline_version": self.pipeline_version,
                "run_id": self.run_id,
                "rule_version": "batch_validation:v2.0",
            }
        )

        rejection_reason = None
        if status == ValidationStatus.REJECTED:
            rejection_reason = reason or "Rejected by batch verifier"

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

    def fast_reject_batch(
        self,
        candidates: list[Candidate],
        haiku_model: str = "claude-3-5-haiku-20241022",
        batch_size: int = 20,
    ) -> tuple[list[Candidate], list[ExtractedEntity]]:
        """
        Use Haiku to fast-reject obvious non-abbreviations.

        Returns:
            (needs_review, rejected): Candidates that need Sonnet validation,
                                      and entities for rejected candidates.
        """
        if not candidates:
            return [], []

        needs_review: list[Candidate] = []
        rejected: list[ExtractedEntity] = []

        # Process in batches
        for batch_start in range(0, len(candidates), batch_size):
            batch = candidates[batch_start : batch_start + batch_size]
            batch_needs_review, batch_rejected = self._fast_reject_single_batch(
                batch, haiku_model
            )
            needs_review.extend(batch_needs_review)
            rejected.extend(batch_rejected)

        return needs_review, rejected

    def _fast_reject_single_batch(
        self,
        batch: list[Candidate],
        haiku_model: str,
    ) -> tuple[list[Candidate], list[ExtractedEntity]]:
        """Process a single batch with Haiku fast-reject."""
        if not batch:
            return [], []

        # Build candidate lines for prompt
        id_to_candidate: dict[str, Candidate] = {}
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

        # Get fast-reject prompt bundle
        bundle = PromptRegistry.get_bundle(PromptTask.FAST_REJECT, version="latest")

        user_prompt = bundle.user_template.format(candidates=candidates_text)

        # Call Haiku
        try:
            if hasattr(self.client, "complete_json_any"):
                raw = self.client.complete_json_any(
                    system_prompt=bundle.system_prompt,
                    user_prompt=user_prompt,
                    model=haiku_model,
                    temperature=0.0,
                    max_tokens=self.max_tokens,
                    top_p=1.0,
                )
            else:
                raw = self.client.complete_json(
                    system_prompt=bundle.system_prompt,
                    user_prompt=user_prompt,
                    model=haiku_model,
                    temperature=0.0,
                    max_tokens=self.max_tokens,
                    top_p=1.0,
                )
        except Exception as e:
            print(f"  [WARN] Haiku error, sending all to Sonnet: {e}")
            return batch, []

        # Parse response
        needs_review: list[Candidate] = []
        rejected: list[ExtractedEntity] = []

        results_list = []
        if isinstance(raw, dict) and "results" in raw:
            results_list = raw["results"]
        elif isinstance(raw, list):
            results_list = raw

        # Build id -> result mapping
        id_to_result: dict[str, dict] = {}
        for r in results_list:
            if isinstance(r, dict) and "id" in r:
                id_to_result[str(r["id"])] = r

        # Process each candidate
        for c in batch:
            cid = str(c.id)
            result = id_to_result.get(cid, {})

            decision = str(result.get("decision", "REVIEW")).upper()
            confidence = float(result.get("confidence", 0.5))
            reason = result.get("reason", "")

            # Only reject if high confidence
            if decision == "REJECT" and confidence >= 0.85:
                entity = self._entity_fast_rejected(c, confidence, reason)
                rejected.append(entity)
            else:
                needs_review.append(c)

        return needs_review, rejected

    def _entity_fast_rejected(
        self,
        candidate: Candidate,
        confidence: float,
        reason: str,
    ) -> ExtractedEntity:
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
            update={
                "pipeline_version": self.pipeline_version,
                "run_id": self.run_id,
                "rule_version": "fast_reject:v1.0",
            }
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

    # -------------------------
    # Helpers
    # -------------------------

    def _select_task(self, field_type: FieldType) -> PromptTask:
        if field_type == FieldType.SHORT_FORM_ONLY:
            return PromptTask.VERIFY_SHORT_FORM_ONLY
        return PromptTask.VERIFY_DEFINITION_PAIR

    def _select_prompt_version(self, candidate: Candidate) -> str:
        """
        Select prompt version based on candidate source.

        For DEFINITION_PAIR/GLOSSARY_ENTRY:
        - v1.2 (permissive): For lexicon-backed candidates (trusted sources)
        - v1.1 (stricter): For syntax/pattern candidates (need explicit evidence)

        For SHORT_FORM_ONLY: Always use v1.0 (only version available)

        This gating reduces false positives from over-trusting pattern matches.
        """
        # SHORT_FORM_ONLY only has v1.0
        if candidate.field_type == FieldType.SHORT_FORM_ONLY:
            return "v1.0"

        # Check if candidate has lexicon provenance
        has_lexicon = False

        # Check generator type
        if candidate.generator_type:
            gen_name = candidate.generator_type.value
            if gen_name == "gen:lexicon_match":
                has_lexicon = True

        # Check provenance for lexicon source
        if candidate.provenance and candidate.provenance.lexicon_source:
            has_lexicon = True

        # Check for external IDs (CUI, etc.) which indicate lexicon origin
        if candidate.provenance and candidate.provenance.lexicon_ids:
            has_lexicon = True

        # Use permissive v1.2 only for lexicon-backed candidates
        if has_lexicon:
            return "v1.2"
        else:
            return "v1.1"

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

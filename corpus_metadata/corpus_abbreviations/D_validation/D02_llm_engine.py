# corpus_metadata/corpus_abbreviations/D_validation/D02_llm_engine.py

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Protocol, Tuple

from pydantic import BaseModel, Field, ValidationError

from A_core.A01_domain_models import (
    Candidate,
    ExtractedEntity,
    EvidenceSpan,
    Coordinate,
    FieldType,
    PipelineStage,
    ProvenanceMetadata,
    LLMParameters,
    ValidationStatus,
)
from A_core.A03_provenance import (
    hash_string,
    get_git_revision_hash,
    generate_run_id,
)
from D_validation.D01_prompt_registry import (
    PromptRegistry,
    PromptTask,
)


# -----------------------------------------------------------------------------
# LLM Client Protocol + Claude Implementation
# -----------------------------------------------------------------------------

class LLMClient(Protocol):
    """
    Vendor-agnostic interface.
    Your implementation should return a dict already parsed from JSON.
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


class ClaudeClient:
    """
    Claude API client implementing LLMClient protocol.
    
    Env vars: ANTHROPIC_API_KEY or CLAUDE_API_KEY
    
    Usage:
        client = ClaudeClient()
        engine = LLMEngine(client, model="claude-opus-4-5-20251101")
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY or CLAUDE_API_KEY environment variable required")
        
        # Lazy import to avoid hard dependency
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")

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
        """Call Claude API and return parsed JSON."""
        message = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        # Extract text
        text = ""
        for block in message.content:
            if block.type == "text":
                text += block.text

        # Parse JSON (handle markdown code blocks)
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            return {
                "status": "AMBIGUOUS",
                "confidence": 0.0,
                "evidence": "",
                "reason": f"Failed to parse JSON: {e}",
                "raw_text": text[:500],
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
# LLM Engine (Verifier)
# -----------------------------------------------------------------------------

class LLMEngine:
    """
    Abbreviation-only verifier.

    Input: Candidate (short_form required, long_form optional)
    Output: ExtractedEntity with EvidenceSpan + immutable provenance trail.
    
    Usage with Claude:
        client = ClaudeClient()
        engine = LLMEngine(
            client,
            model="claude-opus-4-5-20251101",  # validation tier
            temperature=0,
            max_tokens=4096,
        )
        entity = engine.verify_candidate(candidate)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        model: str = "claude-sonnet-4-5-20250929",
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
        bundle = PromptRegistry.get_bundle(task, version=self.prompt_version, llm_parameters=llm_params)
        context_hash = hash_string(context)

        user_prompt = bundle.user_template.format(
            context=context,
            sf=candidate.short_form,
            lf=candidate.long_form or "",
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

        # 4) Validate response schema strictly
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

        # 5) Decide final LF (only allowed for definition/glossary)
        final_lf = candidate.long_form
        if candidate.field_type in (FieldType.DEFINITION_PAIR, FieldType.GLOSSARY_ENTRY):
            if result.corrected_long_form:
                cl = result.corrected_long_form.strip()
                if cl:
                    final_lf = cl

        # 6) Build evidence span (best-effort offsets)
        ev_text = (result.evidence or context).strip()
        start_off, end_off = self._infer_offsets(
            context=context,
            sf=candidate.short_form,
            lf=final_lf,
        )

        primary = EvidenceSpan(
            text=ev_text,
            location=candidate.context_location,
            scope_ref=context_hash,  # stable scope for audit
            start_char_offset=start_off,
            end_char_offset=end_off,
        )

        # 7) Update provenance immutably (frozen model)
        prov = self._updated_provenance(
            candidate.provenance,
            prompt_bundle_hash=bundle.prompt_bundle_hash,
            context_hash=context_hash,
            rule_version=f"{task.value}:{bundle.version}",
        )

        # 8) Validation rules for SHORT_FORM_ONLY: never create LF
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

        # IMPORTANT: ProvenanceMetadata is frozen=True -> use model_copy
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

        # For AMBIGUOUS we keep LF only if it existed on input and field_type requires it.
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

    def _infer_offsets(self, *, context: str, sf: str, lf: Optional[str]) -> Tuple[int, int]:
        """
        Best-effort evidence offsets inside context_text.
        Priority:
          1) Span covering "sf" occurrence
          2) If lf exists and found nearby, extend to cover both
          3) Fallback: whole context
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
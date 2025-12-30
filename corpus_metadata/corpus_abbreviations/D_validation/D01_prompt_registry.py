# corpus_metadata/corpus_abbreviations/D_validation/D01_prompt_registry.py

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, ConfigDict

from A_core.A03_provenance import compute_prompt_bundle_hash


class PromptTask(str, Enum):
    VERIFY_DEFINITION_PAIR = "verify_definition_pair"   # DEFINITION_PAIR + GLOSSARY_ENTRY
    VERIFY_SHORT_FORM_ONLY = "verify_short_form_only"   # SHORT_FORM_ONLY (do NOT guess LF)


class PromptBundle(BaseModel):
    task: PromptTask
    version: str

    system_prompt: str
    user_template: str

    output_schema: Optional[Dict[str, Any]] = None
    prompt_bundle_hash: str

    model_config = ConfigDict(frozen=True, extra="forbid")


class PromptRegistry:
    """
    Centralized versioned prompt store.
    - Hash is deterministic (via compute_prompt_bundle_hash).
    - You can later load these from YAML without changing downstream code.
    """

    _LATEST: Dict[PromptTask, str] = {
        PromptTask.VERIFY_DEFINITION_PAIR: "v1.0",
        PromptTask.VERIFY_SHORT_FORM_ONLY: "v1.0",
    }

    _TEMPLATES: Dict[Tuple[PromptTask, str], Dict[str, Any]] = {
        # -------------------------
        # Definition Pair verification
        # -------------------------
        (PromptTask.VERIFY_DEFINITION_PAIR, "v1.0"): {
            "system": (
                "You are a clinical document QA auditor. "
                "Use ONLY the provided context. Do NOT use external knowledge. "
                "Return JSON only."
            ),
            "user": (
                "Context:\n{context}\n\n"
                "Claim: short form '{sf}' stands for long form '{lf}'.\n\n"
                "Decide if the mapping is supported by this context.\n"
                "Rules:\n"
                "1) VALIDATED only if the context explicitly supports SF->LF.\n"
                "2) If LF is not present or relationship is unclear -> AMBIGUOUS.\n"
                "3) If the context contradicts SF->LF -> REJECTED.\n"
                "4) If LF is slightly wrong, you may provide corrected_long_form.\n\n"
                "Return JSON with keys:\n"
                "{{"
                "\"status\": \"VALIDATED|REJECTED|AMBIGUOUS\", "
                "\"confidence\": number, "
                "\"evidence\": string, "
                "\"reason\": string, "
                "\"corrected_long_form\": string|null"
                "}}"
            ),
            "schema": None,
        },

        # -------------------------
        # Short-form-only (orphan) verification
        # -------------------------
        (PromptTask.VERIFY_SHORT_FORM_ONLY, "v1.0"): {
            "system": (
                "You are a clinical document QA auditor. "
                "Use ONLY the provided context. Do NOT guess expansions. "
                "Return JSON only."
            ),
            "user": (
                "Context:\n{context}\n\n"
                "Token: '{sf}'.\n\n"
                "Task:\n"
                "- Decide if '{sf}' is used as an abbreviation-like token in this context.\n"
                "- Do NOT invent a long form.\n\n"
                "Return JSON with keys:\n"
                "{{"
                "\"status\": \"VALIDATED|REJECTED|AMBIGUOUS\", "
                "\"confidence\": number, "
                "\"evidence\": string, "
                "\"reason\": string"
                "}}"
            ),
            "schema": None,
        },
    }

    @classmethod
    def get_bundle(
        cls,
        task: PromptTask,
        version: str = "latest",
        llm_parameters: Optional[Dict[str, Any]] = None,
    ) -> PromptBundle:
        if version == "latest":
            version = cls._LATEST[task]

        key = (task, version)
        if key not in cls._TEMPLATES:
            raise ValueError(f"Prompt template not found: {task.value}:{version}")

        entry = cls._TEMPLATES[key]
        system_prompt = entry["system"]
        user_template = entry["user"]
        schema = entry.get("schema")

        params = llm_parameters or {}
        bundle_hash = compute_prompt_bundle_hash(system_prompt, user_template, schema, params)

        return PromptBundle(
            task=task,
            version=version,
            system_prompt=system_prompt,
            user_template=user_template,
            output_schema=schema,
            prompt_bundle_hash=bundle_hash,
        )
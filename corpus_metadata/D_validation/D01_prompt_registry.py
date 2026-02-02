# corpus_metadata/D_validation/D01_prompt_registry.py
"""
Prompt registry for LLM validation tasks.

Provides:
- PromptTask: Enum of validation task types
- PromptBundle: Versioned prompt templates with system/user components
- PromptRegistry: Central registry for all validation prompts
- Prompt versioning and hash computation for reproducibility
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, ConfigDict

from A_core.A03_provenance import compute_prompt_bundle_hash


class PromptTask(str, Enum):
    VERIFY_DEFINITION_PAIR = (
        "verify_definition_pair"  # DEFINITION_PAIR + GLOSSARY_ENTRY
    )
    VERIFY_SHORT_FORM_ONLY = (
        "verify_short_form_only"  # SHORT_FORM_ONLY (do NOT guess LF)
    )
    VERIFY_BATCH = "verify_batch"  # Batch validation (multiple candidates)
    FAST_REJECT = "fast_reject"  # Haiku screening: REJECT obvious non-abbreviations

    # Disease detection tasks
    VERIFY_DISEASE = "verify_disease"  # Single disease mention validation
    VERIFY_DISEASE_BATCH = "verify_disease_batch"  # Batch disease validation

    # Author detection tasks
    VERIFY_AUTHOR_BATCH = "verify_author_batch"  # Batch author validation

    # Citation detection tasks
    VERIFY_CITATION_BATCH = "verify_citation_batch"  # Batch citation validation


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
        PromptTask.VERIFY_DEFINITION_PAIR: "v1.2",
        PromptTask.VERIFY_SHORT_FORM_ONLY: "v1.0",
        PromptTask.VERIFY_BATCH: "v2.0",
        PromptTask.FAST_REJECT: "v1.0",
        PromptTask.VERIFY_DISEASE: "v1.0",
        PromptTask.VERIFY_DISEASE_BATCH: "v1.0",
        PromptTask.VERIFY_AUTHOR_BATCH: "v1.0",
        PromptTask.VERIFY_CITATION_BATCH: "v1.0",
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
                '"status": "VALIDATED|REJECTED|AMBIGUOUS", '
                '"confidence": number, '
                '"evidence": string, '
                '"reason": string, '
                '"corrected_long_form": string|null'
                "}}"
            ),
            "schema": None,
        },
        # v1.1: Includes provenance context from lexicons
        (PromptTask.VERIFY_DEFINITION_PAIR, "v1.1"): {
            "system": (
                "You are a clinical document QA auditor validating abbreviations. "
                "Return JSON only."
            ),
            "user": (
                "Context:\n{context}\n\n"
                "Claim: short form '{sf}' stands for long form '{lf}'.\n"
                "{provenance}\n"
                "Decide if this abbreviation mapping is valid.\n"
                "Rules:\n"
                "1) VALIDATED if:\n"
                "   - The context explicitly defines SF->LF, OR\n"
                "   - The SF appears in context AND the mapping comes from a trusted lexicon (UMLS, medical dictionary)\n"
                "2) AMBIGUOUS if the SF appears but relationship to LF is unclear and no lexicon source.\n"
                "3) REJECTED if the context contradicts SF->LF or SF is not an abbreviation.\n"
                "4) If LF is slightly wrong, you may provide corrected_long_form.\n\n"
                "Return JSON with keys:\n"
                "{{"
                '"status": "VALIDATED|REJECTED|AMBIGUOUS", '
                '"confidence": number, '
                '"evidence": string, '
                '"reason": string, '
                '"corrected_long_form": string|null'
                "}}"
            ),
            "schema": None,
        },
        # v1.2: More permissive - trust high-quality lexicons, reduce false rejections
        (PromptTask.VERIFY_DEFINITION_PAIR, "v1.2"): {
            "system": (
                "You are a clinical document QA auditor validating medical abbreviations. "
                "Your goal is to confirm valid abbreviations while rejecting clear errors. "
                "When in doubt, lean toward VALIDATED for established medical terms. "
                "Return JSON only."
            ),
            "user": (
                "Context:\n{context}\n\n"
                "Claim: short form '{sf}' stands for long form '{lf}'.\n"
                "{provenance}\n"
                "Decide if this abbreviation mapping is valid for this medical/scientific document.\n\n"
                "Validation Rules (in priority order):\n"
                "1) VALIDATED (high confidence) if:\n"
                "   - Context explicitly defines SF->LF (e.g., 'long form (SF)' pattern), OR\n"
                "   - SF appears in context AND comes from UMLS/medical lexicon (trust the lexicon)\n"
                "2) VALIDATED (medium confidence) if:\n"
                "   - SF appears in context AND LF is a plausible medical/scientific expansion\n"
                "   - The LF makes semantic sense for this document's domain\n"
                "3) REJECTED only if:\n"
                "   - Context explicitly contradicts the SF->LF mapping, OR\n"
                "   - SF is clearly NOT an abbreviation (e.g., a regular word, number, identifier), OR\n"
                "   - LF is obviously wrong for this SF (e.g., 'FDA' -> 'Food Distribution Agency')\n"
                "4) AMBIGUOUS only if:\n"
                "   - SF does not appear in the context at all, OR\n"
                "   - Multiple conflicting expansions are possible and context doesn't clarify\n\n"
                "Important: Standard medical abbreviations (FDA, EMA, eGFR, UPCR, etc.) from trusted "
                "lexicons should be VALIDATED when SF appears in context, even without explicit definition.\n\n"
                "If LF has minor errors (typos, formatting), provide corrected_long_form.\n\n"
                "Return JSON with keys:\n"
                "{{"
                '"status": "VALIDATED|REJECTED|AMBIGUOUS", '
                '"confidence": number (0.0-1.0), '
                '"evidence": string (quote from context), '
                '"reason": string (brief explanation), '
                '"corrected_long_form": string|null'
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
                '"status": "VALIDATED|REJECTED|AMBIGUOUS", '
                '"confidence": number, '
                '"evidence": string, '
                '"reason": string'
                "}}"
            ),
            "schema": None,
        },
        # -------------------------
        # Batch validation (multiple candidates at once)
        # -------------------------
        (PromptTask.VERIFY_BATCH, "v1.0"): {
            "system": (
                "You are a strict clinical document QA auditor validating medical abbreviations. "
                "Apply rigorous standards: only true abbreviations with correct expansions pass. "
                "You will receive multiple candidates to validate. "
                "Return a JSON array with one result per candidate, in the same order. "
                "Return ONLY the JSON array, no other text."
            ),
            "user": (
                "Validate each abbreviation candidate below. Apply STRICT standards.\n\n"
                "Validation Rules (BE STRICT):\n"
                "VALIDATED only if:\n"
                "- SF is a TRUE abbreviation (typically 2-6 uppercase letters/numbers)\n"
                "- Context explicitly defines SF->LF OR SF appears AND comes from UMLS/medical lexicon\n"
                "- LF is the correct, standard expansion for this abbreviation\n\n"
                "REJECTED if:\n"
                "- SF is a common English word (e.g., Data, Methods, White, Age, Study)\n"
                "- SF is a proper noun/company name (e.g., Novartis, Lancet)\n"
                "- LF is wrong or doesn't match standard medical terminology\n"
                "- SF->LF mapping contradicts context\n"
                "- LF just repeats SF or is circular\n\n"
                "AMBIGUOUS if:\n"
                "- SF not found in context\n"
                "- Multiple conflicting expansions possible\n"
                "- Unclear if SF is abbreviation vs regular word\n\n"
                "Candidates:\n{candidates}\n\n"
                "Return a JSON array with exactly {count} objects, one per candidate in order:\n"
                "[\n"
                '  {{"index": 0, "status": "VALIDATED|REJECTED|AMBIGUOUS", "confidence": 0.0-1.0, '
                '"reason": "brief explanation", "corrected_long_form": null}},\n'
                "  ...\n"
                "]\n"
                "IMPORTANT: Return exactly {count} results. BE STRICT - when in doubt, REJECT."
            ),
            "schema": None,
        },
        # v2.0: Robust output contract + anti-AMBIGUOUS rules
        (PromptTask.VERIFY_BATCH, "v2.0"): {
            "system": (
                "You validate medical abbreviations. Return ONLY valid JSON. No markdown. No extra text."
            ),
            "user": (
                "Task: Validate abbreviation mappings independently.\n"
                "Evaluate each candidate independently. Do not compare candidates to each other.\n\n"
                "=== INPUT FIELDS ===\n"
                "- has_explicit_pair: SF and LF found together in document as definition pattern\n"
                "- ctx_pair: Both SF and LF strings appear in context snippet\n"
                "- source: Where candidate came from (SYNTAX_PATTERN, LEXICON_MATCH, etc.)\n"
                "- lexicon: Source lexicon name (UMLS, disease_lexicon_*, etc.)\n\n"
                "=== DECISION RULES ===\n\n"
                "VALIDATED if ANY of these:\n"
                "- has_explicit_pair=true (explicit definition in document - high trust)\n"
                "- source=GLOSSARY_TABLE (from document's own glossary - high trust)\n"
                "- lexicon contains 'UMLS' AND SF is medical/scientific term in context\n"
                "- ctx_pair=true AND source=LEXICON_MATCH (SF and LF both found in context)\n\n"
                "REJECTED if ANY of these:\n"
                "- SF is common English word (DATA, METHODS, WHITE, BLACK, TABLE, FIGURE, RESULTS, "
                "STUDY, AGE, YEARS, PATIENTS, BASELINE) AND has_explicit_pair=false\n"
                "- LF equals or nearly equals SF (circular)\n"
                "- Context clearly contradicts the SF->LF meaning\n"
                "- SF is a proper noun/company name without medical abbreviation usage\n"
                "- SF appears as regular word in context, not as abbreviation\n\n"
                "AMBIGUOUS only if ALL of these:\n"
                "- Context is insufficient AND\n"
                "- No explicit pair pattern AND\n"
                "- Does not match any REJECTED rule\n\n"
                ">>> IMPORTANT: If uncertain between AMBIGUOUS and REJECTED, choose REJECTED with low confidence. <<<\n\n"
                "=== CANDIDATES ===\n{candidates}\n\n"
                "=== OUTPUT CONTRACT ===\n"
                "Return a single JSON object (not an array at root level):\n"
                "{{\n"
                '  "expected_count": {count},\n'
                '  "results": [\n'
                '    {{"id": "<id from input>", "status": "VALIDATED|REJECTED|AMBIGUOUS", '
                '"confidence": 0.0-1.0, "reason": "<=12 words"}},\n'
                "    ...\n"
                "  ]\n"
                "}}\n\n"
                "HARD CONSTRAINTS:\n"
                "- Output must be valid JSON\n"
                "- results array must have exactly {count} items\n"
                "- Each result must include the id from input\n"
                "- Never return a bare array; always use the wrapper object"
            ),
            "schema": None,
        },
        # -------------------------
        # Fast Reject (Haiku screening)
        # -------------------------
        (PromptTask.FAST_REJECT, "v1.0"): {
            "system": (
                "You are a fast screening filter for medical abbreviations. "
                "Your job is to REJECT obvious non-abbreviations. "
                "When in doubt, return REVIEW (let the main validator decide). "
                "Return ONLY valid JSON. No markdown."
            ),
            "user": (
                "Screen these candidates. Only REJECT if you are VERY confident (>=0.9).\n\n"
                "=== REJECT ONLY IF ===\n"
                "- SF is a common English word used normally (not as abbreviation)\n"
                "- SF is a proper noun/company name (Novartis, Roche, Lancet)\n"
                "- LF is circular (equals or contains only SF)\n"
                "- LF is clearly not a definition (random phrase, incomplete)\n"
                "- SF appears in context as regular word, not abbreviation\n\n"
                "=== REVIEW (default) ===\n"
                "- Any doubt -> REVIEW\n"
                "- SF looks like abbreviation but unsure about LF -> REVIEW\n"
                "- Medical/scientific terms -> REVIEW\n\n"
                "=== CANDIDATES ===\n{candidates}\n\n"
                "=== OUTPUT ===\n"
                "Return JSON object:\n"
                "{{\n"
                '  "results": [\n'
                '    {{"id": "<id>", "decision": "REJECT|REVIEW", "confidence": 0.0-1.0, '
                '"reason": "<=8 words"}},\n'
                "    ...\n"
                "  ]\n"
                "}}\n\n"
                "IMPORTANT: Prefer REVIEW when uncertain. Only REJECT obvious cases."
            ),
            "schema": None,
        },
        # -------------------------
        # Disease validation
        # -------------------------
        (PromptTask.VERIFY_DISEASE, "v1.0"): {
            "system": (
                "You are a clinical document auditor validating disease mentions. "
                "Your task is to verify that detected text spans correctly refer to diseases. "
                "Distinguish diseases from chromosomes, genes, measurements, and other entities. "
                "Return JSON only."
            ),
            "user": (
                "Context:\n{context}\n\n"
                "Detected text: '{matched_text}'\n"
                "Proposed disease: '{preferred_label}'\n"
                "Medical codes: {codes}\n"
                "{provenance}\n\n"
                "Decide if this text span is a valid disease mention.\n\n"
                "=== VALIDATED if ===\n"
                "- Text clearly refers to a disease, disorder, syndrome, or medical condition\n"
                "- The proposed disease name matches the context meaning\n"
                "- Medical codes are appropriate for this disease\n"
                "- Disease abbreviations (PAH, IgAN, ANCA) used in medical context\n\n"
                "=== REJECTED if ===\n"
                "- Text refers to a chromosome number or band (e.g., '10p' in 'chromosome 10p deletion')\n"
                "- Text refers to a gene name used as gene (e.g., 'BRCA1 mutation' - gene, not disease)\n"
                "- Text is a measurement, cell type, protein, or other non-disease entity\n"
                "- Disease name is clearly wrong for this context\n"
                "- Text is a karyotype notation (45,X, 46,XX, etc.) not in disease context\n\n"
                "=== AMBIGUOUS if ===\n"
                "- Insufficient context to determine disease vs non-disease\n"
                "- Text could be either depending on interpretation\n\n"
                "Return JSON:\n"
                "{{\n"
                '"status": "VALIDATED|REJECTED|AMBIGUOUS",\n'
                '"confidence": 0.0-1.0,\n'
                '"evidence": "quote from context",\n'
                '"reason": "brief explanation",\n'
                '"corrected_disease": null | "correct disease name if wrong",\n'
                '"is_disease": true|false\n'
                "}}"
            ),
            "schema": None,
        },
        # -------------------------
        # Disease batch validation
        # -------------------------
        (PromptTask.VERIFY_DISEASE_BATCH, "v1.0"): {
            "system": (
                "You validate disease mentions in clinical documents. "
                "Distinguish diseases from chromosomes, genes, and other entities. "
                "Return ONLY valid JSON. No markdown. No extra text."
            ),
            "user": (
                "Task: Validate disease mentions independently.\n\n"
                "=== INPUT FIELDS ===\n"
                "- matched_text: Text found in document\n"
                "- preferred_label: Proposed disease name\n"
                "- codes: Medical codes (ICD-10, SNOMED, ORPHA, etc.)\n"
                "- source: Lexicon source (specialized vs general)\n"
                "- is_rare_disease: Flag from lexicon\n\n"
                "=== DECISION RULES ===\n\n"
                "VALIDATED if ANY:\n"
                "- Text is a known disease name in medical context\n"
                "- Disease abbreviation (PAH, IgAN, ANCA-GN) used appropriately\n"
                "- source='specialized' (PAH/ANCA/IgAN lexicons - high trust)\n"
                "- Context clearly indicates disease/condition/syndrome\n\n"
                "REJECTED if ANY:\n"
                "- Text is chromosome notation (10p, 22q11, 46,XX) in cytogenetic context\n"
                "- Text is gene name discussing gene/mutation (not gene-associated disease)\n"
                "- Text is measurement, lab value, cell type, or protein\n"
                "- Context clearly contradicts disease interpretation\n"
                "- Matched text is too generic (e.g., single letters, numbers)\n\n"
                "AMBIGUOUS only if:\n"
                "- Context insufficient AND does not match REJECTED rules\n\n"
                ">>> When uncertain between AMBIGUOUS and REJECTED, choose REJECTED. <<<\n\n"
                "=== CANDIDATES ===\n{candidates}\n\n"
                "=== OUTPUT CONTRACT ===\n"
                "Return JSON object:\n"
                "{{\n"
                '  "expected_count": {count},\n'
                '  "results": [\n'
                '    {{"id": "<id>", "status": "VALIDATED|REJECTED|AMBIGUOUS", '
                '"confidence": 0.0-1.0, "reason": "<=12 words", "is_disease": true|false}},\n'
                "    ...\n"
                "  ]\n"
                "}}\n\n"
                "HARD CONSTRAINTS:\n"
                "- Output must be valid JSON\n"
                "- results array must have exactly {count} items\n"
                "- Each result must include the id from input"
            ),
            "schema": None,
        },
        # -------------------------
        # Author batch validation
        # -------------------------
        (PromptTask.VERIFY_AUTHOR_BATCH, "v1.0"): {
            "system": (
                "You validate author/investigator mentions in clinical documents. "
                "Return ONLY valid JSON. No markdown. No extra text."
            ),
            "user": (
                "Task: Validate author/investigator mentions independently.\n\n"
                "=== INPUT FIELDS ===\n"
                "- full_name: Detected person name\n"
                "- role: Detected role (author, principal_investigator, etc.)\n"
                "- context: Text surrounding the mention\n"
                "- source: Detection method (header_pattern, regex, etc.)\n\n"
                "=== DECISION RULES ===\n\n"
                "VALIDATED if ANY:\n"
                "- Name appears with credentials (MD, PhD, etc.) in author context\n"
                "- Name explicitly listed as author, investigator, or committee member\n"
                "- Name appears in byline or author list section\n"
                "- Name associated with affiliation/institution in academic context\n\n"
                "REJECTED if ANY:\n"
                "- Text is not a person's name (institution, company, journal)\n"
                "- Name appears only in reference/citation context (not as author of this doc)\n"
                "- Name is part of a disease/condition name (e.g., 'Parkinson', 'Alzheimer')\n"
                "- Context clearly indicates this is not an author attribution\n\n"
                "AMBIGUOUS only if:\n"
                "- Insufficient context to determine if person is an author\n"
                "- Name could be author or cited reference author\n\n"
                "=== CANDIDATES ===\n{candidates}\n\n"
                "=== OUTPUT CONTRACT ===\n"
                "Return JSON object:\n"
                "{{\n"
                '  "expected_count": {count},\n'
                '  "results": [\n'
                '    {{"id": "<id>", "status": "VALIDATED|REJECTED|AMBIGUOUS", '
                '"confidence": 0.0-1.0, "reason": "<=12 words", "is_author": true|false}},\n'
                "    ...\n"
                "  ]\n"
                "}}\n\n"
                "HARD CONSTRAINTS:\n"
                "- Output must be valid JSON\n"
                "- results array must have exactly {count} items\n"
                "- Each result must include the id from input"
            ),
            "schema": None,
        },
        # -------------------------
        # Citation batch validation
        # -------------------------
        (PromptTask.VERIFY_CITATION_BATCH, "v1.0"): {
            "system": (
                "You validate citation/reference identifiers in clinical documents. "
                "Return ONLY valid JSON. No markdown. No extra text."
            ),
            "user": (
                "Task: Validate citation identifiers independently.\n\n"
                "=== INPUT FIELDS ===\n"
                "- identifier_type: Type of ID (pmid, pmcid, doi, nct, url)\n"
                "- identifier_value: The detected identifier\n"
                "- citation_text: Surrounding citation text\n"
                "- context: Text surrounding the mention\n\n"
                "=== DECISION RULES ===\n\n"
                "VALIDATED if ANY:\n"
                "- PMID/PMCID follows standard format (7-8 digits)\n"
                "- DOI follows standard format (10.xxxx/...)\n"
                "- NCT follows ClinicalTrials.gov format (NCT + 8 digits)\n"
                "- Identifier appears in reference section or citation context\n"
                "- URL points to valid academic/medical resource\n\n"
                "REJECTED if ANY:\n"
                "- Identifier format is invalid\n"
                "- Number is clearly not a citation ID (e.g., page number, year)\n"
                "- URL is clearly not a citation (e.g., institution homepage)\n"
                "- Context indicates this is not a bibliographic reference\n\n"
                "AMBIGUOUS only if:\n"
                "- Insufficient context to determine if this is a citation\n"
                "- Format is valid but usage context is unclear\n\n"
                "=== CANDIDATES ===\n{candidates}\n\n"
                "=== OUTPUT CONTRACT ===\n"
                "Return JSON object:\n"
                "{{\n"
                '  "expected_count": {count},\n'
                '  "results": [\n'
                '    {{"id": "<id>", "status": "VALIDATED|REJECTED|AMBIGUOUS", '
                '"confidence": 0.0-1.0, "reason": "<=12 words", "is_citation": true|false}},\n'
                "    ...\n"
                "  ]\n"
                "}}\n\n"
                "HARD CONSTRAINTS:\n"
                "- Output must be valid JSON\n"
                "- results array must have exactly {count} items\n"
                "- Each result must include the id from input"
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
        bundle_hash = compute_prompt_bundle_hash(
            system_prompt, user_template, schema, params
        )

        return PromptBundle(
            task=task,
            version=version,
            system_prompt=system_prompt,
            user_template=user_template,
            output_schema=schema,
            prompt_bundle_hash=bundle_hash,
        )


__all__ = [
    "PromptTask",
    "PromptBundle",
    "PromptRegistry",
]

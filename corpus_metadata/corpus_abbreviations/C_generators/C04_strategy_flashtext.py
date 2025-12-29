# corpus_metadata/corpus_abbreviations/C_generators/C04_strategy_flashtext.py

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

from flashtext import KeywordProcessor

from A_core.A02_interfaces import BaseCandidateGenerator
from A_core.A01_domain_models import (
    Candidate,
    Coordinate,
    FieldType,
    GeneratorType,
    ProvenanceMetadata,
)
from B_parsing.B02_doc_graph import DocumentGraph, ContentRole


class FlashTextCandidateGenerator(BaseCandidateGenerator):
    """
    LEXICON MATCHER (Abbreviation pipeline)
    - Loads ONE lexicon file: abbreviation_lexicon.json (by default)
    - Matches known short forms in text and emits DEFINITION_PAIR candidates.

    Nota: la “gestión de N lexicons” (merge, priority, etc.) se hará fuera de este módulo.
    """

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config or {})
        self.config = config or {}

        self.lexicon_path = Path(self.config.get("lexicon_path", "abbreviation_lexicon.json"))
        self.context_window = int(self.config.get("context_window", 80))

        # Dos tries: case-sensitive y case-insensitive (porque FlashText no soporta flag por-keyword)
        self.kp_cs = KeywordProcessor(case_sensitive=True)
        self.kp_ci = KeywordProcessor(case_sensitive=False)

        # Auditoría: SF::LF -> {"source": "...", "case": "cs/ci"}
        self.source_map: Dict[str, Dict[str, str]] = {}

        self._load_one_lexicon(self.lexicon_path)

    # -----------------------
    # Public Methods
    # -----------------------
    def extract(self, doc: DocumentGraph) -> List[Candidate]:
        out: List[Candidate] = []

        # fingerprints mínimos (ideal: el Orchestrator te los pasa bien)
        pipeline_version = str(self.config.get("pipeline_version", "DEV"))
        run_id = str(self.config.get("run_id", "RUN_DEV"))
        doc_fingerprint = str(self.config.get("doc_fingerprint", "UNKNOWN_SHA256"))

        for page_num in sorted(doc.pages.keys()):
            page = doc.pages[page_num]
            blocks = sorted(page.blocks, key=lambda b: b.reading_order_index)

            for block in blocks:
                if block.role in (ContentRole.PAGE_HEADER, ContentRole.PAGE_FOOTER):
                    continue
                text = (block.text or "").strip()
                if not text:
                    continue

                # 1) Case-sensitive hits
                cs_hits = self.kp_cs.extract_keywords(text, span_info=True)
                # 2) Case-insensitive hits
                ci_hits = self.kp_ci.extract_keywords(text, span_info=True)

                # unify (LF, start, end)
                seen_spans = set()
                all_hits = []
                for lf, s, e in cs_hits + ci_hits:
                    key = (str(lf), int(s), int(e))
                    if key in seen_spans:
                        continue
                    seen_spans.add(key)
                    all_hits.append((str(lf), int(s), int(e)))

                for long_form, start, end in all_hits:
                    short_form = text[start:end]

                    ctx = self._make_context(text, start, end, window=self.context_window)

                    scope_ref = self._hash_text(ctx)
                    prov = ProvenanceMetadata(
                        pipeline_version=pipeline_version,
                        run_id=run_id,
                        doc_fingerprint=doc_fingerprint,
                        generator_name=GeneratorType.LEXICON_MATCH,
                        rule_version=str(self.config.get("rule_version", "lexicon_v1")),
                        prompt_bundle_hash=None,
                        context_hash=scope_ref,
                        llm_config=None,
                    )

                    cand = Candidate(
                        doc_id=doc.doc_id,
                        field_type=FieldType.DEFINITION_PAIR,
                        generator_type=GeneratorType.LEXICON_MATCH,
                        short_form=short_form,
                        long_form=long_form,
                        context_text=ctx,
                        context_location=Coordinate(page_num=page_num, bbox=block.bbox),
                        initial_confidence=float(self.config.get("confidence", 0.99)),
                        provenance=prov,
                    )
                    out.append(cand)

        return out

    # -----------------------
    # Lexicon loading
    # -----------------------
    def _load_one_lexicon(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Lexicon not found: {path}")

        data = json.loads(path.read_text(encoding="utf-8"))

        for sf, lf, case_insensitive, variants in self._iter_entries(data):
            self._register(sf, lf, case_insensitive=case_insensitive, variants=variants, source=str(path.name))

    def _iter_entries(self, data: Any) -> Iterable[Tuple[str, str, bool, List[str]]]:
        """
        Yields tuples: (short_form, long_form, case_insensitive, variants)

        Supported shapes:
        1) {"SF": "LF"}
        2) {"SF": ["LF1", "LF2"]} -> takes LF1
        3) {"SF": {"canonical_expansion": "...", "case_insensitive": true, "variants": [...]}}
        4) [{"short_form": "...", "long_form": "..."}, ...]
        5) [{"short": "...", "long": "..."}, ...]
        """
        if isinstance(data, dict):
            for sf, payload in data.items():
                if not sf:
                    continue

                # (1) simple dict
                if isinstance(payload, str):
                    yield sf, payload, False, []

                # (2) list of expansions
                elif isinstance(payload, list):
                    if payload and isinstance(payload[0], str):
                        yield sf, payload[0], False, []
                    else:
                        # list-of-objects? ignore for now
                        continue

                # (3) complex object (your snapshot style)
                elif isinstance(payload, dict):
                    lf = (
                        payload.get("canonical_expansion")
                        or (payload.get("expansions", [{}])[0].get("expansion") if isinstance(payload.get("expansions"), list) else None)
                        or payload.get("long_form")
                        or payload.get("long")
                    )
                    if not lf or not isinstance(lf, str):
                        continue

                    case_ins = bool(payload.get("case_insensitive", False))
                    variants = payload.get("variants") or []
                    if not isinstance(variants, list):
                        variants = []
                    variants = [v for v in variants if isinstance(v, str) and v.strip()]
                    yield sf, lf, case_ins, variants

        elif isinstance(data, list):
            for obj in data:
                if not isinstance(obj, dict):
                    continue
                sf = obj.get("short_form") or obj.get("short")
                lf = obj.get("long_form") or obj.get("long")
                if not sf or not lf:
                    continue
                case_ins = bool(obj.get("case_insensitive", False))
                variants = obj.get("variants") or []
                if not isinstance(variants, list):
                    variants = []
                variants = [v for v in variants if isinstance(v, str) and v.strip()]
                yield str(sf), str(lf), case_ins, variants

    def _register(self, sf: str, lf: str, case_insensitive: bool, variants: List[str], source: str) -> None:
        sf = sf.strip()
        lf = lf.strip()
        if len(sf) < 2 or not lf:
            return

        # register main + variants
        keys = [sf] + variants

        for key in keys:
            key = key.strip()
            if not key:
                continue

            if case_insensitive:
                self.kp_ci.add_keyword(key, lf)
                self.source_map[f"{key}::{lf}"] = {"source": source, "case": "ci"}
            else:
                self.kp_cs.add_keyword(key, lf)
                self.source_map[f"{key}::{lf}"] = {"source": source, "case": "cs"}

    # -----------------------
    # Helpers
    # -----------------------
    def _make_context(self, text: str, start: int, end: int, window: int) -> str:
        left = max(0, start - window)
        right = min(len(text), end + window)
        return text[left:right].replace("\n", " ").strip()

    def _hash_text(self, s: str) -> str:
        return hashlib.sha256((s or "").encode("utf-8")).hexdigest()

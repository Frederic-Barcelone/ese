# corpus_metadata/corpus_metadata/F_evaluation/F02_scorer.py

"""
Abbreviation Extraction Scorer

Computes precision, recall, and F1 by comparing system output against gold annotations.
Uses set-based matching on (short_form, long_form) pairs.

Classification:
    - True Positive: system pair matches gold pair
    - False Positive: system extracted pair not in gold
    - False Negative: gold pair not found by system

Metrics:
    - Precision: TP / (TP + FP) -  how much system output is correct
    - Recall: TP / (TP + FN) -  how much gold was found
    - F1: harmonic mean of precision and recall

Evaluation modes:
    - evaluate_doc(): single document
    - evaluate_corpus(): micro (global) and macro (per-doc average) scores

Configuration (ScorerConfig):
    - require_long_form_match: match SF+LF pairs vs SF-only
    - only_validated: skip non-VALIDATED entities
    - allow_sf_only_gold: accept gold entries without long_form

Depends on F01_gold_loader.py for GoldAnnotation format.
"""

from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field

from A_core.A01_domain_models import (
    ExtractedEntity,
    FieldType,
    ValidationStatus,
)
from F_evaluation.F01_gold_loader import GoldAnnotation


Pair = Tuple[str, Optional[str]]  # (SF, LF) where LF can be None for SF-only truth


class ScoreReport(BaseModel):
    """
    Aggregate metrics for a single evaluation run.
    """

    precision: float
    recall: float
    f1: float

    true_positives: int
    false_positives: int
    false_negatives: int

    # Debug samples
    fp_examples: List[str] = Field(default_factory=list)
    fn_examples: List[str] = Field(default_factory=list)

    # Optional: keep raw sets for deeper debugging (can be big)
    tp_set: Optional[List[str]] = None
    fp_set: Optional[List[str]] = None
    fn_set: Optional[List[str]] = None

    model_config = ConfigDict(extra="forbid")


class CorpusScoreReport(BaseModel):
    """
    Corpus-level evaluation:
      - micro: global set logic across all docs
      - macro: average over per-doc scores (equal weight per doc)
    """

    micro: ScoreReport
    macro: ScoreReport
    per_doc: Dict[str, ScoreReport] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class ScorerConfig(BaseModel):
    """
    Configuration knobs to control strictness.
    """

    # Only evaluate these entity field types. Default: definitions (explicit) + glossary entries.
    include_field_types: Set[FieldType] = Field(
        default_factory=lambda: {FieldType.DEFINITION_PAIR, FieldType.GLOSSARY_ENTRY}
    )

    # If True, system long_form must match gold long_form (after normalization).
    # If False, match by short_form only (useful if your gold has LF missing or you evaluate SF detection).
    require_long_form_match: bool = True

    # If gold long_form is missing/None/"UNKNOWN", treat it as SF-only truth (only SF must match).
    allow_sf_only_gold: bool = True

    # If system long_form is missing for a DEFINITION_PAIR/GLOSSARY_ENTRY, normally that's invalid;
    # but in evaluation you can decide whether to keep it (e.g., to score orphan discovery).
    allow_missing_system_lf: bool = False

    # Whether to only evaluate entities validated by the pipeline
    only_validated: bool = True

    # Fuzzy matching for long forms. If True, considers LFs as matching if:
    # - One is a substring of the other, OR
    # - Similarity ratio >= fuzzy_threshold
    # This handles variations like "US Food and Drug Administration" vs "Food and Drug Administration"
    fuzzy_long_form_match: bool = True

    # Minimum similarity ratio for fuzzy matching (0.0-1.0). Default 0.8 = 80% similar.
    fuzzy_threshold: float = 0.8

    # Debug set dumps (can be large). Usually False in CI logs.
    include_sets_in_report: bool = False

    model_config = ConfigDict(extra="forbid")


class Scorer:
    """
    Calculates precision/recall/F1 using set logic.

    Matching logic:
      - Normal mode: compare (SF, LF) pairs (case/whitespace normalized).
      - SF-only mode: compare just SF when gold LF is missing/UNKNOWN and allow_sf_only_gold=True.

    Notes:
      - For clean evaluation, pass system_output and gold_truth *for the same doc_id* into evaluate_doc().
      - For corpus evaluation, use evaluate_corpus() with mixed doc_ids.
    """

    def __init__(self, config: Optional[ScorerConfig] = None):
        self.config = config or ScorerConfig()

    # -------------------------
    # Public API
    # -------------------------

    def evaluate_doc(
        self,
        system_output: List[ExtractedEntity],
        gold_truth: List[GoldAnnotation],
        *,
        doc_id: Optional[str] = None,
    ) -> ScoreReport:
        """
        Evaluate one document.

        If doc_id is provided, we filter both system and gold to that doc_id.
        """
        sys_items = system_output
        gold_items = gold_truth

        if doc_id:
            sys_items = [e for e in sys_items if e.doc_id == doc_id]
            gold_items = [g for g in gold_items if g.doc_id == doc_id]

        sys_set = self._system_to_set(sys_items)
        gold_set = self._gold_to_set(gold_items)

        tp_set, fp_set, fn_set = self._compare_sets(sys_set, gold_set)

        report = self._build_report(tp_set, fp_set, fn_set)
        return report

    def evaluate_corpus(
        self,
        system_output: List[ExtractedEntity],
        gold_truth: List[GoldAnnotation],
    ) -> CorpusScoreReport:
        """
        Evaluate an entire corpus across multiple docs.

        Returns:
          - micro: global TP/FP/FN computed on union sets across all docs
          - macro: average of per-doc metrics
          - per_doc: dict(doc_id -> ScoreReport)
        """
        # Group by doc_id
        sys_by_doc: Dict[str, List[ExtractedEntity]] = defaultdict(list)
        for e in system_output:
            sys_by_doc[e.doc_id].append(e)

        gold_by_doc: Dict[str, List[GoldAnnotation]] = defaultdict(list)
        for g in gold_truth:
            gold_by_doc[g.doc_id].append(g)

        all_doc_ids = sorted(set(sys_by_doc.keys()) | set(gold_by_doc.keys()))

        per_doc: Dict[str, ScoreReport] = {}
        for did in all_doc_ids:
            per_doc[did] = self.evaluate_doc(
                sys_by_doc.get(did, []), gold_by_doc.get(did, []), doc_id=None
            )

        # Micro (global)
        sys_set_all = self._system_to_set(system_output)
        gold_set_all = self._gold_to_set(gold_truth)
        tp_set, fp_set, fn_set = self._compare_sets(sys_set_all, gold_set_all)
        micro = self._build_report(tp_set, fp_set, fn_set)

        # Macro (average over docs)
        macro = self._macro_average(list(per_doc.values()))

        return CorpusScoreReport(micro=micro, macro=macro, per_doc=per_doc)

    # -------------------------
    # Normalization helpers
    # -------------------------

    def _norm_sf(self, sf: str) -> str:
        """
        Normalize short form for comparison.

        - Strips whitespace
        - Uppercases
        - Removes hyphens/dashes for matching (SC5B-9 == SC5B9)
        """
        s = (sf or "").strip().upper()
        # Remove hyphens/dashes to normalize (SC5B-9 -> SC5B9)
        s = re.sub(r"[-–—]", "", s)
        return s

    def _norm_lf(self, lf: Optional[str]) -> Optional[str]:
        if lf is None:
            return None
        s = " ".join(str(lf).strip().split())
        if not s:
            return None
        return s.lower()

    def _is_unknown_lf(self, lf: Optional[str]) -> bool:
        if lf is None:
            return True
        return self._norm_lf(lf) in {"unknown", "unk", "n/a", "na"}

    def _lf_matches(self, sys_lf: Optional[str], gold_lf: Optional[str]) -> bool:
        """
        Check if system long form matches gold long form.

        Matching criteria (in order):
        1. Exact match (after normalization)
        2. If fuzzy_long_form_match enabled:
           - Substring match (one contains the other)
           - Similarity ratio >= fuzzy_threshold
        """
        # Normalize both
        sys_norm = self._norm_lf(sys_lf)
        gold_norm = self._norm_lf(gold_lf)

        # Both None = match
        if sys_norm is None and gold_norm is None:
            return True

        # One None, other not = no match
        if sys_norm is None or gold_norm is None:
            return False

        # Exact match
        if sys_norm == gold_norm:
            return True

        # Fuzzy matching if enabled
        if self.config.fuzzy_long_form_match:
            # Substring match: one contains the other
            if sys_norm in gold_norm or gold_norm in sys_norm:
                return True

            # Similarity ratio
            ratio = SequenceMatcher(None, sys_norm, gold_norm).ratio()
            if ratio >= self.config.fuzzy_threshold:
                return True

        return False

    # -------------------------
    # Convert inputs to comparable sets
    # -------------------------

    def _system_to_set(self, system_output: List[ExtractedEntity]) -> Set[Pair]:
        out: Set[Pair] = set()

        for ent in system_output:
            if self.config.only_validated and ent.status != ValidationStatus.VALIDATED:
                continue

            if ent.field_type not in self.config.include_field_types:
                continue

            sf = self._norm_sf(ent.short_form)
            lf = self._norm_lf(ent.long_form)

            if not sf:
                continue

            # If LF is missing and we don't allow missing LF for definition-like outputs, skip
            if (
                lf is None
                and not self.config.allow_missing_system_lf
                and ent.field_type
                in (
                    FieldType.DEFINITION_PAIR,
                    FieldType.GLOSSARY_ENTRY,
                )
            ):
                continue

            if self.config.require_long_form_match:
                out.add((sf, lf))
            else:
                # SF-only scoring mode
                out.add((sf, None))

        return out

    def _gold_to_set(self, gold_truth: List[GoldAnnotation]) -> Set[Pair]:
        out: Set[Pair] = set()

        for g in gold_truth:
            sf = self._norm_sf(g.short_form)
            lf = self._norm_lf(g.long_form)

            if not sf:
                continue

            # SF-only truth allowed if LF missing/UNKNOWN
            if self._is_unknown_lf(lf) and self.config.allow_sf_only_gold:
                out.add((sf, None))
                continue

            if self.config.require_long_form_match:
                out.add((sf, lf))
            else:
                out.add((sf, None))

        return out

    # -------------------------
    # Set comparison
    # -------------------------

    def _compare_sets(
        self, sys_set: Set[Pair], gold_set: Set[Pair]
    ) -> Tuple[Set[Pair], Set[Pair], Set[Pair]]:
        """
        Handles the special case where gold contains SF-only entries (LF=None).
        If gold has (SF, None), then any system pair with that SF counts as TP.

        With fuzzy_long_form_match enabled, LFs are matched using substring/similarity
        rather than exact string matching.
        """
        if not gold_set:
            return set(), set(sys_set), set()

        # If we have SF-only gold items, do SF-based matching for those
        sf_only_gold = {sf for (sf, lf) in gold_set if lf is None}
        gold_full = [(sf, lf) for (sf, lf) in gold_set if lf is not None]

        tp: Set[Pair] = set()
        fp: Set[Pair] = set()
        fn: Set[Pair] = set()

        # Track which gold items have been matched (for FN calculation)
        matched_gold: Set[Pair] = set()
        # Track which system items matched something (for FP calculation)
        matched_sys: Set[Pair] = set()

        # 1) Full pair matching for gold_full (with fuzzy LF matching)
        for gold_sf, gold_lf in gold_full:
            for sys_sf, sys_lf in sys_set:
                if sys_sf != gold_sf:
                    continue
                # SF matches, check LF
                if self._lf_matches(sys_lf, gold_lf):
                    tp.add((gold_sf, gold_lf))
                    matched_gold.add((gold_sf, gold_lf))
                    matched_sys.add((sys_sf, sys_lf))
                    break  # One match per gold item is enough

        # 2) SF-only matching (gold wants SF presence, LF irrelevant)
        if sf_only_gold:
            for sf, lf in sys_set:
                if sf in sf_only_gold:
                    tp.add((sf, None))  # count as satisfied SF-only truth
                    matched_sys.add((sf, lf))

        # 3) False negatives:
        # - for full gold pairs: those not matched
        for gold_pair in gold_full:
            if gold_pair not in matched_gold:
                fn.add(gold_pair)
        # - for SF-only gold: if no system item has that SF
        for sf in sf_only_gold:
            if not any(s == sf for (s, _lf) in sys_set):
                fn.add((sf, None))

        # 4) False positives:
        # system items not matched by any gold item
        for pair in sys_set:
            sf, lf = pair
            if pair in matched_sys:
                continue
            if sf in sf_only_gold:
                continue
            fp.add(pair)

        return tp, fp, fn

    # -------------------------
    # Report building
    # -------------------------

    def _build_report(
        self, tp_set: Set[Pair], fp_set: Set[Pair], fn_set: Set[Pair]
    ) -> ScoreReport:
        tp = len(tp_set)
        fp = len(fp_set)
        fn = len(fn_set)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        # Pretty examples
        fp_examples = [self._pair_to_str(p) for p in sorted(fp_set)[:10]]
        fn_examples = [self._pair_to_str(p) for p in sorted(fn_set)[:10]]

        report = ScoreReport(
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            fp_examples=fp_examples,
            fn_examples=fn_examples,
        )

        if self.config.include_sets_in_report:
            report.tp_set = [self._pair_to_str(p) for p in sorted(tp_set)]
            report.fp_set = [self._pair_to_str(p) for p in sorted(fp_set)]
            report.fn_set = [self._pair_to_str(p) for p in sorted(fn_set)]

        return report

    def _pair_to_str(self, pair: Pair) -> str:
        sf, lf = pair
        if lf is None:
            return f"{sf}::(SF_ONLY)"
        return f"{sf}::{lf}"

    def _macro_average(self, reports: List[ScoreReport]) -> ScoreReport:
        if not reports:
            return ScoreReport(
                precision=0.0,
                recall=0.0,
                f1=0.0,
                true_positives=0,
                false_positives=0,
                false_negatives=0,
            )

        # Macro averages of metrics (equal weight per doc)
        p = sum(r.precision for r in reports) / len(reports)
        r_ = sum(r.recall for r in reports) / len(reports)
        f = sum(r.f1 for r in reports) / len(reports)

        # Sums for counts are still useful to show scale
        tp = sum(r.true_positives for r in reports)
        fp = sum(r.false_positives for r in reports)
        fn = sum(r.false_negatives for r in reports)

        # Collect a few examples across docs (optional; keep small)
        fp_ex = []
        fn_ex = []
        for rep in reports:
            fp_ex.extend(rep.fp_examples[:2])
            fn_ex.extend(rep.fn_examples[:2])
            if len(fp_ex) >= 10 and len(fn_ex) >= 10:
                break

        return ScoreReport(
            precision=round(p, 4),
            recall=round(r_, 4),
            f1=round(f, 4),
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            fp_examples=fp_ex[:10],
            fn_examples=fn_ex[:10],
        )

    # -------------------------
    # Pretty printing
    # -------------------------

    def print_summary(
        self, report: ScoreReport, title: str = "EVALUATION REPORT"
    ) -> None:
        print(f"\n[CHART] --- {title} ---")
        print(f"[OK] Precision: {report.precision:.2%}")
        print(f"[TARGET] Recall:    {report.recall:.2%}")
        print(f"[SCALE]  F1 Score:  {report.f1:.2%}")
        print("-" * 40)
        print(
            f"TP: {report.true_positives} | FP: {report.false_positives} | FN: {report.false_negatives}"
        )

        if report.fp_examples:
            print("\n[WARN]  False Positives (examples):")
            for ex in report.fp_examples:
                print(f"  - {ex}")

        if report.fn_examples:
            print("\n[WARN]  False Negatives (examples):")
            for ex in report.fn_examples:
                print(f"  - {ex}")

    def print_corpus_summary(self, corpus_report: CorpusScoreReport) -> None:
        self.print_summary(corpus_report.micro, title="CORPUS (MICRO)")
        self.print_summary(corpus_report.macro, title="CORPUS (MACRO)")
        print(f"\n[DOC] Docs evaluated: {len(corpus_report.per_doc)}")

# corpus_metadata/corpus_abbreviations/F_evaluation/F02_scorer.py

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

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
    include_field_types: Set[FieldType] = Field(default_factory=lambda: {FieldType.DEFINITION_PAIR, FieldType.GLOSSARY_ENTRY})

    # If True, system long_form must match gold long_form (after normalization).
    # If False, match by short_form only (useful if your gold has LF missing or you evaluate SF detection).
    require_long_form_match: bool = True

    # If gold long_form is missing/None/"UNKNOWN", treat it as SF-only truth (only SF must match).
    allow_sf_only_gold: bool = True

    # If system long_form is missing for a DEFINITION_PAIR/GLOSSARY_ENTRY, normally thatâ€™s invalid;
    # but in evaluation you can decide whether to keep it (e.g., to score orphan discovery).
    allow_missing_system_lf: bool = False

    # Whether to only evaluate entities validated by the pipeline
    only_validated: bool = True

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
            per_doc[did] = self.evaluate_doc(sys_by_doc.get(did, []), gold_by_doc.get(did, []), doc_id=None)

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
        return (sf or "").strip().upper()

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
            if lf is None and not self.config.allow_missing_system_lf and ent.field_type in (
                FieldType.DEFINITION_PAIR,
                FieldType.GLOSSARY_ENTRY,
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

    def _compare_sets(self, sys_set: Set[Pair], gold_set: Set[Pair]) -> Tuple[Set[Pair], Set[Pair], Set[Pair]]:
        """
        Handles the special case where gold contains SF-only entries (LF=None).
        If gold has (SF, None), then any system pair with that SF counts as TP.
        """
        if not gold_set:
            return set(), set(sys_set), set()

        # If we have SF-only gold items, do SF-based matching for those
        sf_only_gold = {sf for (sf, lf) in gold_set if lf is None}
        gold_full = {(sf, lf) for (sf, lf) in gold_set if lf is not None}

        tp: Set[Pair] = set()
        fp: Set[Pair] = set()
        fn: Set[Pair] = set()

        # 1) Full pair matching for gold_full
        tp_full = sys_set.intersection(gold_full)
        tp |= tp_full

        # 2) SF-only matching (gold wants SF presence, LF irrelevant)
        if sf_only_gold:
            for (sf, lf) in sys_set:
                if sf in sf_only_gold:
                    tp.add((sf, None))  # count as satisfied SF-only truth

        # 3) False negatives:
        # - for full gold pairs: those not found
        fn |= (gold_full - tp_full)
        # - for SF-only gold: if no system item has that SF
        for sf in sf_only_gold:
            if not any(s == sf for (s, _lf) in sys_set):
                fn.add((sf, None))

        # 4) False positives:
        # system items not matched by either full-pair TP or SF-only TP rule
        for pair in sys_set:
            sf, lf = pair
            if pair in tp_full:
                continue
            if sf in sf_only_gold:
                continue
            fp.add(pair)

        return tp, fp, fn

    # -------------------------
    # Report building
    # -------------------------

    def _build_report(self, tp_set: Set[Pair], fp_set: Set[Pair], fn_set: Set[Pair]) -> ScoreReport:
        tp = len(tp_set)
        fp = len(fp_set)
        fn = len(fn_set)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

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

    def print_summary(self, report: ScoreReport, title: str = "EVALUATION REPORT") -> None:
        print(f"\nðŸ“Š --- {title} ---")
        print(f"âœ… Precision: {report.precision:.2%}")
        print(f"ðŸ”Ž Recall:    {report.recall:.2%}")
        print(f"âš–ï¸  F1 Score:  {report.f1:.2%}")
        print("-" * 40)
        print(f"TP: {report.true_positives} | FP: {report.false_positives} | FN: {report.false_negatives}")

        if report.fp_examples:
            print("\nâŒ False Positives (examples):")
            for ex in report.fp_examples:
                print(f"  - {ex}")

        if report.fn_examples:
            print("\nâš ï¸  False Negatives (examples):")
            for ex in report.fn_examples:
                print(f"  - {ex}")

    def print_corpus_summary(self, corpus_report: CorpusScoreReport) -> None:
        self.print_summary(corpus_report.micro, title="CORPUS (MICRO)")
        self.print_summary(corpus_report.macro, title="CORPUS (MACRO)")
        print(f"\nðŸ“ Docs evaluated: {len(corpus_report.per_doc)}")
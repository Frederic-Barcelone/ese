# corpus_metadata/corpus_abbreviations/F_evaluation/F06_extraction_analysis.py

"""
Extraction Analysis Report

Compares system extraction results against gold standard and displays a
detailed analysis report on screen.

Report sections:
    - True Positives (TP): correctly extracted abbreviations
    - False Positives (FP): extracted but not in gold
    - False Negatives (FN): in gold but not extracted
    - Rejected: candidates that were extracted but rejected by validation
    - Ambiguous: candidates marked as ambiguous

Output: Screen only (no JSON, no file output)

This module is called automatically after extraction in the orchestrator.
"""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# Type aliases
Pair = Tuple[str, Optional[str]]  # (SF, LF)


class ExtractionAnalyzer:
    """
    Analyzes extraction results against gold standard.
    Displays report to screen only.
    """

    def __init__(
        self,
        fuzzy_threshold: float = 0.8,
    ):
        self.fuzzy_threshold = fuzzy_threshold

    def analyze(
        self,
        results_data: Dict[str, Any],
        gold_path: str,
    ) -> None:
        """
        Run full analysis and print report to screen.

        Args:
            results_data: Extraction results dict (from _export_results)
            gold_path: Path to gold standard JSON file
        """
        # Load gold data
        gold_data = self._load_gold(gold_path)

        # Extract document info
        doc_name = results_data.get("document", "Unknown")

        # Filter gold to this document
        gold_defined = [g for g in gold_data.get("defined", []) if self._match_doc_id(g.get("doc_id", ""), doc_name)]
        gold_mentioned = [g for g in gold_data.get("mentioned", []) if self._match_doc_id(g.get("doc_id", ""), doc_name)]

        # Get extracted abbreviations (validated)
        validated = results_data.get("abbreviations", [])

        # Get counts
        total_candidates = results_data.get("total_candidates", 0)
        total_validated = results_data.get("total_validated", 0)
        total_rejected = results_data.get("total_rejected", 0)
        total_ambiguous = results_data.get("total_ambiguous", 0)

        # Build comparison sets
        sys_set = self._build_system_set(validated)
        gold_set = self._build_gold_set(gold_defined)

        # Compute TP, FP, FN
        tp, fp, fn = self._compare_sets(sys_set, gold_set)

        # Print report
        self._print_header(doc_name)
        self._print_summary(
            total_candidates=total_candidates,
            total_validated=total_validated,
            total_rejected=total_rejected,
            total_ambiguous=total_ambiguous,
            gold_defined_count=len(gold_defined),
            gold_mentioned_count=len(gold_mentioned),
        )
        self._print_metrics(tp, fp, fn)
        self._print_heuristics(results_data.get("heuristics_counters", {}))
        self._print_true_positives(tp)
        self._print_false_positives(fp)
        self._print_false_negatives(fn, gold_defined)
        self._print_mentioned_not_defined(gold_mentioned, validated)

    def _load_gold(self, path: str) -> Dict[str, List[Dict]]:
        """Load gold standard file."""
        gold_path = Path(path)
        if not gold_path.exists():
            return {"defined": [], "mentioned": [], "excluded": []}

        with open(gold_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle different gold file formats
        if "defined_annotations" in data:
            # papers_gold_v2.json format
            return {
                "defined": data.get("defined_annotations", []),
                "mentioned": data.get("mentioned_annotations", []),
                "excluded": data.get("excluded_annotations", []),
            }
        elif "annotations" in data:
            # Standard format
            return {"defined": data.get("annotations", []), "mentioned": [], "excluded": []}
        elif isinstance(data, list):
            return {"defined": data, "mentioned": [], "excluded": []}
        else:
            return {"defined": [], "mentioned": [], "excluded": []}

    def _match_doc_id(self, gold_doc_id: str, target_doc: str) -> bool:
        """Check if gold doc_id matches target document."""
        gold_stem = Path(gold_doc_id).stem.lower()
        target_stem = Path(target_doc).stem.lower()
        return gold_stem == target_stem or gold_stem in target_stem or target_stem in gold_stem

    def _norm_sf(self, sf: str) -> str:
        return (sf or "").strip().upper()

    def _norm_lf(self, lf: Optional[str]) -> Optional[str]:
        if lf is None:
            return None
        s = " ".join(str(lf).strip().split())
        return s.lower() if s else None

    def _lf_matches(self, sys_lf: Optional[str], gold_lf: Optional[str]) -> bool:
        """Check if long forms match (exact or fuzzy)."""
        sys_norm = self._norm_lf(sys_lf)
        gold_norm = self._norm_lf(gold_lf)

        if sys_norm is None and gold_norm is None:
            return True
        if sys_norm is None or gold_norm is None:
            return False
        if sys_norm == gold_norm:
            return True

        # Substring match
        if sys_norm in gold_norm or gold_norm in sys_norm:
            return True

        # Fuzzy match
        ratio = SequenceMatcher(None, sys_norm, gold_norm).ratio()
        return ratio >= self.fuzzy_threshold

    def _build_system_set(self, validated: List[Dict]) -> Set[Pair]:
        """Build set of (SF, LF) pairs from system output."""
        out: Set[Pair] = set()
        for item in validated:
            sf = self._norm_sf(item.get("short_form", ""))
            lf = self._norm_lf(item.get("long_form"))
            if sf:
                out.add((sf, lf))
        return out

    def _build_gold_set(self, gold_items: List[Dict]) -> Set[Pair]:
        """Build set of (SF, LF) pairs from gold standard."""
        out: Set[Pair] = set()
        for item in gold_items:
            sf = self._norm_sf(item.get("short_form", ""))
            lf = self._norm_lf(item.get("long_form"))
            if sf:
                out.add((sf, lf))
        return out

    def _compare_sets(
        self, sys_set: Set[Pair], gold_set: Set[Pair]
    ) -> Tuple[Set[Pair], Set[Pair], Set[Pair]]:
        """Compare system and gold sets, returning TP, FP, FN."""
        tp: Set[Pair] = set()
        fp: Set[Pair] = set()
        fn: Set[Pair] = set()

        matched_gold: Set[Pair] = set()

        # Match system items against gold
        for (sys_sf, sys_lf) in sys_set:
            matched = False
            for (gold_sf, gold_lf) in gold_set:
                if sys_sf != gold_sf:
                    continue
                if self._lf_matches(sys_lf, gold_lf):
                    tp.add((sys_sf, sys_lf))
                    matched_gold.add((gold_sf, gold_lf))
                    matched = True
                    break
            if not matched:
                fp.add((sys_sf, sys_lf))

        # Find unmatched gold items (FN)
        for gold_pair in gold_set:
            if gold_pair not in matched_gold:
                fn.add(gold_pair)

        return tp, fp, fn

    def _print_header(self, doc_name: str) -> None:
        """Print report header."""
        print("\n" + "=" * 70)
        print("EXTRACTION ANALYSIS REPORT (F06)")
        print("=" * 70)
        print(f"Document: {doc_name}")
        print("=" * 70)

    def _print_summary(
        self,
        total_candidates: int,
        total_validated: int,
        total_rejected: int,
        total_ambiguous: int,
        gold_defined_count: int,
        gold_mentioned_count: int,
    ) -> None:
        """Print extraction summary."""
        print("\n--- EXTRACTION SUMMARY ---")
        print(f"  Total candidates generated:  {total_candidates}")
        print(f"  Validated (accepted):        {total_validated}")
        print(f"  Rejected:                    {total_rejected}")
        print(f"  Ambiguous:                   {total_ambiguous}")
        print()
        print(f"  Gold defined abbreviations:  {gold_defined_count}")
        print(f"  Gold mentioned (no def):     {gold_mentioned_count}")

    def _print_metrics(self, tp: Set[Pair], fp: Set[Pair], fn: Set[Pair]) -> None:
        """Print precision/recall/F1 metrics."""
        tp_count = len(tp)
        fp_count = len(fp)
        fn_count = len(fn)

        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) else 0.0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

        print("\n--- METRICS ---")
        print(f"  Precision: {precision:.2%} ({tp_count} / {tp_count + fp_count})")
        print(f"  Recall:    {recall:.2%} ({tp_count} / {tp_count + fn_count})")
        print(f"  F1 Score:  {f1:.2%}")
        print()
        print(f"  True Positives (TP):   {tp_count}")
        print(f"  False Positives (FP):  {fp_count}")
        print(f"  False Negatives (FN):  {fn_count}")

    def _print_heuristics(self, counters: Dict[str, int]) -> None:
        """Print heuristics counters."""
        if not counters:
            return

        print("\n--- HEURISTICS COUNTERS ---")
        for key, value in sorted(counters.items()):
            print(f"  {key}: {value}")

    def _print_true_positives(self, tp: Set[Pair]) -> None:
        """Print true positives list."""
        print("\n--- TRUE POSITIVES (TP) ---")
        if not tp:
            print("  (none)")
            return

        for sf, lf in sorted(tp):
            lf_display = lf or "(no long form)"
            print(f"  [OK] {sf} -> {lf_display}")

    def _print_false_positives(self, fp: Set[Pair]) -> None:
        """Print false positives list."""
        print("\n--- FALSE POSITIVES (FP) ---")
        print("  (Extracted but not in gold standard)")
        if not fp:
            print("  (none)")
            return

        for sf, lf in sorted(fp):
            lf_display = lf or "(no long form)"
            print(f"  [FP] {sf} -> {lf_display}")

    def _print_false_negatives(self, fn: Set[Pair], gold_defined: List[Dict]) -> None:
        """Print false negatives list with context from gold."""
        print("\n--- FALSE NEGATIVES (FN) ---")
        print("  (In gold standard but not extracted)")
        if not fn:
            print("  (none)")
            return

        # Build lookup for gold details
        gold_lookup = {}
        for g in gold_defined:
            sf = self._norm_sf(g.get("short_form", ""))
            lf = self._norm_lf(g.get("long_form"))
            gold_lookup[(sf, lf)] = g

        for sf, lf in sorted(fn):
            lf_display = lf or "(no long form)"
            gold_info = gold_lookup.get((sf, lf), {})
            page = gold_info.get("page", "?")
            print(f"  [MISS] {sf} -> {lf_display} (page {page})")

    def _print_mentioned_not_defined(
        self, gold_mentioned: List[Dict], validated: List[Dict]
    ) -> None:
        """Print mentioned abbreviations and their extraction status."""
        if not gold_mentioned:
            return

        print("\n--- MENTIONED (not defined in gold) ---")
        print("  (Abbreviations used in document without explicit definition)")

        # Build set of extracted SFs
        extracted_sfs = {self._norm_sf(v.get("short_form", "")) for v in validated}

        found_count = 0
        not_found_count = 0

        for g in sorted(gold_mentioned, key=lambda x: x.get("short_form", "")):
            sf = self._norm_sf(g.get("short_form", ""))
            page = g.get("page", "?")

            if sf in extracted_sfs:
                print(f"  [FOUND]  {sf} (page {page}) - extracted with definition")
                found_count += 1
            else:
                print(f"  [MISS]   {sf} (page {page}) - not extracted")
                not_found_count += 1

        print(f"\n  Mentioned found: {found_count}, Mentioned missing: {not_found_count}")


def run_analysis(results_data: Dict[str, Any], gold_path: str) -> None:
    """
    Convenience function to run extraction analysis.
    Called from orchestrator after _export_results.

    Args:
        results_data: The export data dict built in _export_results
        gold_path: Path to gold standard JSON file
    """
    analyzer = ExtractionAnalyzer()
    analyzer.analyze(results_data, gold_path)

# corpus_metadata/corpus_abbreviations/F_evaluation/F02b_scorer_v2.py
"""
Scorer v2: Three-tier metrics aligned with gold audit.

Metrics:
1. SF-only F1 (primary KPI) - target 90-93%
2. Defined Pairs F1 (EXTRACTABLE_PAIR only) - honest pair metric
3. Parser Coverage (% of gold SFs found in parsed text)

Usage:
    python F02b_scorer_v2.py <output_json> <gold_v2_json>
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def normalize_sf(sf: str) -> str:
    """Normalize SF for comparison (uppercase, strip)."""
    return sf.strip().upper()


def normalize_lf(lf: str) -> str:
    """Normalize LF for comparison (lowercase, collapse whitespace)."""
    if not lf:
        return ""
    return " ".join(lf.lower().split())


@dataclass
class MetricReport:
    name: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    tp_items: Set[Any] = field(default_factory=set)
    fp_items: Set[Any] = field(default_factory=set)
    fn_items: Set[Any] = field(default_factory=set)

    def compute(self):
        self.precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        self.recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
        self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall) if (self.precision + self.recall) > 0 else 0.0


@dataclass
class ScorerV2Report:
    sf_only: MetricReport
    defined_pairs: MetricReport
    parser_coverage: float  # % of gold SFs found
    parser_missed: List[str]  # SFs not found

    # Breakdowns
    gold_total: int = 0
    gold_defined: int = 0
    gold_mentioned: int = 0
    gold_excluded: int = 0
    system_total: int = 0


def score_v2(
    system_output_path: str,
    gold_v2_path: str,
    doc_filter: Optional[str] = None,
) -> ScorerV2Report:
    """
    Score system output against reclassified gold.

    Args:
        system_output_path: Path to system output JSON
        gold_v2_path: Path to reclassified gold JSON (from F00b)
        doc_filter: Optional doc_id to filter (e.g., "01_Article_Iptacopan")

    Returns:
        ScorerV2Report with all metrics
    """
    # Load system output
    with open(system_output_path, "r", encoding="utf-8") as f:
        system_data = json.load(f)

    # Load gold v2
    with open(gold_v2_path, "r", encoding="utf-8") as f:
        gold_data = json.load(f)

    # Extract system abbreviations
    sys_abbrevs = system_data.get("abbreviations", [])

    # Filter by doc if specified
    if doc_filter:
        doc_name = system_data.get("document", "")
        if doc_filter not in doc_name:
            # No match, return empty
            return ScorerV2Report(
                sf_only=MetricReport("sf_only"),
                defined_pairs=MetricReport("defined_pairs"),
                parser_coverage=0.0,
                parser_missed=[],
            )

    # Build system sets
    sys_sf_set: Set[str] = set()
    sys_pair_set: Set[Tuple[str, str]] = set()

    for abbr in sys_abbrevs:
        sf = normalize_sf(abbr.get("short_form", ""))
        lf = normalize_lf(abbr.get("long_form", ""))
        if sf:
            sys_sf_set.add(sf)
            if lf:
                sys_pair_set.add((sf, lf))

    # Build gold sets
    defined_annots = gold_data.get("defined_annotations", [])
    mentioned_annots = gold_data.get("mentioned_annotations", [])
    excluded_annots = gold_data.get("excluded_annotations", [])
    all_sf_annots = gold_data.get("all_sf_annotations", [])

    # Filter by doc if specified
    if doc_filter:
        defined_annots = [a for a in defined_annots if doc_filter in a.get("doc_id", "")]
        mentioned_annots = [a for a in mentioned_annots if doc_filter in a.get("doc_id", "")]
        excluded_annots = [a for a in excluded_annots if doc_filter in a.get("doc_id", "")]
        all_sf_annots = [a for a in all_sf_annots if doc_filter in a.get("doc_id", "")]

    # Gold SF set (for SF-only scoring) - excludes PARSE_ISSUE
    gold_sf_set: Set[str] = set()
    for anno in all_sf_annots:
        sf = normalize_sf(anno.get("short_form", ""))
        if sf:
            gold_sf_set.add(sf)

    # Gold pair set (for defined pairs scoring) - only EXTRACTABLE_PAIR
    gold_pair_set: Set[Tuple[str, str]] = set()
    for anno in defined_annots:
        sf = normalize_sf(anno.get("short_form", ""))
        lf = normalize_lf(anno.get("long_form", ""))
        if sf and lf:
            gold_pair_set.add((sf, lf))

    # Parser coverage: how many gold SFs were found?
    all_gold_sfs: Set[str] = set()
    for anno in defined_annots + mentioned_annots + excluded_annots:
        sf = normalize_sf(anno.get("short_form", ""))
        if sf:
            all_gold_sfs.add(sf)

    excluded_sfs: Set[str] = set()
    for anno in excluded_annots:
        sf = normalize_sf(anno.get("short_form", ""))
        if sf:
            excluded_sfs.add(sf)

    # ═══════════════════════════════════════════════════════════════════════════
    # METRIC 1: SF-only (primary KPI)
    # ═══════════════════════════════════════════════════════════════════════════
    sf_tp = sys_sf_set & gold_sf_set
    sf_fp = sys_sf_set - gold_sf_set
    sf_fn = gold_sf_set - sys_sf_set

    sf_report = MetricReport(
        name="sf_only",
        tp=len(sf_tp),
        fp=len(sf_fp),
        fn=len(sf_fn),
        tp_items=sf_tp,
        fp_items=sf_fp,
        fn_items=sf_fn,
    )
    sf_report.compute()

    # ═══════════════════════════════════════════════════════════════════════════
    # METRIC 2: Defined Pairs (honest pair metric)
    # ═══════════════════════════════════════════════════════════════════════════
    pair_tp = sys_pair_set & gold_pair_set
    pair_fp = sys_pair_set - gold_pair_set
    pair_fn = gold_pair_set - sys_pair_set

    pair_report = MetricReport(
        name="defined_pairs",
        tp=len(pair_tp),
        fp=len(pair_fp),
        fn=len(pair_fn),
        tp_items=pair_tp,
        fp_items=pair_fp,
        fn_items=pair_fn,
    )
    pair_report.compute()

    # ═══════════════════════════════════════════════════════════════════════════
    # METRIC 3: Parser Coverage
    # ═══════════════════════════════════════════════════════════════════════════
    parser_found = all_gold_sfs - excluded_sfs
    parser_coverage = len(parser_found) / len(all_gold_sfs) if all_gold_sfs else 1.0
    parser_missed = sorted(excluded_sfs)

    # ═══════════════════════════════════════════════════════════════════════════
    # BUILD REPORT
    # ═══════════════════════════════════════════════════════════════════════════
    report = ScorerV2Report(
        sf_only=sf_report,
        defined_pairs=pair_report,
        parser_coverage=parser_coverage,
        parser_missed=parser_missed,
        gold_total=len(defined_annots) + len(mentioned_annots) + len(excluded_annots),
        gold_defined=len(defined_annots),
        gold_mentioned=len(mentioned_annots),
        gold_excluded=len(excluded_annots),
        system_total=len(sys_abbrevs),
    )

    return report


def print_report(report: ScorerV2Report) -> None:
    """Print formatted scorer report."""
    print()
    print("=" * 70)
    print("SCORER V2 REPORT")
    print("=" * 70)
    print()

    # Summary
    print("GOLD BREAKDOWN:")
    print(f"  Total: {report.gold_total}")
    print(f"  DEFINED (for pair scoring): {report.gold_defined}")
    print(f"  MENTIONED (for SF-only): {report.gold_mentioned}")
    print(f"  EXCLUDED (parser issues): {report.gold_excluded}")
    print(f"  System output: {report.system_total}")
    print()

    # SF-only (primary)
    sf = report.sf_only
    print("─" * 70)
    print("📊 SF-ONLY (Primary KPI) — Target: 90-93%")
    print("─" * 70)
    print(f"  Precision: {sf.precision * 100:5.1f}%")
    print(f"  Recall:    {sf.recall * 100:5.1f}%")
    print(f"  F1:        {sf.f1 * 100:5.1f}%  {'✓' if sf.f1 >= 0.90 else '↑'}")
    print(f"  TP: {sf.tp}, FP: {sf.fp}, FN: {sf.fn}")
    print()

    if sf.fp_items:
        print(f"  FP SFs (not in gold): {sorted(sf.fp_items)[:10]}")
    if sf.fn_items:
        print(f"  FN SFs (missing): {sorted(sf.fn_items)[:10]}")
    print()

    # Defined Pairs
    pair = report.defined_pairs
    print("─" * 70)
    print("📊 DEFINED PAIRS (Honest Pair Metric)")
    print("─" * 70)
    print(f"  Precision: {pair.precision * 100:5.1f}%")
    print(f"  Recall:    {pair.recall * 100:5.1f}%")
    print(f"  F1:        {pair.f1 * 100:5.1f}%")
    print(f"  TP: {pair.tp}, FP: {pair.fp}, FN: {pair.fn}")
    print()

    # Parser Coverage
    print("─" * 70)
    print("📊 PARSER COVERAGE")
    print("─" * 70)
    print(f"  Coverage: {report.parser_coverage * 100:5.1f}%")
    if report.parser_missed:
        print(f"  Missed SFs: {report.parser_missed}")
    print()


def main():
    if len(sys.argv) < 3:
        print("Usage: python F02b_scorer_v2.py <output_json> <gold_v2_json> [doc_filter]")
        print()
        print("Example:")
        print("  python F02b_scorer_v2.py output.json papers_gold_v2.json Iptacopan")
        sys.exit(1)

    output_path = sys.argv[1]
    gold_path = sys.argv[2]
    doc_filter = sys.argv[3] if len(sys.argv) > 3 else None

    report = score_v2(output_path, gold_path, doc_filter)
    print_report(report)


if __name__ == "__main__":
    main()

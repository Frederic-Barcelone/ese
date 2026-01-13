# F05_extraction_analysis.py
"""
Extraction Analysis Report - Version 2 (Simplified)

Clear 2-section structure:
    SECTION 1: GOLD STANDARD - What we SHOULD find (checklist)
    SECTION 2: EXTRACTED     - What we DID find (results)

Plus: Summary metrics at the top for quick overview.
"""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional


class ExtractionAnalyzer:
    """Analyzes extraction results against gold standard."""

    def __init__(self, fuzzy_threshold: float = 0.8):
        self.fuzzy_threshold = fuzzy_threshold

    # -------------------------------------------------------------------------
    # MAIN ENTRY POINT
    # -------------------------------------------------------------------------
    def analyze(self, results_data: Dict[str, Any], gold_path: str) -> Dict[str, Any]:
        """
        Run analysis and print report.

        Returns metrics dict for programmatic use.
        """
        # Load data
        gold_data = self._load_gold(gold_path)
        doc_name = results_data.get("document", "Unknown")

        # Filter gold to this document
        gold_defined = [
            g
            for g in gold_data.get("defined", [])
            if self._match_doc_id(g.get("doc_id", ""), doc_name)
        ]
        gold_mentioned = [
            g
            for g in gold_data.get("mentioned", [])
            if self._match_doc_id(g.get("doc_id", ""), doc_name)
        ]
        gold_diseases = [
            g
            for g in gold_data.get("diseases", [])
            if self._match_doc_id(g.get("doc_id", ""), doc_name)
        ]
        gold_drugs = [
            g
            for g in gold_data.get("drugs", [])
            if self._match_doc_id(g.get("doc_id", ""), doc_name)
        ]

        # Get system extractions
        validated = results_data.get("abbreviations", [])

        # Build comparison data
        comparison = self._build_comparison(validated, gold_defined)

        # Print report
        self._print_report(
            doc_name=doc_name,
            comparison=comparison,
            gold_defined=gold_defined,
            gold_mentioned=gold_mentioned,
            gold_diseases=gold_diseases,
            gold_drugs=gold_drugs,
            validated=validated,
            results_data=results_data,
        )

        return comparison["metrics"]

    # -------------------------------------------------------------------------
    # DATA LOADING
    # -------------------------------------------------------------------------
    def _load_gold(self, path: str) -> Dict[str, List[Dict]]:
        """Load gold standard file (supports multiple formats)."""
        gold_path = Path(path)
        if not gold_path.exists():
            return {"defined": [], "mentioned": [], "excluded": [], "diseases": [], "drugs": []}

        with open(gold_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "defined_annotations" in data:
            return {
                "defined": data.get("defined_annotations", []),
                "mentioned": data.get("mentioned_annotations", []),
                "excluded": data.get("excluded_annotations", []),
                "diseases": data.get("defined_diseases", []),
                "drugs": data.get("defined_drugs", []),
            }
        elif "annotations" in data:
            return {
                "defined": data.get("annotations", []),
                "mentioned": [],
                "excluded": [],
                "diseases": [],
                "drugs": [],
            }
        elif isinstance(data, list):
            return {"defined": data, "mentioned": [], "excluded": [], "diseases": [], "drugs": []}
        return {"defined": [], "mentioned": [], "excluded": [], "diseases": [], "drugs": []}

    def _match_doc_id(self, gold_doc_id: str, target_doc: str) -> bool:
        """Check if document IDs match."""
        gold_stem = Path(gold_doc_id).stem.lower()
        target_stem = Path(target_doc).stem.lower()
        return (
            gold_stem == target_stem
            or gold_stem in target_stem
            or target_stem in gold_stem
        )

    # -------------------------------------------------------------------------
    # NORMALIZATION
    # -------------------------------------------------------------------------
    def _norm_sf(self, sf: str) -> str:
        return (sf or "").strip().upper()

    def _norm_lf(self, lf: Optional[str]) -> Optional[str]:
        if lf is None:
            return None
        s = " ".join(str(lf).strip().split()).lower()
        return s if s else None

    def _lf_matches(self, sys_lf: Optional[str], gold_lf: Optional[str]) -> bool:
        """Check if long forms match (exact, substring, or fuzzy)."""
        sys_norm = self._norm_lf(sys_lf)
        gold_norm = self._norm_lf(gold_lf)

        if sys_norm is None and gold_norm is None:
            return True
        if sys_norm is None or gold_norm is None:
            return False
        if sys_norm == gold_norm:
            return True
        if sys_norm in gold_norm or gold_norm in sys_norm:
            return True

        ratio = SequenceMatcher(None, sys_norm, gold_norm).ratio()
        return ratio >= self.fuzzy_threshold

    # -------------------------------------------------------------------------
    # COMPARISON LOGIC
    # -------------------------------------------------------------------------
    def _build_comparison(
        self, validated: List[Dict], gold_defined: List[Dict]
    ) -> Dict[str, Any]:
        """
        Build comparison between system output and gold standard.

        Returns structured data for both sections of the report.
        """
        # Deduplicate validated by (SF, LF)
        seen = set()
        unique_validated = []
        for item in validated:
            sf = self._norm_sf(item.get("short_form", ""))
            lf = self._norm_lf(item.get("long_form"))
            key = (sf, lf)
            if key not in seen and sf:
                seen.add(key)
                unique_validated.append(
                    {
                        "short_form": sf,
                        "long_form": item.get("long_form"),
                        "long_form_norm": lf,
                    }
                )

        # Deduplicate gold by (SF, LF)
        seen_gold = set()
        unique_gold = []
        for item in gold_defined:
            sf = self._norm_sf(item.get("short_form", ""))
            lf = self._norm_lf(item.get("long_form"))
            key = (sf, lf)
            if key not in seen_gold and sf:
                seen_gold.add(key)
                unique_gold.append(
                    {
                        "short_form": sf,
                        "long_form": item.get("long_form"),
                        "long_form_norm": lf,
                        "page": item.get("page", "?"),
                    }
                )

        # SECTION 1: Check each gold item - was it found?
        gold_results = []
        matched_gold_keys = set()

        for gold_item in unique_gold:
            gold_sf = gold_item["short_form"]
            gold_lf_norm = gold_item["long_form_norm"]

            found = False
            matched_lf = None
            for sys_item in unique_validated:
                if sys_item["short_form"] == gold_sf:
                    if self._lf_matches(sys_item["long_form_norm"], gold_lf_norm):
                        found = True
                        matched_lf = sys_item["long_form"]
                        matched_gold_keys.add((gold_sf, gold_lf_norm))
                        break

            gold_results.append(
                {
                    "short_form": gold_sf,
                    "long_form": gold_item["long_form"],
                    "page": gold_item["page"],
                    "found": found,
                    "matched_lf": matched_lf,
                }
            )

        # SECTION 2: Check each extracted item - does it match gold?
        extracted_results = []

        for sys_item in unique_validated:
            sys_sf = sys_item["short_form"]
            sys_lf_norm = sys_item["long_form_norm"]

            matches_gold = False
            matched_gold_lf = None
            for gold_item in unique_gold:
                if gold_item["short_form"] == sys_sf:
                    if self._lf_matches(sys_lf_norm, gold_item["long_form_norm"]):
                        matches_gold = True
                        matched_gold_lf = gold_item["long_form"]
                        break

            extracted_results.append(
                {
                    "short_form": sys_sf,
                    "long_form": sys_item["long_form"],
                    "matches_gold": matches_gold,
                    "gold_long_form": matched_gold_lf,
                }
            )

        # Sort extracted alphabetically
        extracted_results.sort(key=lambda x: x["short_form"])

        # Calculate metrics
        tp = sum(1 for g in gold_results if g["found"])
        fn = sum(1 for g in gold_results if not g["found"])
        fp = sum(1 for e in extracted_results if not e["matches_gold"])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "gold_results": gold_results,
            "extracted_results": extracted_results,
            "metrics": {
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            },
        }

    # -------------------------------------------------------------------------
    # REPORT PRINTING
    # -------------------------------------------------------------------------
    def _print_report(
        self,
        doc_name: str,
        comparison: Dict[str, Any],
        gold_defined: List[Dict],
        gold_mentioned: List[Dict],
        gold_diseases: List[Dict],
        gold_drugs: List[Dict],
        validated: List[Dict],
        results_data: Dict[str, Any],
    ) -> None:
        """Print the full report."""

        gold_results = comparison["gold_results"]
        extracted_results = comparison["extracted_results"]
        metrics = comparison["metrics"]

        # =====================================================================
        # HEADER
        # =====================================================================
        print("\n")
        print("╔" + "═" * 78 + "╗")
        print("║" + " EXTRACTION ANALYSIS REPORT ".center(78) + "║")
        print("╠" + "═" * 78 + "╣")
        print("║" + f" Document: {doc_name[:66]}".ljust(78) + "║")
        print("╚" + "═" * 78 + "╝")

        # =====================================================================
        # QUICK SUMMARY BOX
        # =====================================================================
        tp = metrics["true_positives"]
        fp = metrics["false_positives"]
        fn = metrics["false_negatives"]
        precision = metrics["precision"]
        recall = metrics["recall"]
        f1 = metrics["f1"]

        # Get disease and drug counts
        disease_count = len(results_data.get("diseases", []))
        drug_count = len(results_data.get("drugs", []))

        print("\n┌─────────────────────────────────────────────────────────────────┐")
        print("│                        QUICK SUMMARY                            │")
        print("├─────────────────────────────────────────────────────────────────┤")
        print(
            f"│  Gold Standard:  {len(gold_results):3} abbreviations to find                      │"
        )
        print(
            f"│  Extracted:      {len(extracted_results):3} abbreviations found                        │"
        )
        if disease_count > 0 or drug_count > 0:
            print(
                f"│  Diseases:       {disease_count:3}  |  Drugs: {drug_count:3}                            │"
            )
        print("├─────────────────────────────────────────────────────────────────┤")
        print(
            f"│  ✓ Correct (TP):     {tp:3}    │  Precision: {precision:6.1%}                │"
        )
        print(
            f"│  ✗ Extra (FP):       {fp:3}    │  Recall:    {recall:6.1%}                │"
        )
        print(
            f"│  ○ Missed (FN):      {fn:3}    │  F1 Score:  {f1:6.1%}                │"
        )
        print("└─────────────────────────────────────────────────────────────────┘")

        # =====================================================================
        # SECTION 1: GOLD STANDARD CHECKLIST
        # =====================================================================
        print("\n")
        print("━" * 80)
        print(" SECTION 1: GOLD STANDARD CHECKLIST ")
        print(" (What we SHOULD find - from human-annotated ground truth)")
        print("━" * 80)

        if not gold_results:
            print("\n  (No gold standard entries for this document)")
        else:
            # Header
            print(f"\n  {'Status':<8} {'Short Form':<15} {'Long Form':<55} {'Page':<5}")
            print("  " + "─" * 85)

            # Sort: found first, then not found
            gold_sorted = sorted(
                gold_results, key=lambda x: (not x["found"], x["short_form"])
            )

            for item in gold_sorted:
                sf = item["short_form"]
                lf = item["long_form"] or "(no definition)"
                page = str(item["page"])

                if item["found"]:
                    status = "  ✓ FOUND"
                    print(f"  {status:<8} {sf:<15} {lf:<55} {page:<5}")
                else:
                    status = "  ✗ MISS "
                    print(
                        f"  \033[91m{status:<8} {sf:<15} {lf:<55} {page:<5}\033[0m"
                    )

            # Summary
            found_count = sum(1 for g in gold_results if g["found"])
            total_count = len(gold_results)
            print("  " + "─" * 85)
            print(
                f"  TOTAL: {found_count}/{total_count} found ({found_count / total_count * 100:.0f}% recall)"
            )

        # =====================================================================
        # SECTION 2: EXTRACTED ABBREVIATIONS
        # =====================================================================
        print("\n")
        print("━" * 80)
        print(" SECTION 2: EXTRACTED ABBREVIATIONS ")
        print(" (What we DID find - sorted alphabetically)")
        print("━" * 80)
        print()
        print("  ✓ MATCH = Extracted AND in gold standard (True Positive)")
        print("  ○ EXTRA = Extracted but NOT in gold standard (False Positive)")
        print()

        if not extracted_results:
            print("\n  (No abbreviations extracted)")
        else:
            # Header
            print(f"\n  {'Status':<10} {'Short Form':<15} {'Extracted Long Form':<55}")
            print("  " + "─" * 85)

            for item in extracted_results:
                sf = item["short_form"]
                lf = item["long_form"] or "(no expansion)"

                if item["matches_gold"]:
                    status = "  ✓ MATCH "
                    print(f"  {status:<10} {sf:<15} {lf:<55}")
                else:
                    status = "  ○ EXTRA "
                    print(f"  \033[93m{status:<10} {sf:<15} {lf:<55}\033[0m")

            # Summary
            match_count = sum(1 for e in extracted_results if e["matches_gold"])
            total_count = len(extracted_results)
            print("  " + "─" * 85)
            print(
                f"  TOTAL: {match_count}/{total_count} match gold ({match_count / total_count * 100:.0f}% precision)"
            )

        # =====================================================================
        # SECTION 3: MENTIONED (BONUS - not scored)
        # =====================================================================
        if gold_mentioned:
            print("\n")
            print("━" * 80)
            print(" SECTION 3: MENTIONED ABBREVIATIONS (not scored) ")
            print(" (Used in document but not explicitly defined - bonus if found)")
            print("━" * 80)

            # Build set of extracted SFs
            extracted_sfs = {e["short_form"] for e in extracted_results}

            # Deduplicate mentioned
            seen_mentioned = set()
            unique_mentioned = []
            for m in gold_mentioned:
                sf = self._norm_sf(m.get("short_form", ""))
                if sf not in seen_mentioned:
                    seen_mentioned.add(sf)
                    unique_mentioned.append(m)

            print(f"\n  {'Status':<10} {'Short Form':<15} {'Expected Long Form':<55}")
            print("  " + "─" * 85)

            found_count = 0
            for m in sorted(unique_mentioned, key=lambda x: x.get("short_form", "")):
                sf = self._norm_sf(m.get("short_form", ""))
                lf = m.get("long_form", "(unknown)")

                if sf in extracted_sfs:
                    print(f"  {'  ✓ FOUND':<10} {sf:<15} {lf:<55}")
                    found_count += 1
                else:
                    print(
                        f"  \033[90m{'  - MISS':<10} {sf:<15} {lf:<55}\033[0m"
                    )

            print("  " + "─" * 85)
            print(f"  BONUS: {found_count}/{len(unique_mentioned)} found")

        # =====================================================================
        # SECTION 4: DISEASE & DRUG EXTRACTIONS (with gold standard eval)
        # =====================================================================
        diseases = results_data.get("diseases", [])
        drugs = results_data.get("drugs", [])

        if diseases or drugs or gold_diseases or gold_drugs:
            print("\n")
            print("━" * 80)
            print(" SECTION 4: ENTITY EXTRACTIONS BY TYPE ")
            print(" (Diseases and Drugs detected vs gold standard)")
            print("━" * 80)

            # Build extracted name sets for matching
            extracted_disease_names = {d.get("name", "").lower() for d in diseases}
            extracted_drug_names = {d.get("name", "").lower() for d in drugs}

            # === DISEASES ===
            if gold_diseases:
                # Evaluate against gold standard
                disease_tp = 0
                disease_fn = 0
                print(f"\n  DISEASES - Gold Standard ({len(gold_diseases)} expected)")
                print("  " + "─" * 85)
                print(f"  {'Status':<10} {'Name':<45} {'Rare?':<6} {'ORPHA':<12}")
                print("  " + "─" * 85)

                for g in gold_diseases:
                    gold_name = g.get("name", "").lower()
                    gold_abbrev = (g.get("abbreviation") or "").lower()
                    # Check if found (by name or abbreviation)
                    found = any(
                        gold_name in ext or ext in gold_name or gold_abbrev in ext
                        for ext in extracted_disease_names
                    )
                    if found:
                        disease_tp += 1
                        status = "✓ FOUND"
                    else:
                        disease_fn += 1
                        status = "✗ MISS"

                    is_rare = "Yes" if g.get("is_rare") else "No"
                    orpha = g.get("orpha") or "-"
                    print(f"  {status:<10} {g.get('name', ''):<45} {is_rare:<6} {orpha:<12}")

                disease_recall = disease_tp / len(gold_diseases) * 100 if gold_diseases else 0
                print("  " + "─" * 85)
                print(f"  Disease Recall: {disease_tp}/{len(gold_diseases)} ({disease_recall:.0f}%)")

            elif diseases:
                print(f"\n  DISEASES ({len(diseases)} found, no gold standard)")
                print("  " + "─" * 85)
                print(f"  {'Name':<50} {'Rare?':<6} {'ICD-10':<12} {'ORPHA':<12}")
                print("  " + "─" * 85)

                seen_diseases = set()
                for d in diseases:
                    name = d.get("name", "")
                    if name.lower() in seen_diseases:
                        continue
                    seen_diseases.add(name.lower())
                    is_rare = "Yes" if d.get("is_rare") else "No"
                    icd10 = d.get("icd10") or "-"
                    orpha = d.get("orpha") or "-"
                    print(f"  {name:<50} {is_rare:<6} {icd10:<12} {orpha:<12}")

            # === DRUGS ===
            if gold_drugs:
                # Evaluate against gold standard
                drug_tp = 0
                drug_fn = 0
                print(f"\n  DRUGS - Gold Standard ({len(gold_drugs)} expected)")
                print("  " + "─" * 85)
                print(f"  {'Status':<10} {'Name':<35} {'Compound':<15} {'Mechanism':<25}")
                print("  " + "─" * 85)

                for g in gold_drugs:
                    gold_name = g.get("name", "").lower()
                    gold_compound = (g.get("compound_id") or "").lower()
                    # Check if found (by name or compound ID)
                    found = any(
                        gold_name in ext or ext in gold_name or (gold_compound and gold_compound in ext)
                        for ext in extracted_drug_names
                    )
                    if found:
                        drug_tp += 1
                        status = "✓ FOUND"
                    else:
                        drug_fn += 1
                        status = "✗ MISS"

                    compound = g.get("compound_id") or "-"
                    mechanism = g.get("mechanism") or "-"
                    print(f"  {status:<10} {g.get('name', ''):<35} {compound:<15} {mechanism:<25}")

                drug_recall = drug_tp / len(gold_drugs) * 100 if gold_drugs else 0
                print("  " + "─" * 85)
                print(f"  Drug Recall: {drug_tp}/{len(gold_drugs)} ({drug_recall:.0f}%)")

            elif drugs:
                print(f"\n  DRUGS ({len(drugs)} found, no gold standard)")
                print("  " + "─" * 85)
                print(f"  {'Name':<40} {'Compound ID':<15} {'Phase':<20} {'Invest?':<8}")
                print("  " + "─" * 85)

                seen_drugs = set()
                for d in drugs:
                    name = d.get("name", "")
                    if name.lower() in seen_drugs:
                        continue
                    seen_drugs.add(name.lower())
                    compound = d.get("compound_id") or "-"
                    phase = d.get("phase") or "-"
                    is_inv = "Yes" if d.get("is_investigational") else "No"
                    print(f"  {name:<40} {compound:<15} {phase:<20} {is_inv:<8}")

        # =====================================================================
        # PIPELINE STATS (optional)
        # =====================================================================
        total_candidates = results_data.get("total_candidates", 0)
        total_validated = results_data.get("total_validated", 0)
        total_rejected = results_data.get("total_rejected", 0)
        heuristics = results_data.get("heuristics_counters", {})

        if total_candidates > 0 or heuristics:
            print("\n")
            print("━" * 80)
            print(" PIPELINE STATISTICS ")
            print("━" * 80)
            print(f"\n  Candidates generated:  {total_candidates}")
            print(f"  Validated:             {total_validated}")
            print(f"  Rejected:              {total_rejected}")

            # Entity counts
            if diseases or drugs:
                print("\n  Entity extraction:")
                print(f"    • Diseases: {len(diseases)}")
                print(f"    • Drugs: {len(drugs)}")

            if heuristics:
                print("\n  Heuristics applied:")
                for key, value in sorted(heuristics.items()):
                    if value > 0:
                        print(f"    • {key}: {value}")

        print("\n" + "═" * 80)
        print(" END OF REPORT ")
        print("═" * 80 + "\n")


# =============================================================================
# PUBLIC API
# =============================================================================
def run_analysis(results_data: Dict[str, Any], gold_path: str) -> Dict[str, Any]:
    """
    Run extraction analysis.

    Args:
        results_data: Export data dict from orchestrator
        gold_path: Path to gold standard JSON

    Returns:
        Metrics dict with precision, recall, f1, etc.
    """
    analyzer = ExtractionAnalyzer()
    return analyzer.analyze(results_data, gold_path)


# =============================================================================
# STANDALONE TEST
# =============================================================================
if __name__ == "__main__":
    # Test with mock data
    mock_results = {
        "document": "01_Article_Iptacopan C3G Trial.pdf",
        "abbreviations": [
            {"short_form": "C3G", "long_form": "C3 glomerulopathy"},
            {"short_form": "eGFR", "long_form": "estimated glomerular filtration rate"},
            {"short_form": "FDA", "long_form": "Food and Drug Administration"},
            {"short_form": "CI", "long_form": "confidence interval"},
            {"short_form": "SD", "long_form": "standard deviation"},
        ],
        "total_candidates": 608,
        "total_validated": 62,
        "total_rejected": 84,
        "heuristics_counters": {
            "recovered_by_stats_whitelist": 8,
            "blacklisted_fp_count": 9,
        },
    }

    # You would pass real gold path here
    # run_analysis(mock_results, "/path/to/papers_gold_v2.json")
    print("Module loaded successfully. Call run_analysis() to use.")

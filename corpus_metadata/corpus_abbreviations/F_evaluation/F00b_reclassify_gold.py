# corpus_metadata/corpus_abbreviations/F_evaluation/F00b_reclassify_gold.py
"""
Reclassify gold annotations based on audit results.

Creates a new gold file with:
- DEFINED: EXTRACTABLE_PAIR (SF+LF with evidence)
- MENTIONED: SF_ONLY + UNSUPPORTED_PAIR (SF found, LF not extractable)
- EXCLUDED: PARSE_ISSUE (for parser coverage metric)

Usage:
    python F00b_reclassify_gold.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def reclassify_gold(
    original_gold_path: str,
    audit_report_path: str,
    output_path: str,
) -> Dict[str, Any]:
    """
    Reclassify gold annotations based on audit buckets.

    Returns summary statistics.
    """
    # Load original gold
    with open(original_gold_path, "r", encoding="utf-8") as f:
        original = json.load(f)

    # Load audit report
    with open(audit_report_path, "r", encoding="utf-8") as f:
        audit = json.load(f)

    # Build lookup: (doc_id, sf, lf) -> bucket
    bucket_lookup: Dict[tuple, str] = {}
    for detail in audit.get("details", []):
        key = (
            detail["doc_id"],
            detail["short_form"].upper(),
            (detail["long_form"] or "").lower(),
        )
        bucket_lookup[key] = detail["bucket"]

    # Reclassify
    defined = []
    mentioned = []
    excluded = []

    for anno in original.get("annotations", []):
        doc_id = anno.get("doc_id", "")
        sf = anno.get("short_form", "").strip()
        lf = anno.get("long_form", "").strip() if anno.get("long_form") else ""

        key = (doc_id, sf.upper(), lf.lower())
        bucket = bucket_lookup.get(key, "UNKNOWN")

        # Add classification to annotation
        anno_copy = anno.copy()
        anno_copy["audit_bucket"] = bucket

        if bucket == "EXTRACTABLE_PAIR":
            anno_copy["gold_type"] = "DEFINED"
            defined.append(anno_copy)
        elif bucket in ["SF_ONLY", "UNSUPPORTED_PAIR"]:
            anno_copy["gold_type"] = "MENTIONED"
            mentioned.append(anno_copy)
        elif bucket == "PARSE_ISSUE":
            anno_copy["gold_type"] = "EXCLUDED"
            excluded.append(anno_copy)
        else:
            # Unknown - treat as MENTIONED
            anno_copy["gold_type"] = "MENTIONED"
            mentioned.append(anno_copy)

    # Build new gold structure
    new_gold = {
        "metadata": {
            "source": original_gold_path,
            "audit_report": audit_report_path,
            "reclassification_version": "1.0",
        },
        "summary": {
            "total_original": len(original.get("annotations", [])),
            "defined": len(defined),
            "mentioned": len(mentioned),
            "excluded": len(excluded),
        },
        "defined_annotations": defined,      # For pair scoring
        "mentioned_annotations": mentioned,  # For SF-only scoring (combined)
        "excluded_annotations": excluded,    # For parser coverage
        # Combined for SF-only scoring
        "all_sf_annotations": defined + mentioned,
    }

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_gold, f, indent=2, ensure_ascii=False)

    print(f"Reclassified gold saved to: {output_path}")
    print()
    print("Summary:")
    print(f"  Original annotations: {new_gold['summary']['total_original']}")
    print(f"  DEFINED (for pair scoring): {new_gold['summary']['defined']}")
    print(f"  MENTIONED (SF-only): {new_gold['summary']['mentioned']}")
    print(f"  EXCLUDED (parse issues): {new_gold['summary']['excluded']}")

    return new_gold["summary"]


if __name__ == "__main__":
    ORIGINAL_GOLD = "/Users/frederictetard/Projects/ese/gold_data/papers_gold.json"
    AUDIT_REPORT = "/Users/frederictetard/Projects/ese/gold_data/audit_report.json"
    OUTPUT = "/Users/frederictetard/Projects/ese/gold_data/papers_gold_v2.json"

    reclassify_gold(ORIGINAL_GOLD, AUDIT_REPORT, OUTPUT)

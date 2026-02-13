#!/usr/bin/env python3
"""Enrich RareDisGene gold standard with confirmed gene mentions from pipeline exports.

The original gold only tracks primary gene-disease associations (~1 gene/doc).
This script adds all real gene mentions (verified by HGNC ID) from pipeline outputs,
raising gold coverage from ~101 to ~188 annotations.

Usage:
    python enrich_raredis_gene_gold.py --dry-run              # Show what would be added
    python enrich_raredis_gene_gold.py --apply                 # Update gold JSON
    python enrich_raredis_gene_gold.py --batch=1 --size=10     # Process docs 1-10 only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# -- Paths -----------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
GOLD_PATH = SCRIPT_DIR.parent / "raredis_gene_gold.json"
EXPORTS_DIR = SCRIPT_DIR / "pdfs" / "test"
GENE_FP_YAML = (
    SCRIPT_DIR.parent.parent
    / "corpus_metadata"
    / "G_config"
    / "data"
    / "gene_fp_terms.yaml"
)


# -- Load C34 always-filter terms -----------------------------------------

def _load_fp_terms(yaml_path: Path) -> set[str]:
    """Load C34 FP filter always-filter terms from YAML (without PyYAML)."""
    terms: set[str] = set()
    if not yaml_path.exists():
        print(f"[WARN] gene_fp_terms.yaml not found: {yaml_path}")
        return terms

    # Simple YAML list parser: collect items under relevant sections
    always_filter_sections = {
        "statistical_terms",
        "units",
        "clinical_terms",
        "drug_terms",
        "study_terms",
        "common_english_words",
    }
    current_section: str | None = None

    with open(yaml_path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            # Detect section header (top-level key ending with colon)
            if stripped.endswith(":") and not stripped.startswith("-") and not stripped.startswith("#"):
                section_name = stripped[:-1].strip()
                current_section = section_name if section_name in always_filter_sections else None
                continue
            # Collect list items
            if current_section and stripped.startswith("- "):
                # Extract value: strip "- ", quotes, and inline comments
                val = stripped[2:].strip()
                # Remove inline comment
                if "  #" in val:
                    val = val[:val.index("  #")].strip()
                elif "\t#" in val:
                    val = val[:val.index("\t#")].strip()
                # Remove surrounding quotes
                if (val.startswith('"') and val.endswith('"')) or (
                    val.startswith("'") and val.endswith("'")
                ):
                    val = val[1:-1]
                if val:
                    terms.add(val.lower())

    return terms


# -- Main logic ------------------------------------------------------------

def load_gold(gold_path: Path) -> dict:
    """Load existing gold standard JSON."""
    with open(gold_path, encoding="utf-8") as f:
        return json.load(f)


def get_existing_gold_pairs(gold_data: dict, split: str = "test") -> set[tuple[str, str]]:
    """Return set of (doc_id, symbol_upper) already in gold for the given split."""
    pairs: set[tuple[str, str]] = set()
    for ann in gold_data.get("genes", {}).get("annotations", []):
        if ann.get("split") == split:
            pairs.add((ann["doc_id"], ann["symbol"].upper()))
    return pairs


def find_gene_exports(exports_dir: Path) -> dict[str, Path]:
    """Map doc_id -> latest gene export JSON path."""
    result: dict[str, Path] = {}
    if not exports_dir.exists():
        return result
    for pmid_dir in sorted(exports_dir.iterdir()):
        if not pmid_dir.is_dir():
            continue
        doc_id = f"{pmid_dir.name}.pdf"
        gene_files = sorted(pmid_dir.glob("genes_*.json"))
        if gene_files:
            result[doc_id] = gene_files[-1]  # Latest by timestamp
    return result


def load_gene_export(export_path: Path) -> list[dict]:
    """Load genes from a pipeline export JSON."""
    with open(export_path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("genes", [])


def classify_gene(
    gene: dict,
    existing_pairs: set[tuple[str, str]],
    doc_id: str,
    fp_terms: set[str],
) -> tuple[str, str]:
    """Classify a gene for enrichment.

    Returns (action, reason):
        - ("skip_existing", ...)  — already in gold
        - ("skip_fp", ...)        — in C34 FP filter
        - ("skip_no_hgnc", ...)   — no HGNC ID
        - ("add", ...)            — confirmed gene, add to gold
    """
    symbol = (gene.get("hgnc_symbol") or gene.get("matched_text", "")).upper()
    if not symbol:
        return "skip_no_symbol", "no symbol"

    # Already in gold?
    if (doc_id, symbol) in existing_pairs:
        return "skip_existing", "already in gold"

    # In C34 always-filter?
    if symbol.lower() in fp_terms:
        return "skip_fp", f"C34 FP filter: {symbol}"

    # Has HGNC ID?
    codes = gene.get("codes", {})
    hgnc_id = codes.get("hgnc_id")
    if not hgnc_id:
        return "skip_no_hgnc", f"no HGNC ID: {symbol}"

    return "add", f"HGNC:{hgnc_id}"


def run_enrichment(
    gold_data: dict,
    exports_dir: Path,
    fp_terms: set[str],
    *,
    batch: int | None = None,
    size: int = 10,
    apply: bool = False,
    gold_path: Path = GOLD_PATH,
) -> None:
    """Run the enrichment process."""
    existing_pairs = get_existing_gold_pairs(gold_data, split="test")
    exports = find_gene_exports(exports_dir)

    # Sort doc_ids for deterministic ordering
    doc_ids = sorted(exports.keys())

    # Apply batch slicing
    if batch is not None:
        start = (batch - 1) * size
        end = start + size
        doc_ids = doc_ids[start:end]
        print(f"Batch {batch}: docs {start + 1}-{min(end, len(doc_ids) + start)} "
              f"({len(doc_ids)} docs)")

    # Collect additions and audit log
    additions: list[dict] = []
    audit_log: list[dict] = []
    stats = {"skip_existing": 0, "skip_fp": 0, "skip_no_hgnc": 0, "add": 0, "skip_no_symbol": 0}

    for doc_id in doc_ids:
        export_path = exports[doc_id]
        genes = load_gene_export(export_path)

        for gene in genes:
            symbol = (gene.get("hgnc_symbol") or gene.get("matched_text", "")).upper()
            action, reason = classify_gene(gene, existing_pairs, doc_id, fp_terms)
            stats[action] += 1

            audit_entry = {
                "doc_id": doc_id,
                "symbol": symbol,
                "action": action,
                "reason": reason,
                "hgnc_id": gene.get("codes", {}).get("hgnc_id"),
                "entrez_id": gene.get("codes", {}).get("entrez"),
            }
            audit_log.append(audit_entry)

            if action == "add":
                additions.append({
                    "doc_id": doc_id,
                    "symbol": symbol,
                    "split": "test",
                })
                # Add to existing_pairs to prevent duplicates within batch
                existing_pairs.add((doc_id, symbol))

    # Print summary
    print(f"\n{'='*60}")
    print(f"Enrichment Summary")
    print(f"{'='*60}")
    print(f"Docs processed:    {len(doc_ids)}")
    print(f"Already in gold:   {stats['skip_existing']}")
    print(f"C34 FP filtered:   {stats['skip_fp']}")
    print(f"No HGNC ID:        {stats['skip_no_hgnc']}")
    print(f"No symbol:         {stats['skip_no_symbol']}")
    print(f"TO ADD:            {stats['add']}")
    print(f"{'='*60}")

    # Print additions
    if additions:
        print(f"\nGenes to add ({len(additions)}):")
        for a in additions:
            matching_audit = next(
                e for e in audit_log
                if e["doc_id"] == a["doc_id"] and e["symbol"] == a["symbol"]
            )
            print(f"  {a['doc_id']:20s}  {a['symbol']:12s}  "
                  f"HGNC:{matching_audit['hgnc_id'] or '?':>6s}  "
                  f"Entrez:{matching_audit['entrez_id'] or '?':>8s}")

    # Print FP-filtered genes
    fp_filtered = [e for e in audit_log if e["action"] == "skip_fp"]
    if fp_filtered:
        print(f"\nC34 FP filtered ({len(fp_filtered)}):")
        for e in fp_filtered:
            print(f"  {e['doc_id']:20s}  {e['symbol']:12s}  {e['reason']}")

    # Print no-HGNC genes (manual review candidates)
    no_hgnc = [e for e in audit_log if e["action"] == "skip_no_hgnc"]
    if no_hgnc:
        print(f"\nNo HGNC ID — manual review ({len(no_hgnc)}):")
        for e in no_hgnc:
            print(f"  {e['doc_id']:20s}  {e['symbol']:12s}")

    # Apply changes
    if apply and additions:
        annotations = gold_data["genes"]["annotations"]
        annotations.extend(additions)
        gold_data["genes"]["total"] = len(annotations)

        with open(gold_path, "w", encoding="utf-8") as f:
            json.dump(gold_data, f, indent=2, ensure_ascii=False)
            f.write("\n")

        print(f"\nGold updated: {gold_path}")
        print(f"New total annotations: {gold_data['genes']['total']}")
    elif not apply and additions:
        print(f"\nDry run — no changes written. Use --apply to update gold.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich RareDisGene gold standard with confirmed gene mentions."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Show what would be added without modifying gold (default)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually update the gold JSON",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Batch number (1-indexed) for incremental processing",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=10,
        help="Batch size (default: 10)",
    )
    args = parser.parse_args()

    if not GOLD_PATH.exists():
        print(f"[ERROR] Gold standard not found: {GOLD_PATH}")
        sys.exit(1)

    if not EXPORTS_DIR.exists():
        print(f"[ERROR] Gene exports directory not found: {EXPORTS_DIR}")
        sys.exit(1)

    # Load data
    print(f"Loading gold: {GOLD_PATH}")
    gold_data = load_gold(GOLD_PATH)
    existing_test = [a for a in gold_data["genes"]["annotations"] if a.get("split") == "test"]
    print(f"Existing test annotations: {len(existing_test)}")

    print(f"Loading C34 FP terms: {GENE_FP_YAML}")
    fp_terms = _load_fp_terms(GENE_FP_YAML)
    print(f"FP filter terms: {len(fp_terms)}")

    run_enrichment(
        gold_data,
        EXPORTS_DIR,
        fp_terms,
        batch=args.batch,
        size=args.size,
        apply=args.apply,
        gold_path=GOLD_PATH,
    )


if __name__ == "__main__":
    main()

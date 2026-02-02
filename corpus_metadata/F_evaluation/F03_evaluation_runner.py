# corpus_metadata/F_evaluation/F03_evaluation_runner.py
#!/usr/bin/env python3
"""
Unified Evaluation Runner for Abbreviation Extraction Pipeline.

PURPOSE:
    End-to-end evaluation of the extraction pipeline against gold standard corpora.
    Targets 100% precision and recall on defined abbreviation pairs.

DATASETS:
    1. NLP4RARE - Rare disease medical documents (dev/test/train splits)
    2. PAPERS - Research papers with human-annotated abbreviations

CONFIGURATION:
    All parameters are in the CONFIGURATION section below.
    By default, runs all tests on all datasets.

USAGE:
    python F03_evaluation_runner.py

OUTPUT:
    - Per-document: TP, FP, FN, Precision, Recall, F1
    - Per-dataset: Aggregate metrics
    - Overall: Combined metrics with pass/fail status (target: 100%)
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add corpus_metadata to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from A_core.A01_domain_models import ValidationStatus
from Z_utils.Z11_console_output import Colors, C

# Check color support
_COLOR_ENABLED = Colors.supports_color()


def _c(color: str, text: str) -> str:
    """Apply color to text if colors are enabled."""
    if _COLOR_ENABLED:
        return f"{color}{text}{C.RESET}"
    return text


# =============================================================================
# CONFIGURATION - Modify these parameters as needed
# =============================================================================

BASE_PATH = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT / "G_config" / "config.yaml"

# NLP4RARE paths
NLP4RARE_PATH = BASE_PATH / "gold_data" / "NLP4RARE"
NLP4RARE_GOLD = BASE_PATH / "gold_data" / "nlp4rare_gold.json"

# Papers paths
PAPERS_PATH = BASE_PATH / "gold_data" / "PAPERS"
PAPERS_GOLD = BASE_PATH / "gold_data" / "papers_gold_v2.json"

# -----------------------------------------------------------------------------
# EVALUATION SETTINGS - Change these to control what gets evaluated
# -----------------------------------------------------------------------------

# Which datasets to run (set to False to skip)
RUN_NLP4RARE = False  # External NLP4RARE rare disease corpus (2000+ PDFs)
RUN_PAPERS = True     # Your papers in gold_data/PAPERS/

# NLP4RARE splits (only used if RUN_NLP4RARE=True)
NLP4RARE_SPLITS = ["dev", "test", "train"]

# Max documents per dataset (None = all documents)
MAX_DOCS = None

# Matching settings
FUZZY_THRESHOLD = 0.8  # Long form matching threshold (0.8 = 80% similarity)
TARGET_ACCURACY = 1.0  # Target: 100%


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class GoldEntry:
    """A single gold standard annotation."""
    doc_id: str
    short_form: str
    long_form: str
    category: Optional[str] = None

    @property
    def sf_normalized(self) -> str:
        return self.short_form.strip().upper()

    @property
    def lf_normalized(self) -> str:
        return " ".join(self.long_form.strip().lower().split())


@dataclass
class ExtractedEntry:
    """A single extracted abbreviation."""
    short_form: str
    long_form: Optional[str]
    confidence: float = 0.0

    @property
    def sf_normalized(self) -> str:
        return self.short_form.strip().upper()

    @property
    def lf_normalized(self) -> Optional[str]:
        if not self.long_form:
            return None
        return " ".join(self.long_form.strip().lower().split())


@dataclass
class DocumentResult:
    """Evaluation results for a single document."""
    doc_id: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    gold_count: int = 0
    extracted_count: int = 0
    processing_time: float = 0.0
    error: Optional[str] = None
    tp_pairs: List[Tuple[str, str]] = field(default_factory=list)
    fp_pairs: List[Tuple[str, str]] = field(default_factory=list)
    fn_pairs: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 1.0

    @property
    def is_perfect(self) -> bool:
        return self.fp == 0 and self.fn == 0


@dataclass
class DatasetResult:
    """Aggregate results for a dataset."""
    name: str
    docs_total: int = 0
    docs_processed: int = 0
    docs_failed: int = 0
    docs_perfect: int = 0
    total_tp: int = 0
    total_fp: int = 0
    total_fn: int = 0
    total_gold: int = 0
    total_extracted: int = 0
    total_time: float = 0.0
    doc_results: List[DocumentResult] = field(default_factory=list)

    @property
    def precision(self) -> float:
        return self.total_tp / (self.total_tp + self.total_fp) if (self.total_tp + self.total_fp) > 0 else 1.0

    @property
    def recall(self) -> float:
        return self.total_tp / (self.total_tp + self.total_fn) if (self.total_tp + self.total_fn) > 0 else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 1.0

    @property
    def is_target_met(self) -> bool:
        return self.precision >= TARGET_ACCURACY and self.recall >= TARGET_ACCURACY


# =============================================================================
# GOLD STANDARD LOADING
# =============================================================================


def load_nlp4rare_gold(gold_path: Path) -> Dict[str, List[GoldEntry]]:
    """Load NLP4RARE gold standard annotations."""
    if not gold_path.exists():
        print(f"  {_c(C.BRIGHT_YELLOW, '[WARN]')} NLP4RARE gold not found: {gold_path}")
        return {}

    with open(gold_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    annotations = data.get("annotations", [])
    by_doc: Dict[str, List[GoldEntry]] = {}

    for ann in annotations:
        entry = GoldEntry(
            doc_id=ann["doc_id"],
            short_form=ann["short_form"],
            long_form=ann["long_form"],
            category=ann.get("category"),
        )
        by_doc.setdefault(entry.doc_id, []).append(entry)

    return by_doc


def load_papers_gold(gold_path: Path) -> Dict[str, List[GoldEntry]]:
    """Load papers gold standard annotations (v2 format with defined_annotations)."""
    if not gold_path.exists():
        print(f"  {_c(C.BRIGHT_YELLOW, '[WARN]')} Papers gold not found: {gold_path}")
        return {}

    with open(gold_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # v2 format uses defined_annotations for extractable pairs
    annotations = data.get("defined_annotations", [])
    by_doc: Dict[str, List[GoldEntry]] = {}

    for ann in annotations:
        entry = GoldEntry(
            doc_id=ann["doc_id"],
            short_form=ann["short_form"],
            long_form=ann["long_form"],
            category=ann.get("category"),
        )
        by_doc.setdefault(entry.doc_id, []).append(entry)

    return by_doc


# =============================================================================
# MATCHING LOGIC
# =============================================================================


def lf_matches(sys_lf: Optional[str], gold_lf: str, threshold: float = FUZZY_THRESHOLD) -> bool:
    """Check if long forms match (exact, substring, or fuzzy)."""
    if sys_lf is None:
        return False

    sys_norm = " ".join(sys_lf.strip().lower().split())
    gold_norm = " ".join(gold_lf.strip().lower().split())

    # Exact match
    if sys_norm == gold_norm:
        return True

    # Substring match (either direction)
    if sys_norm in gold_norm or gold_norm in sys_norm:
        return True

    # Synonym normalization (disease ↔ syndrome)
    synonyms = [("disease", "syndrome"), ("disorder", "syndrome"), ("deficiency", "deficit")]
    sys_syn = sys_norm
    gold_syn = gold_norm
    for term1, term2 in synonyms:
        sys_syn = sys_syn.replace(term2, term1)
        gold_syn = gold_syn.replace(term2, term1)

    if sys_syn == gold_syn or sys_syn in gold_syn or gold_syn in sys_syn:
        return True

    # Fuzzy match
    ratio = SequenceMatcher(None, sys_syn, gold_syn).ratio()
    return ratio >= threshold


def compare_extractions(
    extracted: List[ExtractedEntry],
    gold: List[GoldEntry],
) -> DocumentResult:
    """Compare extracted abbreviations against gold standard."""
    doc_id = gold[0].doc_id if gold else "unknown"
    result = DocumentResult(
        doc_id=doc_id,
        gold_count=len(gold),
        extracted_count=len(extracted),
    )

    # Track which gold entries have been matched
    matched_gold: set[Tuple[str, str]] = set()

    # Check each extraction
    for ext in extracted:
        matched = False
        ext_sf = ext.sf_normalized

        # Find matching gold entries
        for g in gold:
            if ext_sf == g.sf_normalized:
                gold_key = (g.sf_normalized, g.lf_normalized)
                if gold_key not in matched_gold:
                    if lf_matches(ext.lf_normalized, g.lf_normalized):
                        result.tp += 1
                        result.tp_pairs.append((ext.short_form, ext.long_form or ""))
                        matched_gold.add(gold_key)
                        matched = True
                        break

        if not matched:
            result.fp += 1
            result.fp_pairs.append((ext.short_form, ext.long_form or ""))

    # Count missed gold entries
    for g in gold:
        gold_key = (g.sf_normalized, g.lf_normalized)
        if gold_key not in matched_gold:
            result.fn += 1
            result.fn_pairs.append((g.short_form, g.long_form))

    return result


# =============================================================================
# ORCHESTRATOR RUNNER
# =============================================================================


def create_orchestrator():
    """Create and initialize orchestrator."""
    from orchestrator import Orchestrator
    return Orchestrator(config_path=str(CONFIG_PATH))


def run_extraction(orch, pdf_path: Path) -> List[ExtractedEntry]:
    """Run extraction pipeline on a single PDF."""
    results = orch.process_pdf(str(pdf_path))

    extracted = []
    for entity in results:
        if entity.status == ValidationStatus.VALIDATED:
            extracted.append(ExtractedEntry(
                short_form=entity.short_form,
                long_form=entity.long_form,
                confidence=entity.confidence_score,
            ))

    return extracted


# =============================================================================
# EVALUATION RUNNER
# =============================================================================


def evaluate_dataset(
    name: str,
    pdf_folder: Path,
    gold_by_doc: Dict[str, List[GoldEntry]],
    orch,
    max_docs: Optional[int] = None,
    splits: Optional[List[str]] = None,
) -> DatasetResult:
    """Evaluate all PDFs in a dataset against gold standard."""
    print(f"\n{_c(C.BOLD + C.BRIGHT_CYAN, '=' * 70)}")
    print(f" {_c(C.BOLD + C.BRIGHT_WHITE, f'EVALUATING: {name.upper()}')}")
    print(f"{_c(C.BOLD + C.BRIGHT_CYAN, '=' * 70)}")

    result = DatasetResult(name=name)

    # Find PDFs to process
    if splits:
        # NLP4RARE has dev/test/train subdirectories
        pdf_files = []
        for split in splits:
            split_path = pdf_folder / split
            if split_path.exists():
                pdf_files.extend(sorted(split_path.glob("*.pdf")))
    else:
        pdf_files = sorted(pdf_folder.glob("*.pdf"))

    # Filter to PDFs with gold annotations
    pdf_with_gold = [pdf for pdf in pdf_files if pdf.name in gold_by_doc]

    if max_docs:
        pdf_with_gold = pdf_with_gold[:max_docs]

    result.docs_total = len(pdf_with_gold)

    print(f"  PDFs in folder:  {len(pdf_files)}")
    print(f"  PDFs with gold:  {len([p for p in pdf_files if p.name in gold_by_doc])}")
    print(f"  PDFs to process: {len(pdf_with_gold)}")

    for i, pdf_path in enumerate(pdf_with_gold, 1):
        gold_entries = gold_by_doc.get(pdf_path.name, [])

        print(f"\n  [{_c(C.BRIGHT_CYAN, f'{i}/{len(pdf_with_gold)}')}] {pdf_path.name}")
        print(f"      Gold annotations: {len(gold_entries)}")

        start_time = time.time()

        try:
            extracted = run_extraction(orch, pdf_path)
            elapsed = time.time() - start_time

            print(f"      Extracted:        {len(extracted)}")
            print(f"      Time:             {elapsed:.1f}s")

            doc_result = compare_extractions(extracted, gold_entries)
            doc_result.processing_time = elapsed
            result.doc_results.append(doc_result)

            result.docs_processed += 1
            result.total_tp += doc_result.tp
            result.total_fp += doc_result.fp
            result.total_fn += doc_result.fn
            result.total_gold += doc_result.gold_count
            result.total_extracted += doc_result.extracted_count
            result.total_time += elapsed

            if doc_result.is_perfect:
                result.docs_perfect += 1
                status = _c(C.BRIGHT_GREEN, "PERFECT")
            elif doc_result.fn > 0:
                status = _c(C.BRIGHT_YELLOW, f"MISSED {doc_result.fn}")
            else:
                status = _c(C.BRIGHT_RED, f"FP={doc_result.fp}")

            print(f"      Results:          TP={doc_result.tp}, FP={doc_result.fp}, FN={doc_result.fn} [{status}]")

        except Exception as e:
            print(f"      {_c(C.BRIGHT_RED, '[ERROR]')} {e}")
            doc_result = DocumentResult(doc_id=pdf_path.name, error=str(e))
            result.doc_results.append(doc_result)
            result.docs_failed += 1

    return result


# =============================================================================
# REPORTING
# =============================================================================


def print_dataset_summary(result: DatasetResult):
    """Print summary for a dataset."""
    print(f"\n{_c(C.DIM, '-' * 70)}")
    print(f" {_c(C.BOLD, f'{result.name.upper()} SUMMARY')}")
    print(f"{_c(C.DIM, '-' * 70)}")

    print(f"  Documents: {result.docs_processed}/{result.docs_total} processed, {result.docs_perfect} perfect")
    if result.docs_failed > 0:
        print(f"  {_c(C.BRIGHT_RED, f'Failed: {result.docs_failed}')}")
    print(f"  Time:      {result.total_time:.1f}s")
    print()

    # Metrics box
    p_color = C.BRIGHT_GREEN if result.precision >= TARGET_ACCURACY else C.BRIGHT_YELLOW
    r_color = C.BRIGHT_GREEN if result.recall >= TARGET_ACCURACY else C.BRIGHT_YELLOW
    f1_color = C.BRIGHT_GREEN if result.f1 >= TARGET_ACCURACY else C.BRIGHT_YELLOW

    print("  ┌─────────────────────────────────────┐")
    print(f"  │  True Positives (TP):  {_c(C.BRIGHT_GREEN, f'{result.total_tp:>5}')}        │")
    print(f"  │  False Positives (FP): {_c(C.BRIGHT_RED if result.total_fp > 0 else C.BRIGHT_GREEN, f'{result.total_fp:>5}')}        │")
    print(f"  │  False Negatives (FN): {_c(C.BRIGHT_RED if result.total_fn > 0 else C.BRIGHT_GREEN, f'{result.total_fn:>5}')}        │")
    print("  ├─────────────────────────────────────┤")
    print(f"  │  Precision:           {_c(p_color, f'{result.precision:>6.1%}')}       │")
    print(f"  │  Recall:              {_c(r_color, f'{result.recall:>6.1%}')}       │")
    print(f"  │  F1 Score:            {_c(f1_color, f'{result.f1:>6.1%}')}       │")
    print("  └─────────────────────────────────────┘")


def print_error_analysis(result: DatasetResult, max_examples: int = 10):
    """Print detailed error analysis."""
    all_fn = [(doc.doc_id, sf, lf) for doc in result.doc_results for sf, lf in doc.fn_pairs]
    all_fp = [(doc.doc_id, sf, lf) for doc in result.doc_results for sf, lf in doc.fp_pairs]

    if not all_fn and not all_fp:
        print(f"\n  {_c(C.BRIGHT_GREEN, '✓ No errors - all extractions correct!')}")
        return

    print(f"\n{_c(C.BOLD, ' ERROR ANALYSIS')}")

    if all_fn:
        print(f"\n  {_c(C.BRIGHT_YELLOW, f'FALSE NEGATIVES (Missed): {len(all_fn)}')}")
        for doc_id, sf, lf in all_fn[:max_examples]:
            lf_short = (lf[:40] + "...") if len(lf) > 43 else lf
            print(f"    - {_c(C.BRIGHT_WHITE, sf)}: {lf_short} [{doc_id[:30]}]")
        if len(all_fn) > max_examples:
            print(f"    ... and {len(all_fn) - max_examples} more")

    if all_fp:
        print(f"\n  {_c(C.BRIGHT_RED, f'FALSE POSITIVES (Extra): {len(all_fp)}')}")
        for doc_id, sf, lf in all_fp[:max_examples]:
            lf_short = (lf[:40] + "...") if len(lf) > 43 else lf
            print(f"    - {_c(C.BRIGHT_WHITE, sf)}: {lf_short} [{doc_id[:30]}]")
        if len(all_fp) > max_examples:
            print(f"    ... and {len(all_fp) - max_examples} more")


def print_final_summary(results: List[DatasetResult]):
    """Print final summary across all datasets."""
    print(f"\n{_c(C.BOLD + C.BRIGHT_CYAN, '=' * 70)}")
    print(f" {_c(C.BOLD + C.BRIGHT_WHITE, 'FINAL EVALUATION SUMMARY')}")
    print(f"{_c(C.BOLD + C.BRIGHT_CYAN, '=' * 70)}")

    # Aggregate across all datasets
    total_tp = sum(r.total_tp for r in results)
    total_fp = sum(r.total_fp for r in results)
    total_fn = sum(r.total_fn for r in results)
    total_docs = sum(r.docs_processed for r in results)
    total_perfect = sum(r.docs_perfect for r in results)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 1.0

    all_perfect = total_fp == 0 and total_fn == 0

    print(f"\n  Total documents: {total_docs} ({total_perfect} perfect)")
    print()
    print(f"  {_c(C.BOLD, 'OVERALL METRICS:')}")
    print(f"    TP: {total_tp}  |  FP: {total_fp}  |  FN: {total_fn}")
    print()

    if all_perfect:
        print(f"  {_c(C.BOLD + C.BRIGHT_GREEN, '████████████████████████████████████████')}")
        print(f"  {_c(C.BOLD + C.BRIGHT_GREEN, '█                                      █')}")
        print(f"  {_c(C.BOLD + C.BRIGHT_GREEN, '█   ✓ TARGET MET: 100% ACCURACY        █')}")
        print(f"  {_c(C.BOLD + C.BRIGHT_GREEN, '█                                      █')}")
        print(f"  {_c(C.BOLD + C.BRIGHT_GREEN, '████████████████████████████████████████')}")
    else:
        p_str = f"Precision: {precision:.1%}"
        r_str = f"Recall: {recall:.1%}"
        f1_str = f"F1: {f1:.1%}"

        if precision >= TARGET_ACCURACY and recall >= TARGET_ACCURACY:
            status = _c(C.BRIGHT_GREEN, "TARGET MET")
        else:
            status = _c(C.BRIGHT_RED, "TARGET NOT MET")

        print(f"    {p_str}  |  {r_str}  |  {f1_str}")
        print()
        print(f"  Status: {status}")

        # Show what needs to be fixed
        if total_fn > 0:
            print(f"\n  {_c(C.BRIGHT_YELLOW, f'To reach 100%: Fix {total_fn} missed abbreviations')}")
        if total_fp > 0:
            print(f"  {_c(C.BRIGHT_RED, f'To reach 100%: Remove {total_fp} false positives')}")

    print()


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Run evaluation on all configured datasets."""
    print(f"\n{_c(C.BOLD + C.BRIGHT_CYAN, '=' * 70)}")
    print(f" {_c(C.BOLD + C.BRIGHT_WHITE, 'ABBREVIATION EXTRACTION EVALUATION')}")
    print(f" {_c(C.DIM, 'Target: 100% Precision & Recall')}")
    print(f"{_c(C.BOLD + C.BRIGHT_CYAN, '=' * 70)}")

    # Show configuration
    print("\n  Configuration:")
    print(f"    NLP4RARE: {'enabled' if RUN_NLP4RARE else 'disabled'}")
    print(f"    Papers:   {'enabled' if RUN_PAPERS else 'disabled'}")
    print(f"    Max docs: {MAX_DOCS if MAX_DOCS else 'all'}")
    if RUN_NLP4RARE:
        print(f"    Splits:   {', '.join(NLP4RARE_SPLITS)}")

    # Initialize orchestrator once
    print("\n  Initializing orchestrator...")
    orch = create_orchestrator()

    results = []

    # Evaluate NLP4RARE
    if RUN_NLP4RARE and NLP4RARE_PATH.exists():
        gold_by_doc = load_nlp4rare_gold(NLP4RARE_GOLD)
        if gold_by_doc:
            result = evaluate_dataset(
                name="NLP4RARE",
                pdf_folder=NLP4RARE_PATH,
                gold_by_doc=gold_by_doc,
                orch=orch,
                max_docs=MAX_DOCS,
                splits=NLP4RARE_SPLITS,
            )
            print_dataset_summary(result)
            print_error_analysis(result)
            results.append(result)

    # Evaluate Papers
    if RUN_PAPERS and PAPERS_PATH.exists():
        gold_by_doc = load_papers_gold(PAPERS_GOLD)
        if gold_by_doc:
            result = evaluate_dataset(
                name="Papers",
                pdf_folder=PAPERS_PATH,
                gold_by_doc=gold_by_doc,
                orch=orch,
                max_docs=MAX_DOCS,
            )
            print_dataset_summary(result)
            print_error_analysis(result)
            results.append(result)

    # Final summary
    if results:
        print_final_summary(results)

    # Exit with error code if target not met
    all_perfect = all(r.is_target_met for r in results)
    sys.exit(0 if all_perfect else 1)


if __name__ == "__main__":
    main()


__all__ = [
    "GoldEntry",
    "ExtractedEntry",
    "DocumentResult",
    "DatasetResult",
    "evaluate_dataset",
    "load_nlp4rare_gold",
    "load_papers_gold",
]

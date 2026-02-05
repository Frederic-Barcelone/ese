#!/usr/bin/env python3
"""
CADEC Drug Evaluation Script.

Benchmarks the ESE DrugDetector against the CADEC gold standard.
Supports entity-level P/R/F1 (matching F03 pattern) and optional
token-level BIO F1 via seqeval.

Usage:
    cd corpus_metadata
    python ../gold_data/CADEC/evaluate_cadec_drugs.py [--split=test] [--max-docs=N] [--seqeval]

Options:
    --split=SPLIT     Which split to evaluate: train, test, or all [default: test]
    --max-docs=N      Maximum documents to evaluate (default: all)
    --seqeval         Also compute token-level BIO F1 (requires seqeval package)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional

# Add corpus_metadata to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # gold_data/CADEC → gold_data → ese
CORPUS_METADATA = PROJECT_ROOT / "corpus_metadata"
if str(CORPUS_METADATA) not in sys.path:
    sys.path.insert(0, str(CORPUS_METADATA))

from A_core.A01_domain_models import BoundingBox
from B_parsing.B02_doc_graph import ContentRole, DocumentGraph, Page, TextBlock

# Reuse F03 matching logic
FUZZY_THRESHOLD = 0.8

# Brand/generic equivalences for evaluation matching.
# CADEC gold uses consumer brand names; ESE detects generic/INN names.
# Both directions are checked: if extracted=acetaminophen and gold=tylenol, match.
BRAND_GENERIC_EQUIVALENCES: dict[str, set[str]] = {
    "acetaminophen": {"tylenol", "paracetamol", "panadol", "tylenol 3",
                      "tylenol extra", "arthritis strength tylenol"},
    "rofecoxib": {"vioxx"},
    "valdecoxib": {"bextra"},
    "celecoxib": {"celebrex"},
    "etanercept": {"enbrel", "enbrel injections"},
    "furosemide": {"lasix"},
    "naproxen": {"aleve", "alleve", "naprosyn"},
    "simvastatin": {"zocor", "zocar", "zorcor"},
    "atorvastatin": {"lipitor", "liptor", "lipitors"},
    "pravastatin": {"pravachol", "pravochol"},
    "lovastatin": {"mevacor", "mevacore"},
    "rosuvastatin": {"crestor"},
    "fluvastatin": {"lescol"},
    "diclofenac": {"voltaren", "arthrotec", "artrotec", "cataflam", "solaraze",
                   "zipsor", "pennsaid"},
    "ibuprofen": {"advil", "motrin", "nurofen"},
    "captopril": {"capoten", "capiten"},
    "glipizide": {"glucotrol", "glcotrol"},
    "pioglitazone": {"actos", "actose"},
    "nabumetone": {"relafen", "relefen"},
    "ezetimibe": {"zetia", "ezetimbe"},
    "trazodone": {"desyrel", "trazadone"},
    "hydrocodone": {"vicodin", "hydrocodine"},
    "capsaicin": {"capsacian", "zostrix"},
}


# ---------------------------------------------------------------------------
# Data classes (mirroring F03 pattern)
# ---------------------------------------------------------------------------

@dataclass
class GoldDrug:
    """A single gold standard drug entity."""
    doc_id: str
    name: str

    @property
    def name_normalized(self) -> str:
        return " ".join(self.name.strip().lower().split())


@dataclass
class ExtractedDrugEval:
    """A single extracted drug entity for evaluation."""
    name: str
    confidence: float = 0.0

    @property
    def name_normalized(self) -> str:
        return " ".join(self.name.strip().lower().split())


@dataclass
class EntityResult:
    """Evaluation results for a single entity type."""
    entity_type: str
    doc_id: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    gold_count: int = 0
    extracted_count: int = 0
    tp_items: List[str] = field(default_factory=list)
    fp_items: List[str] = field(default_factory=list)
    fn_items: List[str] = field(default_factory=list)

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


# ---------------------------------------------------------------------------
# Matching logic (from F03)
# ---------------------------------------------------------------------------

def _are_brand_generic_equivalent(name_a: str, name_b: str) -> bool:
    """Check if two drug names are brand/generic equivalents."""
    a = name_a.lower()
    b = name_b.lower()

    for generic, brands in BRAND_GENERIC_EQUIVALENCES.items():
        all_names = brands | {generic}
        if a in all_names and b in all_names:
            return True
    return False


def drug_matches(sys_name: str, gold_name: str, threshold: float = FUZZY_THRESHOLD) -> bool:
    """Check if drug names match (exact, substring, fuzzy, or brand/generic)."""
    sys_norm = " ".join(sys_name.strip().lower().split())
    gold_norm = " ".join(gold_name.strip().lower().split())

    if sys_norm == gold_norm:
        return True

    if sys_norm in gold_norm or gold_norm in sys_norm:
        return True

    # Brand/generic equivalence (acetaminophen ↔ Tylenol, etc.)
    if _are_brand_generic_equivalent(sys_norm, gold_norm):
        return True

    ratio = SequenceMatcher(None, sys_norm, gold_norm).ratio()
    return ratio >= threshold


def compare_drugs(
    extracted: List[ExtractedDrugEval],
    gold: List[GoldDrug],
    doc_id: str,
) -> EntityResult:
    """Compare extracted drugs against gold standard."""
    result = EntityResult(
        entity_type="drugs",
        doc_id=doc_id,
        gold_count=len(gold),
        extracted_count=len(extracted),
    )

    matched_gold: set[str] = set()

    for ext in extracted:
        matched = False
        ext_name = ext.name_normalized

        for g in gold:
            gold_key = g.name_normalized
            if gold_key not in matched_gold:
                if drug_matches(ext_name, gold_key):
                    result.tp += 1
                    result.tp_items.append(ext.name)
                    matched_gold.add(gold_key)
                    matched = True
                    break

        if not matched:
            result.fp += 1
            result.fp_items.append(ext.name)

    for g in gold:
        gold_key = g.name_normalized
        if gold_key not in matched_gold:
            result.fn += 1
            result.fn_items.append(g.name)

    return result


# ---------------------------------------------------------------------------
# Document shim: plain text → DocumentGraph
# ---------------------------------------------------------------------------

def text_to_doc_graph(doc_id: str, text: str) -> DocumentGraph:
    """Create a minimal DocumentGraph from plain text for DrugDetector."""
    block = TextBlock(
        text=text,
        page_num=1,
        reading_order_index=0,
        role=ContentRole.BODY_TEXT,
        bbox=BoundingBox(coords=(0, 0, 612, 792)),
    )
    page = Page(number=1, width=612.0, height=792.0, blocks=[block])
    return DocumentGraph(doc_id=doc_id, pages={1: page})


# ---------------------------------------------------------------------------
# Token-level BIO evaluation
# ---------------------------------------------------------------------------

def tokens_to_char_spans(tokens: list[str]) -> list[tuple[int, int]]:
    """Map tokens to character spans in space-joined text."""
    spans = []
    offset = 0
    for tok in tokens:
        spans.append((offset, offset + len(tok)))
        offset += len(tok) + 1  # +1 for space
    return spans


def char_spans_to_bio(
    drug_char_spans: list[tuple[int, int]],
    token_char_spans: list[tuple[int, int]],
) -> list[str]:
    """Convert character-level drug spans to token-level BIO tags."""
    bio = ["O"] * len(token_char_spans)

    for drug_start, drug_end in drug_char_spans:
        in_entity = False
        for i, (tok_start, tok_end) in enumerate(token_char_spans):
            # Token overlaps with drug span
            if tok_start < drug_end and tok_end > drug_start:
                if not in_entity:
                    bio[i] = "B-Drug"
                    in_entity = True
                else:
                    bio[i] = "I-Drug"
            elif in_entity:
                break

    return bio


def run_seqeval(
    all_gold_tags: list[list[str]],
    all_pred_tags: list[list[str]],
) -> Optional[str]:
    """Run seqeval and return classification report string."""
    try:
        from seqeval.metrics import classification_report
        return classification_report(all_gold_tags, all_pred_tags)
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def load_gold(gold_path: Path) -> dict:
    """Load cadec_gold.json."""
    with open(gold_path, encoding="utf-8") as f:
        return json.load(f)


def run_evaluation(
    split: str,
    max_docs: Optional[int],
    do_seqeval: bool,
) -> None:
    gold_path = SCRIPT_DIR / "cadec_gold.json"
    if not gold_path.exists():
        print(f"ERROR: Gold file not found: {gold_path}")
        print("Run generate_cadec_gold.py first.")
        sys.exit(1)

    gold_data = load_gold(gold_path)
    documents = gold_data["documents"]
    annotations = gold_data["drugs"]["annotations"]

    # Filter by split
    if split != "all":
        doc_ids = [
            doc_id for doc_id, doc in documents.items()
            if doc["split"] == split
        ]
    else:
        doc_ids = list(documents.keys())

    if max_docs is not None:
        doc_ids = doc_ids[:max_docs]

    # Build per-document gold sets
    gold_by_doc: dict[str, list[GoldDrug]] = {}
    for ann in annotations:
        if split != "all" and ann["split"] != split:
            continue
        doc_id = ann["doc_id"]
        if doc_id not in gold_by_doc:
            gold_by_doc[doc_id] = []
        gold_by_doc[doc_id].append(GoldDrug(doc_id=doc_id, name=ann["name"]))

    # Initialize DrugDetector
    print("Initializing DrugDetector...")
    t0 = time.time()

    from C_generators.C07_strategy_drug import DrugDetector

    lexicon_path = PROJECT_ROOT / "ouput_datasources"
    if not lexicon_path.exists():
        print(f"ERROR: Lexicon directory not found: {lexicon_path}")
        print("Ensure ouput_datasources/ exists with drug lexicon files.")
        sys.exit(1)

    detector = DrugDetector(config={
        "lexicon_base_path": str(lexicon_path),
    })
    init_time = time.time() - t0
    print(f"DrugDetector initialized in {init_time:.1f}s")

    # Run evaluation
    split_label = split if split != "all" else "all splits"
    print()
    print("=" * 70)
    print(f" CADEC DRUG EVALUATION ({split_label}, {len(doc_ids)} docs)")
    print("=" * 70)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    perfect_docs = 0
    all_gold_bio = []
    all_pred_bio = []

    for idx, doc_id in enumerate(doc_ids, 1):
        doc = documents[doc_id]
        text = doc["text"]
        gold_drugs = gold_by_doc.get(doc_id, [])

        # Run detection
        doc_graph = text_to_doc_graph(doc_id, text)
        candidates = detector.detect(doc_graph)

        # Deduplicate extracted drugs by preferred_name
        seen_names = set()
        extracted = []
        for c in candidates:
            name_lower = c.preferred_name.strip().lower()
            if name_lower not in seen_names:
                seen_names.add(name_lower)
                extracted.append(ExtractedDrugEval(
                    name=c.preferred_name,
                    confidence=getattr(c, "initial_confidence", 0.7),
                ))

        # Compare
        result = compare_drugs(extracted, gold_drugs, doc_id)
        total_tp += result.tp
        total_fp += result.fp
        total_fn += result.fn

        is_perfect = result.fp == 0 and result.fn == 0
        if is_perfect:
            perfect_docs += 1

        # Print per-doc summary
        status = "OK" if is_perfect else "  "
        print(f"  [{idx:>{len(str(len(doc_ids)))}}/{len(doc_ids)}] {status} {doc_id}")
        print(f"    Gold: {result.gold_count}, Extracted: {result.extracted_count}, "
              f"TP={result.tp} FP={result.fp} FN={result.fn}")

        if result.fp_items:
            print(f"    FP: {', '.join(result.fp_items[:5])}"
                  f"{'...' if len(result.fp_items) > 5 else ''}")
        if result.fn_items:
            print(f"    FN: {', '.join(result.fn_items[:5])}"
                  f"{'...' if len(result.fn_items) > 5 else ''}")

        # Token-level BIO
        if do_seqeval and "tokens" in doc and "drug_bio_tags" in doc:
            gold_tokens = doc["tokens"]
            gold_bio = doc["drug_bio_tags"]
            token_spans = tokens_to_char_spans(gold_tokens)

            # Get character-level drug spans from FlashText
            drug_char_spans = []
            for c in candidates:
                start = text.find(c.matched_text)
                if start >= 0:
                    drug_char_spans.append((start, start + len(c.matched_text)))

            pred_bio = char_spans_to_bio(drug_char_spans, token_spans)
            all_gold_bio.append(gold_bio)
            all_pred_bio.append(pred_bio)

    # Summary
    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 1.0

    print()
    print("-" * 70)
    print(f" ENTITY-LEVEL SUMMARY:")
    print(f"   TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print(f"   Precision = {p:.1%}")
    print(f"   Recall    = {r:.1%}")
    print(f"   F1        = {f1:.1%}")
    print(f"   Perfect docs: {perfect_docs}/{len(doc_ids)} ({perfect_docs/len(doc_ids):.1%})")
    print("-" * 70)

    # Seqeval report
    if do_seqeval and all_gold_bio:
        report = run_seqeval(all_gold_bio, all_pred_bio)
        if report:
            print()
            print(" TOKEN-LEVEL (seqeval):")
            print(report)
        else:
            print()
            print(" TOKEN-LEVEL: seqeval not installed (pip install seqeval)")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ESE DrugDetector against CADEC gold standard."
    )
    parser.add_argument(
        "--split", default="test",
        choices=["train", "test", "all"],
        help="Which split to evaluate (default: test)",
    )
    parser.add_argument(
        "--max-docs", type=int, default=None,
        help="Maximum documents to evaluate (default: all)",
    )
    parser.add_argument(
        "--seqeval", action="store_true",
        help="Also compute token-level BIO F1 (requires seqeval)",
    )

    args = parser.parse_args()
    run_evaluation(split=args.split, max_docs=args.max_docs, do_seqeval=args.seqeval)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# corpus_metadata/corpus_abbreviations/F_evaluation/F05_generator_pipeline_test.py
"""
Pipeline Evaluation Test

Runs full extraction pipeline (Orchestrator) on annotated PDFs and scores against gold.
Process: PDF parsing -> Generation -> Claude validation -> Normalization -> Scoring

Output: Per-document and corpus-level precision/recall/F1.

All paths are loaded from config.yaml - no hardcoded parameters.
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

# Ensure imports work
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from A_core.A01_domain_models import ExtractedEntity, ValidationStatus
from A_core.A03_provenance import generate_run_id
from A_core.A04_heuristics_config import HeuristicsConfig, DEFAULT_HEURISTICS_CONFIG
from F_evaluation.F01_gold_loader import GoldLoader, GoldAnnotation
from F_evaluation.F02_scorer import Scorer, ScorerConfig, ScoreReport
from orchestrator import Orchestrator

# =============================================================================
# CONFIGURATION - All loaded from config.yaml
# =============================================================================

DEFAULT_CONFIG_PATH = "/Users/frederictetard/Projects/ese/corpus_metadata/corpus_config/config.yaml"


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        print(f"[WARN] Config file not found: {config_path}, using defaults")
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[WARN] Failed to load config: {e}")
        return {}


def get_paths_from_config(config: Dict[str, Any]) -> Tuple[str, str]:
    """Extract papers folder and gold JSON paths from config."""
    paths = config.get("paths", {})
    base_path = paths.get("base", "/Users/frederictetard/Projects/ese")

    papers_folder = str(Path(base_path) / paths.get("papers_folder", "gold_data/PAPERS"))
    gold_json = str(Path(base_path) / paths.get("gold_json", "gold_data/papers_gold.json"))

    return papers_folder, gold_json


# Load config and extract paths
_CONFIG = load_config()
PAPERS_FOLDER, GOLD_JSON = get_paths_from_config(_CONFIG)


# =============================================================================
# PIPELINE EVALUATOR
# =============================================================================

class PipelineEvaluator:
    """
    Runs Orchestrator on annotated PDFs and scores against gold.
    """

    def __init__(
        self,
        papers_folder: str = PAPERS_FOLDER,
        gold_path: str = GOLD_JSON,
    ):
        self.papers_folder = Path(papers_folder)
        self.gold_path = Path(gold_path)
        self.run_id = generate_run_id("EVAL")

        # Load gold
        self.gold_loader = GoldLoader(strict=False)
        self.gold, self.gold_index = self._load_gold()

        # Scorer
        self.scorer = Scorer(ScorerConfig(
            require_long_form_match=True,
            only_validated=True,
            allow_sf_only_gold=False,
            include_sets_in_report=True,
        ))

        # Orchestrator (full pipeline)
        self.orchestrator = Orchestrator(
            run_id=self.run_id,
            skip_validation=False,
        )

    def _load_gold(self) -> Tuple[any, Dict[str, List[GoldAnnotation]]]:
        """Load gold annotations."""
        if not self.gold_path.exists():
            raise FileNotFoundError(f"Gold file not found: {self.gold_path}")
        return self.gold_loader.load_json(str(self.gold_path))

    def get_annotated_pdfs(self) -> List[Path]:
        """Get PDFs that have gold annotations."""
        annotated_docs = set(self.gold_index.keys())
        pdfs = []
        for pdf in self.papers_folder.glob("*.pdf"):
            if pdf.name in annotated_docs and pdf.name.startswith("01_"):
                pdfs.append(pdf)
        return sorted(pdfs)

    def evaluate_document(self, pdf_path: Path) -> Tuple[List[ExtractedEntity], ScoreReport]:
        """Process and score single document using Orchestrator."""
        doc_id = pdf_path.name
        gold_annos = self.gold_index.get(doc_id, [])

        if not gold_annos:
            print(f"  [WARN] No gold annotations for {doc_id}")
            return [], ScoreReport(
                precision=0.0, recall=0.0, f1=0.0,
                true_positives=0, false_positives=0, false_negatives=0,
            )

        # Run full pipeline via Orchestrator
        entities = self.orchestrator.process_pdf(str(pdf_path))

        # Fix doc_id to match gold format (filename only)
        entities = [e.model_copy(update={"doc_id": doc_id}) for e in entities]

        # Score
        report = self.scorer.evaluate_doc(entities, gold_annos)

        return entities, report

    def evaluate_corpus(self):
        """Evaluate all annotated PDFs."""
        pdfs = self.get_annotated_pdfs()

        if not pdfs:
            print("[X] No annotated PDFs found")
            return

        # Count unique gold SFs across all docs to evaluate
        gold_sfs_in_scope = set()
        for pdf in pdfs:
            doc_id = pdf.name
            for anno in self.gold_index.get(doc_id, []):
                gold_sfs_in_scope.add(anno.short_form.upper())

        # Determine scoring mode from scorer config
        scoring_mode = "sf+lf_match" if self.scorer.config.require_long_form_match else "sf_only_unique"

        # Print evaluation header for quick comparison
        eval_header = self.orchestrator.heuristics.eval_header(
            gold_file=self.gold_path.name,
            gold_count=len(gold_sfs_in_scope),
            scoring_mode=scoring_mode,
        )
        print("\n" + eval_header)

        print("=" * 70)
        print("PIPELINE EVALUATION TEST")
        print("=" * 70)
        print(f"Run ID: {self.run_id}")
        print(f"PDFs to evaluate: {len(pdfs)}")
        print("=" * 70)

        all_entities: List[ExtractedEntity] = []
        all_gold: List[GoldAnnotation] = []

        for pdf_path in pdfs:
            doc_id = pdf_path.name
            print(f"\n{'='*70}")
            print(f"[DOC] {doc_id}")
            print(f"{'='*70}")

            start = time.time()
            entities, report = self.evaluate_document(pdf_path)
            elapsed = time.time() - start

            gold_count = len(self.gold_index.get(doc_id, []))
            validated = [e for e in entities if e.status == ValidationStatus.VALIDATED]

            print(f"\n  SCORE:")
            print(f"  Extracted: {len(validated)} | Gold: {gold_count}")
            print(f"  P: {report.precision:.1%} | R: {report.recall:.1%} | F1: {report.f1:.1%}")
            print(f"  TP: {report.true_positives} | FP: {report.false_positives} | FN: {report.false_negatives}")
            print(f"  Time: {elapsed:.1f}s")

            all_entities.extend(entities)
            all_gold.extend(self.gold_index.get(doc_id, []))

        # Corpus-level scores
        print("\n" + "=" * 70)
        print("CORPUS SUMMARY")
        print("=" * 70)

        corpus_report = self.scorer.evaluate_corpus(all_entities, all_gold)

        print(f"\nMICRO (global):")
        print(f"  Precision: {corpus_report.micro.precision:.1%}")
        print(f"  Recall:    {corpus_report.micro.recall:.1%}")
        print(f"  F1:        {corpus_report.micro.f1:.1%}")
        print(f"  TP: {corpus_report.micro.true_positives} | FP: {corpus_report.micro.false_positives} | FN: {corpus_report.micro.false_negatives}")

        print(f"\nMACRO (per-doc average):")
        print(f"  Precision: {corpus_report.macro.precision:.1%}")
        print(f"  Recall:    {corpus_report.macro.recall:.1%}")
        print(f"  F1:        {corpus_report.macro.f1:.1%}")

        # Error analysis
        if corpus_report.micro.fp_examples:
            print(f"\n[X] False Positives (sample):")
            for ex in corpus_report.micro.fp_examples[:5]:
                print(f"   {ex}")

        if corpus_report.micro.fn_examples:
            print(f"\n[WARN]ï¸  False Negatives (sample):")
            for ex in corpus_report.micro.fn_examples[:5]:
                print(f"   {ex}")

        print("\n" + "=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    evaluator = PipelineEvaluator(
        papers_folder=PAPERS_FOLDER,
        gold_path=GOLD_JSON,
    )
    evaluator.evaluate_corpus()


if __name__ == "__main__":
    main()
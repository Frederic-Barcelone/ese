# corpus_metadata/corpus_abbreviations/orchestrator.py
"""
Orchestrator v0.7: PDF -> Parse -> Generate -> Validate -> Log

Processes all PDFs in /Users/frederictetard/Projects/ese/Pdfs

Usage:
    python orchestrator.py

Requires:
    - ANTHROPIC_API_KEY environment variable (or set in config.yaml)
    - pip install anthropic pyyaml
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure imports work
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from A_core.A01_domain_models import Candidate, ExtractedEntity, ValidationStatus, GeneratorType, EvidenceSpan
from A_core.A03_provenance import generate_run_id, compute_doc_fingerprint
from B_parsing.B01_pdf_to_docgraph import PDFToDocGraphParser
from B_parsing.B03_table_extractor import TableExtractor
from C_generators.C01_strategy_abbrev import AbbrevSyntaxCandidateGenerator
from C_generators.C01b_strategy_glossary import GlossaryTableCandidateGenerator
from C_generators.C02_strategy_regex import RegexCandidateGenerator
from C_generators.C03_strategy_layout import LayoutCandidateGenerator
from C_generators.C04_strategy_flashtext import RegexLexiconGenerator
from D_validation.D02_llm_engine import ClaudeClient, LLMEngine
from D_validation.D03_validation_logger import ValidationLogger


class Orchestrator:
    """
    Main pipeline orchestrator.
    
    Stages:
        1. Parse PDF -> DocumentGraph
        2. Generate candidates (syntax + lexicon)
        3. Validate with Claude
        4. Log results to corpus_log/
    """

    # Default paths
    DEFAULT_CONFIG = "/Users/frederictetard/Projects/ese/corpus_metadata/document_config/config.yaml"
    DEFAULT_LOG_DIR = "/Users/frederictetard/Projects/ese/corpus_log"
    DEFAULT_PDF_DIR = "/Users/frederictetard/Projects/ese/Pdfs"

    def __init__(
        self,
        log_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        config_path: Optional[str] = None,
        run_id: Optional[str] = None,
        skip_validation: bool = False,
    ):
        self.config_path = config_path or self.DEFAULT_CONFIG
        self.log_dir = Path(log_dir or self.DEFAULT_LOG_DIR)
        self.output_dir = Path(output_dir) if output_dir else None
        self.model = model
        self.run_id = run_id or generate_run_id("RUN")
        self.skip_validation = skip_validation

        # Parser
        self.parser = PDFToDocGraphParser()
        self.table_extractor = TableExtractor()

        # Generators (all 5 strategies)
        self.generators = [
            AbbrevSyntaxCandidateGenerator(config={"run_id": self.run_id}),   # C01: Syntax/Schwartz-Hearst
            GlossaryTableCandidateGenerator(config={"run_id": self.run_id}),  # C01b: Glossary tables
            RegexCandidateGenerator(config={                                    # C02: Rigid patterns
                "run_id": self.run_id,
                # Only extract entity types relevant for abbreviation validation
                # Exclude: DOSE, CONCENTRATION, PERCENTAGE, DATE, URL, YEAR (not abbreviations)
                "enabled_types": ["TRIAL_ID", "COMPOUND_ID", "DOI", "PMID", "PMCID"],
            }),
            LayoutCandidateGenerator(config={"run_id": self.run_id}),         # C03: Layout heuristics
            RegexLexiconGenerator(config={"run_id": self.run_id}),            # C04: Lexicon/FlashText
        ]

        # Validation (Claude) - reads config from YAML
        if not skip_validation:
            self.claude_client = ClaudeClient(
                api_key=api_key,
                model=model,
                config_path=self.config_path,
            )
            self.llm_engine = LLMEngine(
                llm_client=self.claude_client,
                model=model,
                run_id=self.run_id,
            )
        else:
            self.claude_client = None
            self.llm_engine = None

        # Logger - writes to corpus_log/
        self.logger = ValidationLogger(
            log_dir=str(self.log_dir),
            run_id=self.run_id,
        )

        print(f"Orchestrator v0.6 initialized")
        print(f"  Run ID: {self.run_id}")
        print(f"  Config: {self.config_path}")
        print(f"  Model: {model}")
        print(f"  Log dir: {self.log_dir}")
        print(f"  Validation: {'ENABLED' if not skip_validation else 'DISABLED'}")

    def process_pdf(
        self,
        pdf_path: str,
        batch_delay_ms: float = 100,
    ) -> List[ExtractedEntity]:
        """
        Process a single PDF through the full pipeline.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        print(f"\n{'='*60}")
        print(f"Processing: {pdf_path.name}")
        print(f"{'='*60}")

        # -----------------------------
        # Stage 1: Parse PDF
        # -----------------------------
        print("\n[1/4] Parsing PDF...")
        start = time.time()

        doc = self.parser.parse(str(pdf_path))
        doc = self.table_extractor.populate_document_graph(doc, str(pdf_path))

        total_blocks = sum(len(p.blocks) for p in doc.pages.values())
        total_tables = sum(len(p.tables) for p in doc.pages.values())

        print(f"  Pages: {len(doc.pages)}")
        print(f"  Blocks: {total_blocks}")
        print(f"  Tables: {total_tables}")
        print(f"  Time: {time.time() - start:.2f}s")

        # -----------------------------
        # Stage 2: Generate candidates
        # -----------------------------
        print("\n[2/4] Generating candidates...")
        start = time.time()

        all_candidates: List[Candidate] = []
        for gen in self.generators:
            candidates = gen.extract(doc)
            print(f"  {gen.generator_type.value}: {len(candidates)} candidates")
            all_candidates.extend(candidates)

        # Deduplicate by (SF, LF)
        seen = set()
        unique_candidates = []
        for c in all_candidates:
            key = (c.short_form.upper(), (c.long_form or "").lower())
            if key not in seen:
                seen.add(key)
                unique_candidates.append(c)

        print(f"  Total unique: {len(unique_candidates)}")
        print(f"  Time: {time.time() - start:.2f}s")

        # -----------------------------
        # Stage 3: Validate with Claude
        # -----------------------------
        if self.skip_validation:
            print("\n[3/4] Validation SKIPPED")
            # Still export candidates without validation
            self._export_results(pdf_path, [], unique_candidates)
            return []

        # Pre-filter: Auto-approve high-confidence lexicon matches
        # These have SF/LF from trusted lexicons and don't need Claude validation
        auto_approved: List[ExtractedEntity] = []
        needs_validation: List[Candidate] = []

        for candidate in unique_candidates:
            # Auto-approve: lexicon matches with high confidence AND both SF and LF present
            if (candidate.generator_type == GeneratorType.LEXICON_MATCH
                and candidate.initial_confidence >= 0.90
                and candidate.long_form
                and len(candidate.long_form) > len(candidate.short_form)):
                # Create evidence span from candidate context
                primary_evidence = EvidenceSpan(
                    text=candidate.context_text,
                    location=candidate.context_location,
                    scope_ref=f"block:{candidate.context_location.block_id or 'unknown'}",
                    start_char_offset=0,
                    end_char_offset=len(candidate.context_text),
                )
                # Create auto-approved entity without Claude call
                auto_approved.append(ExtractedEntity(
                    candidate_id=candidate.id,
                    doc_id=candidate.doc_id,
                    field_type=candidate.field_type,
                    short_form=candidate.short_form,
                    long_form=candidate.long_form,
                    primary_evidence=primary_evidence,
                    supporting_evidence=[],
                    status=ValidationStatus.VALIDATED,
                    confidence_score=candidate.initial_confidence,
                    rejection_reason=None,
                    validation_flags=["AUTO_APPROVED"],
                    provenance=candidate.provenance,
                    raw_llm_response="AUTO_APPROVED:lexicon_match",
                ))
            else:
                needs_validation.append(candidate)

        print(f"\n[3/4] Validating candidates with Claude...")
        print(f"  Auto-approved (lexicon): {len(auto_approved)}")
        print(f"  Needs Claude validation: {len(needs_validation)}")
        start = time.time()

        results: List[ExtractedEntity] = list(auto_approved)
        for i, candidate in enumerate(needs_validation, 1):
            if i % 10 == 0 or i == len(needs_validation):
                print(f"  Progress: {i}/{len(needs_validation)}")

            try:
                val_start = time.time()
                entity = self.llm_engine.verify_candidate(candidate)
                elapsed_ms = (time.time() - val_start) * 1000

                # Log result to corpus_log/
                self.logger.log_validation(
                    candidate=candidate,
                    entity=entity,
                    llm_response=entity.raw_llm_response,
                    elapsed_ms=elapsed_ms,
                )
                results.append(entity)

            except Exception as e:
                self.logger.log_error(candidate, str(e))
                print(f"  ⚠ Error: {candidate.short_form} - {e}")

            # Rate limit
            if batch_delay_ms > 0 and i < len(needs_validation):
                time.sleep(batch_delay_ms / 1000)

        print(f"  Time: {time.time() - start:.2f}s")

        # -----------------------------
        # Stage 4: Summary & Export
        # -----------------------------
        print("\n[4/4] Writing summary...")
        
        # Write validation log summary
        self.logger.write_summary()
        self.logger.print_summary()

        # Export results
        self._export_results(pdf_path, results, unique_candidates)

        # Print validated abbreviations
        validated = [r for r in results if r.status == ValidationStatus.VALIDATED]
        if validated:
            print(f"\nValidated abbreviations ({len(validated)}):")
            for v in validated[:20]:
                lf = v.long_form or "(no expansion)"
                print(f"  • {v.short_form} → {lf}")
            if len(validated) > 20:
                print(f"  ... and {len(validated) - 20} more")

        return results

    def _export_results(
        self,
        pdf_path: Path,
        results: List[ExtractedEntity],
        candidates: List[Candidate],
    ) -> None:
        """
        Export results to JSON in the PDF directory.
        """
        # Determine output directory
        out_dir = self.output_dir or pdf_path.parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Build export data
        export_data = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "document": pdf_path.name,
            "document_path": str(pdf_path),
            "total_candidates": len(candidates),
            "total_validated": len([r for r in results if r.status == ValidationStatus.VALIDATED]),
            "total_rejected": len([r for r in results if r.status == ValidationStatus.REJECTED]),
            "total_ambiguous": len([r for r in results if r.status == ValidationStatus.AMBIGUOUS]),
            "abbreviations": [],
        }

        # Add validated abbreviations
        for entity in results:
            if entity.status == ValidationStatus.VALIDATED:
                # Extract lexicon_ids as list of dicts
                lexicon_ids = None
                if entity.provenance.lexicon_ids:
                    lexicon_ids = [
                        {"source": lid.source, "id": lid.id}
                        for lid in entity.provenance.lexicon_ids
                    ]

                export_data["abbreviations"].append({
                    "short_form": entity.short_form,
                    "long_form": entity.long_form,
                    "confidence": entity.confidence_score,
                    "field_type": entity.field_type.value,
                    "page": entity.primary_evidence.location.page_num if entity.primary_evidence else None,
                    "context_text": entity.primary_evidence.text if entity.primary_evidence else None,
                    "lexicon_source": entity.provenance.lexicon_source,
                    "lexicon_ids": lexicon_ids,
                })

        # Write JSON
        out_file = out_dir / f"abbreviations_{pdf_path.stem}_{timestamp}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n  Exported: {out_file}")

    def process_folder(
        self,
        folder_path: Optional[str] = None,
        batch_delay_ms: float = 100,
    ) -> Dict[str, List[ExtractedEntity]]:
        """
        Process all PDFs in a folder.
        
        Returns dict of {pdf_name: [entities]}
        """
        folder = Path(folder_path or self.DEFAULT_PDF_DIR)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")

        pdf_files = sorted(folder.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in {folder}")
            return {}

        print(f"\n{'#'*60}")
        print(f"BATCH PROCESSING: {len(pdf_files)} PDFs")
        print(f"Folder: {folder}")
        print(f"{'#'*60}")

        all_results: Dict[str, List[ExtractedEntity]] = {}
        
        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"\n[{i}/{len(pdf_files)}] {pdf_path.name}")
            try:
                results = self.process_pdf(str(pdf_path), batch_delay_ms=batch_delay_ms)
                all_results[pdf_path.name] = results
            except Exception as e:
                print(f"  ⚠ ERROR processing {pdf_path.name}: {e}")
                all_results[pdf_path.name] = []

        # Final summary
        print(f"\n{'#'*60}")
        print(f"BATCH COMPLETE")
        print(f"{'#'*60}")
        print(f"PDFs processed: {len(all_results)}")
        
        total_validated = sum(
            len([r for r in results if r.status == ValidationStatus.VALIDATED])
            for results in all_results.values()
        )
        print(f"Total validated abbreviations: {total_validated}")
        print(f"Log directory: {self.log_dir}")

        return all_results


def main():
    """Process all PDFs in the default folder."""
    orchestrator = Orchestrator()
    orchestrator.process_folder()


if __name__ == "__main__":
    main()
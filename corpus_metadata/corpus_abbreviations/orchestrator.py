# corpus_metadata/corpus_abbreviations/orchestrator.py
"""
Orchestrator v0.7: PDF -> Parse -> Generate -> Validate -> Log

Processes all PDFs in /Users/frederictetard/Projects/ese/Pdfs

Usage:
    python orchestrator.py

Requires:
    - ANTHROPIC_API_KEY in .env file (or set in config.yaml)
    - pip install anthropic pyyaml python-dotenv
"""

from __future__ import annotations

from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

import json
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Ensure imports work
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from A_core.A01_domain_models import Candidate, ExtractedEntity, ValidationStatus, GeneratorType, EvidenceSpan
from A_core.A03_provenance import generate_run_id, compute_doc_fingerprint, hash_string
from B_parsing.B01_pdf_to_docgraph import PDFToDocGraphParser
from B_parsing.B03_table_extractor import TableExtractor
from C_generators.C01_strategy_abbrev import AbbrevSyntaxCandidateGenerator
from C_generators.C01b_strategy_glossary import GlossaryTableCandidateGenerator
from C_generators.C02_strategy_regex import RegexCandidateGenerator
from C_generators.C03_strategy_layout import LayoutCandidateGenerator
from C_generators.C04_strategy_flashtext import RegexLexiconGenerator
from D_validation.D02_llm_engine import ClaudeClient, LLMEngine
from D_validation.D03_validation_logger import ValidationLogger
from E_normalization.E01_term_mapper import TermMapper
from E_normalization.E02_disambiguator import Disambiguator


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

        # Normalization (E01) - canonicalize long forms + attach standard IDs
        self.term_mapper = TermMapper(config={
            "mapping_file_path": "/Users/frederictetard/Projects/ese/ouput_datasources/2025_08_abbreviation_general.json",
            "enable_fuzzy_matching": False,
            "fill_long_form_for_orphans": False,
        })

        # Disambiguation (E02) - resolve ambiguous abbreviations using context
        self.disambiguator = Disambiguator(config={
            "min_context_score": 2,
            "min_margin": 1,
            "fill_long_form_for_orphans": True,
        })

        print(f"Orchestrator v0.7 initialized")
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

        # Collect SFs found by non-lexicon generators (syntax, glossary, layout, regex)
        # These have evidence of being actual abbreviations in the document
        corroborated_sfs: set = {
            c.short_form.upper()
            for c in unique_candidates
            if c.generator_type != GeneratorType.LEXICON_MATCH
        }

        # Count SF occurrences in document text (for frequency-based filtering)
        full_text = " ".join(
            block.text for block in doc.iter_linear_blocks() if block.text
        )
        from collections import Counter
        import re
        # Find all abbreviation-like tokens (including mixed case like IgA, C3a, eGFR)
        # Pattern: starts with letter, contains uppercase, 2-12 chars
        abbrev_pattern = r'\b[A-Za-z][A-Za-z0-9]{1,11}\b'
        all_tokens = re.findall(abbrev_pattern, full_text)
        # Normalize to uppercase for counting
        word_counts = Counter(t.upper() for t in all_tokens if any(c.isupper() for c in t))

        # Filter: keep lexicon matches if:
        # 1. SF is corroborated by another generator, OR
        # 2. SF appears 1+ times in document (relaxed from 2+ to improve recall)
        def should_keep(c: Candidate) -> bool:
            if c.generator_type != GeneratorType.LEXICON_MATCH:
                return True
            sf_upper = c.short_form.upper()
            if sf_upper in corroborated_sfs:
                return True
            # Relaxed: keep if SF appears at least once (was 2+)
            if word_counts.get(sf_upper, 0) >= 1:
                return True
            return False

        needs_validation: List[Candidate] = [c for c in unique_candidates if should_keep(c)]
        frequent_sfs = sum(1 for c in unique_candidates
                          if c.generator_type == GeneratorType.LEXICON_MATCH
                          and c.short_form.upper() not in corroborated_sfs
                          and word_counts.get(c.short_form.upper(), 0) >= 2)

        filtered_count = len(unique_candidates) - len(needs_validation)

        # ========================================
        # TIERED VALIDATION: Auto-approve/reject high-confidence candidates
        # ========================================
        auto_results: List[Tuple[Candidate, ExtractedEntity]] = []
        llm_candidates: List[Candidate] = []

        # Common English words that look like abbreviations but aren't
        COMMON_WORDS = {
            'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL',
            'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET',
            'HAS', 'HIM', 'HIS', 'HOW', 'ITS', 'MAY', 'NEW', 'NOW',
            'OLD', 'SEE', 'TWO', 'WAY', 'WHO', 'BOY', 'DID', 'ANY',
            'DATA', 'WHITE', 'METHODS', 'STUDY', 'RESULTS', 'AGE',
            'YEARS', 'PATIENTS', 'TABLE', 'FIGURE', 'BASELINE',
        }

        auto_approved_count = 0
        auto_rejected_count = 0

        def _create_auto_entity(c: Candidate, status: ValidationStatus,
                                 confidence: float, reason: str, flags: List[str],
                                 raw_response: Dict) -> ExtractedEntity:
            """Helper to create ExtractedEntity for auto-approved/rejected candidates."""
            context = (c.context_text or "").strip()
            ctx_hash = hash_string(context) if context else "no_context"
            primary = EvidenceSpan(
                text=context,
                location=c.context_location,
                scope_ref=ctx_hash,
                start_char_offset=0,
                end_char_offset=len(context),
            )
            return ExtractedEntity(
                candidate_id=c.id,
                doc_id=c.doc_id,
                field_type=c.field_type,
                short_form=c.short_form.strip(),
                long_form=(c.long_form.strip() if c.long_form else None),
                primary_evidence=primary,
                supporting_evidence=[],
                status=status,
                confidence_score=confidence,
                rejection_reason=reason if status == ValidationStatus.REJECTED else None,
                validation_flags=flags,
                provenance=c.provenance,
                raw_llm_response=raw_response,
            )

        for c in needs_validation:
            sf_upper = c.short_form.upper()

            # Auto-reject: Common English words
            if sf_upper in COMMON_WORDS:
                entity = _create_auto_entity(
                    c, ValidationStatus.REJECTED, 0.95,
                    "Common English word, not an abbreviation",
                    ["auto_rejected"], {"auto": "rejected_common_word"}
                )
                auto_results.append((c, entity))
                auto_rejected_count += 1
                continue

            # Auto-approve: DISABLED - lexicon LFs often don't match document context
            # Example: UMLS maps "eGFR" to "epidermal growth factor receptor"
            # but in clinical docs it's "estimated glomerular filtration rate"
            # if (sf_upper in corroborated_sfs
            #     and c.provenance
            #     and c.provenance.lexicon_source):
            #     entity = _create_auto_entity(...)
            #     auto_approved_count += 1

            # Needs LLM validation
            llm_candidates.append(c)

        print(f"\n[3/4] Validating candidates with Claude...")
        print(f"  Corroborated SFs: {len(corroborated_sfs)}")
        print(f"  Frequent SFs (2+ occurrences): {frequent_sfs}")
        print(f"  Filtered (lexicon-only, rare): {filtered_count}")
        print(f"  Auto-approved (corroborated+lexicon): {auto_approved_count}")
        print(f"  Auto-rejected (common words): {auto_rejected_count}")
        print(f"  Candidates for LLM: {len(llm_candidates)}")
        start = time.time()

        # Collect all results
        results: List[ExtractedEntity] = []

        # Log and add auto-approved/rejected
        for candidate, entity in auto_results:
            self.logger.log_validation(
                candidate=candidate,
                entity=entity,
                llm_response=entity.raw_llm_response,
                elapsed_ms=0,
            )
            results.append(entity)

        # Use individual validation (batch had quality issues - F1 44% vs 49%)
        batch_size = 15
        for batch_start in range(0, len(llm_candidates), batch_size):
            batch_end = min(batch_start + batch_size, len(llm_candidates))
            batch = llm_candidates[batch_start:batch_end]

            print(f"  Progress: {batch_end}/{len(llm_candidates)}")

            for candidate in batch:
                try:
                    val_start = time.time()
                    entity = self.llm_engine.verify_candidate(candidate)
                    elapsed_ms = (time.time() - val_start) * 1000
                    self.logger.log_validation(
                        candidate=candidate,
                        entity=entity,
                        llm_response=entity.raw_llm_response,
                        elapsed_ms=elapsed_ms,
                    )
                    results.append(entity)
                except Exception as e:
                    self.logger.log_error(candidate, str(e))

            # Rate limit between batches
            if batch_delay_ms > 0 and batch_end < len(llm_candidates):
                time.sleep(batch_delay_ms / 1000)

        print(f"  Time: {time.time() - start:.2f}s")

        # -----------------------------
        # Stage 3.5: Normalize & Disambiguate
        # -----------------------------
        print("\n[3.5/4] Normalizing & disambiguating...")

        # Get full document text for disambiguation context
        full_doc_text = " ".join(
            block.text for block in doc.iter_linear_blocks() if block.text
        )

        # Apply normalization (E01) - canonicalize long forms
        normalized_count = 0
        for i, entity in enumerate(results):
            normalized = self.term_mapper.normalize(entity)
            if "normalized" in (normalized.validation_flags or []):
                normalized_count += 1
            results[i] = normalized

        # Apply disambiguation (E02) - resolve ambiguous SFs using context
        results = self.disambiguator.resolve(results, full_doc_text)
        disambiguated_count = sum(
            1 for r in results if "disambiguated" in (r.validation_flags or [])
        )

        print(f"  Normalized: {normalized_count}")
        print(f"  Disambiguated: {disambiguated_count}")

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
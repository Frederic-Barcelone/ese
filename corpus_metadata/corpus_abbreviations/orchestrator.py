# corpus_metadata/corpus_abbreviations/orchestrator.py
"""
Metadata Extraction Orchestrator (v0.5)

Pipeline: Parse → Generate → Validate (Claude Opus) → Export

Generators:
  - C01 syntax:    Schwartz-Hearst abbreviation extraction (LF/SF pairs)
  - C01b glossary: Glossary table extraction
  - C02 regex:     Rigid pattern matching (Trial IDs, DOIs, doses, dates)
  - C03 layout:    Spatial extraction (zones, label-value, columns)
  - C04 flashtext: Lexicon matching (abbreviations + disease terms)

Usage:
    python orchestrator.py

Configure paths in __main__ section at bottom of file.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Core imports
from A_core.A01_domain_models import Candidate, ExtractedEntity, ValidationStatus
from A_core.A03_provenance import generate_run_id, get_git_revision_hash, compute_doc_fingerprint

# Parsing imports
from B_parsing.B01_pdf_to_docgraph import PDFToDocGraphParser
from B_parsing.B02_doc_graph import DocumentGraph
from B_parsing.B03_table_extractor import TableExtractor

# Generator imports
from C_generators.C01_strategy_abbrev import AbbrevSyntaxCandidateGenerator, GlossaryTableCandidateGenerator
from C_generators.C02_strategy_regex import RegexCandidateGenerator
from C_generators.C03_strategy_layout import LayoutCandidateGenerator
from C_generators.C04_strategy_flashtext import RegexLexiconGenerator

# Validation imports
from D_validation.D02_llm_engine import LLMEngine, ClaudeClient


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class ClaudeModelConfig:
    """Claude model configuration."""
    model: str
    max_tokens: int
    temperature: float


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    # Paths
    pdfs_dir: Path
    output_dir: Path

    # Parser
    parser_strategy: str = "hi_res"  # hi_res for best accuracy, fast for speed

    # Generators (all enabled by default)
    enabled_generators: List[str] = field(default_factory=lambda: ["syntax", "glossary", "regex", "layout", "flashtext"])

    # Validation
    validation_enabled: bool = False
    claude_fast: Optional[ClaudeModelConfig] = None
    claude_validation: Optional[ClaudeModelConfig] = None

    # Output
    output_format: str = "json"  # "json" for readable, "jsonl" for streaming
    include_rejected: bool = False

    # Provenance
    run_id: str = field(default_factory=lambda: generate_run_id("ABBR"))
    pipeline_version: str = field(default_factory=get_git_revision_hash)


def load_config(config_path: str, pdfs_dir: str) -> PipelineConfig:
    """Load config from YAML."""
    cfg = {}
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    # Base path from env or default
    base_path = Path(os.environ.get("CORPUS_BASE_PATH", "/Users/frederictetard/Projects/ese"))

    # Paths
    paths_cfg = cfg.get("paths", {})
    
    # Output dir
    logs_path = paths_cfg.get("logs", "corpus_logs")
    if not Path(logs_path).is_absolute():
        output_dir = base_path / logs_path
    else:
        output_dir = Path(logs_path)

    # Claude API config
    api_cfg = cfg.get("api", {}).get("claude", {})
    
    claude_fast = None
    claude_validation = None
    
    if api_cfg.get("fast"):
        fast = api_cfg["fast"]
        claude_fast = ClaudeModelConfig(
            model=fast.get("model", "claude-sonnet-4-5-20250929"),
            max_tokens=fast.get("max_tokens", 1500),
            temperature=fast.get("temperature", 0),
        )
    
    if api_cfg.get("validation"):
        val = api_cfg["validation"]
        claude_validation = ClaudeModelConfig(
            model=val.get("model", "claude-opus-4-5-20251101"),
            max_tokens=val.get("max_tokens", 4096),
            temperature=val.get("temperature", 0),
        )

    # Features
    features = cfg.get("features", {})
    validation_enabled = features.get("ai_validation", False)

    return PipelineConfig(
        pdfs_dir=Path(pdfs_dir),
        output_dir=output_dir,
        parser_strategy=cfg.get("parser", {}).get("strategy", "hi_res"),
        enabled_generators=["syntax", "glossary", "regex", "layout", "flashtext"],
        validation_enabled=validation_enabled,
        claude_fast=claude_fast,
        claude_validation=claude_validation,
        output_format=cfg.get("output", {}).get("format", "json"),
        include_rejected=False,
    )


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------

class Orchestrator:
    """
    Abbreviation extraction pipeline.
    
    Flow:
        1. Parse PDF → DocumentGraph
        2. Run generators → List[Candidate]
        3. Deduplicate
        4. Validate with Claude (Opus) if enabled
        5. Export results
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Parser (uses hi_res by default for best accuracy)
        self.parser = PDFToDocGraphParser({
            "unstructured_strategy": config.parser_strategy,
        })
        self.table_extractor = TableExtractor()

        # Generators
        self.generators = self._init_generators()

        # Validator (lazy init)
        self._validator: Optional[LLMEngine] = None

    def _init_generators(self) -> Dict[str, Any]:
        """Initialize generators."""
        base_cfg = {
            "pipeline_version": self.config.pipeline_version,
            "run_id": self.config.run_id,
        }
        return {
            "syntax": AbbrevSyntaxCandidateGenerator(base_cfg),
            "glossary": GlossaryTableCandidateGenerator(base_cfg),
            "regex": RegexCandidateGenerator(base_cfg),
            "layout": LayoutCandidateGenerator(base_cfg),
            "flashtext": RegexLexiconGenerator(base_cfg),
        }

    def _get_validator(self) -> Optional[LLMEngine]:
        """Lazy init validator with Claude Opus."""
        if self._validator is not None:
            return self._validator

        if not self.config.validation_enabled:
            return None

        if not self.config.claude_validation:
            print("  ⚠ Validation enabled but no claude.validation config")
            return None

        try:
            client = ClaudeClient()
            model_cfg = self.config.claude_validation
            
            self._validator = LLMEngine(
                llm_client=client,
                model=model_cfg.model,
                temperature=model_cfg.temperature,
                max_tokens=model_cfg.max_tokens,
                run_id=self.config.run_id,
                pipeline_version=self.config.pipeline_version,
            )
            print(f"  ✓ Validator initialized: {model_cfg.model}")
            return self._validator
        except Exception as e:
            print(f"  ✗ Failed to init validator: {e}")
            return None

    def _collect_pdfs(self) -> List[Path]:
        """Collect PDF files to process."""
        path = self.config.pdfs_dir
        
        if path.is_file() and path.suffix.lower() == ".pdf":
            return [path]
        elif path.is_dir():
            pdfs = sorted(path.glob("*.pdf"))
            return pdfs
        else:
            raise ValueError(f"Invalid input path: {path}")

    def process_pdf(self, pdf_path: Path, index: int, total: int) -> Tuple[List[Candidate], List[ExtractedEntity]]:
        """Process a single PDF."""
        print(f"\n[{index}/{total}] {pdf_path.name}")

        # 1. Parse
        print(f"  → Parsing...")
        doc = self.parser.parse(str(pdf_path))
        doc = self.table_extractor.populate_document_graph(doc, str(pdf_path))
        print(f"    Pages: {len(doc.pages)}")

        # Doc fingerprint
        with open(pdf_path, "rb") as f:
            doc_fp = compute_doc_fingerprint(f.read())

        # 2. Generate candidates
        print(f"  → Generating candidates...")
        all_candidates: List[Candidate] = []
        
        for name in self.config.enabled_generators:
            gen = self.generators.get(name)
            if not gen:
                continue
            if hasattr(gen, "config"):
                gen.config["doc_fingerprint"] = doc_fp
            
            candidates = gen.extract(doc)
            if candidates:
                print(f"    {name}: {len(candidates)}")
            all_candidates.extend(candidates)

        # 3. Deduplicate
        deduped = self._deduplicate(all_candidates)
        print(f"  → Deduplicated: {len(all_candidates)} → {len(deduped)}")

        # 4. Validate (if enabled)
        entities: List[ExtractedEntity] = []
        validator = self._get_validator()
        
        if validator and deduped:
            print(f"  → Validating with Claude Opus...")
            for i, cand in enumerate(deduped):
                try:
                    entity = validator.verify_candidate(cand)
                    entities.append(entity)
                except Exception as e:
                    print(f"    ✗ {cand.short_form}: {e}")

            # Stats
            validated = sum(1 for e in entities if e.status == ValidationStatus.VALIDATED)
            rejected = sum(1 for e in entities if e.status == ValidationStatus.REJECTED)
            ambiguous = sum(1 for e in entities if e.status == ValidationStatus.AMBIGUOUS)
            print(f"    ✓ {validated} validated | ✗ {rejected} rejected | ? {ambiguous} ambiguous")

        return deduped, entities

    def _deduplicate(self, candidates: List[Candidate]) -> List[Candidate]:
        """Dedupe by SF+LF, keep highest confidence."""
        seen: Dict[Tuple[str, Optional[str]], Candidate] = {}
        for c in candidates:
            key = (c.short_form.upper(), (c.long_form or "").lower() if c.long_form else None)
            if key not in seen or c.initial_confidence > seen[key].initial_confidence:
                seen[key] = c
        return list(seen.values())

    def _get_context_with_sf(self, context: str, short_form: str, max_len: int = 200) -> str:
        """Get context centered around short_form to ensure it's visible."""
        if not context:
            return ""

        import re

        # Try exact match first, then word boundary match
        patterns = [
            re.compile(r'\(' + re.escape(short_form) + r'\)', re.IGNORECASE),  # (SF)
            re.compile(r'\b' + re.escape(short_form) + r'\b', re.IGNORECASE),   # word boundary
            re.compile(re.escape(short_form), re.IGNORECASE),                    # anywhere
        ]

        match = None
        for pattern in patterns:
            match = pattern.search(context)
            if match:
                break

        if match:
            # Center the window around the short_form
            sf_start = match.start()
            sf_end = match.end()

            # Calculate window to center around the match
            half_window = (max_len - (sf_end - sf_start)) // 2
            start = max(0, sf_start - half_window)
            end = min(len(context), sf_end + half_window)

            # Adjust if we hit boundaries
            if start == 0:
                end = min(len(context), max_len)
            elif end == len(context):
                start = max(0, len(context) - max_len)

            result = context[start:end]
            # Add ellipsis if truncated
            if start > 0:
                result = "..." + result
            if end < len(context):
                result = result + "..."
            return result

        # Fallback: SF not in context, return with note
        if len(context) > max_len - 20:
            return context[:max_len - 20] + f"... [{short_form}]"
        return context + f" [{short_form}]"

    def _count_occurrences(self, short_form: str, doc_name: str, all_candidates: Dict[str, List[Candidate]]) -> int:
        """Count occurrences of short_form across all candidates for a document."""
        import re
        count = 0
        cands = all_candidates.get(doc_name, [])
        for c in cands:
            # Count in context_text
            if c.context_text:
                pattern = re.compile(r'\b' + re.escape(short_form) + r'\b', re.IGNORECASE)
                count += len(pattern.findall(c.context_text))
        # At minimum return 1 (the definition occurrence)
        return max(1, count)

    def run(self) -> Dict[str, Any]:
        """Run pipeline on all PDFs."""
        pdfs = self._collect_pdfs()
        
        if not pdfs:
            raise ValueError(f"No PDFs found in {self.config.pdfs_dir}")

        # Header
        print(f"\n{'='*60}")
        print(f"ABBREVIATION EXTRACTION PIPELINE")
        print(f"{'='*60}")
        print(f"Run ID:     {self.config.run_id}")
        print(f"Input:      {self.config.pdfs_dir}")
        print(f"PDFs:       {len(pdfs)}")
        print(f"Generators: {', '.join(self.config.enabled_generators)}")
        print(f"Validation: {'Claude Opus' if self.config.validation_enabled else 'DISABLED'}")
        print(f"Output:     {self.config.output_dir}")
        print(f"{'='*60}")

        # Process each PDF
        all_candidates: Dict[str, List[Candidate]] = {}
        all_entities: Dict[str, List[ExtractedEntity]] = {}
        errors: List[str] = []

        for i, pdf_path in enumerate(pdfs, 1):
            try:
                candidates, entities = self.process_pdf(pdf_path, i, len(pdfs))
                all_candidates[pdf_path.name] = candidates
                all_entities[pdf_path.name] = entities
            except Exception as e:
                print(f"\n[{i}/{len(pdfs)}] ✗ {pdf_path.name}: {e}")
                errors.append(f"{pdf_path.name}: {e}")

        # Export
        output_path = self._export(all_candidates, all_entities)

        # Summary
        total_cand = sum(len(c) for c in all_candidates.values())
        total_valid = sum(
            1 for ents in all_entities.values() 
            for e in ents if e.status == ValidationStatus.VALIDATED
        )

        print(f"\n{'='*60}")
        print(f"COMPLETED")
        print(f"{'='*60}")
        print(f"PDFs processed: {len(pdfs) - len(errors)}/{len(pdfs)}")
        print(f"Candidates:     {total_cand}")
        if self.config.validation_enabled:
            print(f"Validated:      {total_valid}")
        if errors:
            print(f"Errors:         {len(errors)}")
        print(f"Output:         {output_path}")
        print(f"{'='*60}\n")

        return {
            "run_id": self.config.run_id,
            "pdfs_processed": len(pdfs) - len(errors),
            "pdfs_total": len(pdfs),
            "total_candidates": total_cand,
            "total_validated": total_valid if self.config.validation_enabled else None,
            "errors": errors,
            "output_path": str(output_path),
        }

    def _export(
        self,
        candidates: Dict[str, List[Candidate]],
        entities: Dict[str, List[ExtractedEntity]],
    ) -> Path:
        """Export results to JSON/JSONL with improved formatting."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"abbreviations_{timestamp}.{self.config.output_format}"
        # Output to PDFs directory (same location as input files)
        output_path = self.config.pdfs_dir / filename

        # Group records by document
        docs_data: Dict[str, Dict[str, Any]] = {}

        # If validation was run, export entities
        if self.config.validation_enabled and any(entities.values()):
            for doc_name, ents in entities.items():
                doc_records = []
                for e in ents:
                    if e.status != ValidationStatus.VALIDATED and not self.config.include_rejected:
                        continue
                    doc_records.append({
                        "short_form": e.short_form,
                        "long_form": e.long_form,
                        "field_type": e.field_type.value,
                        "status": e.status.value,
                        "confidence": round(e.confidence_score, 3),
                        "page": e.primary_evidence.location.page_num,
                        "evidence": e.primary_evidence.text[:200] if e.primary_evidence.text else "",
                    })
                if doc_records:
                    docs_data[doc_name] = {
                        "doc_id": ents[0].doc_id if ents else "",
                        "abbreviations": doc_records,
                    }
        else:
            # Export candidates (no validation)
            for doc_name, cands in candidates.items():
                doc_records = []
                for c in cands:
                    # Get context centered around short_form
                    context = self._get_context_with_sf(c.context_text, c.short_form, max_len=200)
                    doc_records.append({
                        "short_form": c.short_form,
                        "long_form": c.long_form,
                        "field_type": c.field_type.value,
                        "generator": c.generator_type.value,
                        "confidence": round(c.initial_confidence, 3),
                        "page": c.context_location.page_num,
                        "context": context,
                        "occurrences": self._count_occurrences(c.short_form, doc_name, candidates),
                    })
                if doc_records:
                    docs_data[doc_name] = {
                        "doc_id": cands[0].doc_id if cands else "",
                        "abbreviations": doc_records,
                    }

        # Build final output structure
        output_data = {
            "run_id": self.config.run_id,
            "timestamp": timestamp,
            "pipeline_version": self.config.pipeline_version,
            "parser_strategy": self.config.parser_strategy,
            "validation_enabled": self.config.validation_enabled,
            "documents": docs_data,
            "summary": {
                "total_documents": len(docs_data),
                "total_abbreviations": sum(
                    len(d["abbreviations"]) for d in docs_data.values()
                ),
            },
        }

        # Write with nice formatting
        with open(output_path, "w", encoding="utf-8") as f:
            if self.config.output_format == "jsonl":
                # JSONL: one document per line (still structured)
                for doc_name, doc_data in docs_data.items():
                    record = {"doc_name": doc_name, **doc_data, "run_id": self.config.run_id}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                # JSON: pretty printed with proper indentation
                json.dump(output_data, f, indent=2, ensure_ascii=False)

        return output_path


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    
    # =========================================================================
    # CONFIGURATION - Edit these paths as needed
    # =========================================================================
    
    CONFIG_PATH = "/Users/frederictetard/Projects/ese/corpus_metadata/document_config/config.yaml"
    PDFS_DIR = "/Users/frederictetard/Projects/ese/Pdfs"
    
    # =========================================================================
    
    config = load_config(CONFIG_PATH, PDFS_DIR)
    result = Orchestrator(config).run()
    
    print(json.dumps(result, indent=2))
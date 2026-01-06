# corpus_metadata/corpus_abbreviations/orchestrator.py
"""
Orchestrator v0.7: PDF -> Parse -> Generate -> Validate -> Log

All configuration is loaded from config.yaml - no hardcoded parameters.

Usage:
    python orchestrator.py

Requires:
    - ANTHROPIC_API_KEY in .env file (or set in config.yaml)
    - pip install anthropic pyyaml python-dotenv
"""

from __future__ import annotations

from pathlib import Path
from dotenv import load_dotenv
import yaml
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

# =============================================================================
# SILENCE KNOWN DEPENDENCY WARNINGS (must be before imports that trigger them)
# =============================================================================
import warnings

# 1) scispaCy AbbreviationDetector: global_matcher has no patterns (benign)
warnings.filterwarnings(
    "ignore",
    message=r".*\[W036\].*component 'matcher' does not have any patterns defined.*",
)

# 2) sklearn version mismatch: UMLS linker loads TfidfVectorizer pickled with 1.1.2
#    TODO: Consider regenerating scispacy artifacts with current sklearn version
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*InconsistentVersionWarning.*",
)
warnings.filterwarnings(
    "ignore",
    module=r"sklearn\.base",
)

# 3) transformers/unstructured: max_size deprecated (low risk, cosmetic)
warnings.filterwarnings(
    "ignore",
    message=r".*max_size.*parameter is deprecated.*",
)

# 4) spaCy tokenizer: FutureWarning about set union (internal regex issue)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"spacy\.language",
)
# =============================================================================

import json
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Ensure imports work
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from A_core.A01_domain_models import Candidate, ExtractedEntity, ValidationStatus, GeneratorType, EvidenceSpan, FieldType, Coordinate, ProvenanceMetadata
from A_core.A03_provenance import generate_run_id, compute_doc_fingerprint, hash_string
from A_core.A04_heuristics_config import (
    HeuristicsConfig, HeuristicsCounters, DEFAULT_HEURISTICS_CONFIG,
    check_context_match, check_trial_id, get_canonical_case
)
from B_parsing.B01_pdf_to_docgraph import PDFToDocGraphParser
from B_parsing.B03_table_extractor import TableExtractor
from C_generators.C01_strategy_abbrev import AbbrevSyntaxCandidateGenerator
from C_generators.C05_strategy_glossary import GlossaryTableCandidateGenerator
from C_generators.C02_strategy_regex import RegexCandidateGenerator
from C_generators.C03_strategy_layout import LayoutCandidateGenerator
from C_generators.C04_strategy_flashtext import RegexLexiconGenerator
from D_validation.D02_llm_engine import ClaudeClient, LLMEngine
from D_validation.D03_validation_logger import ValidationLogger
from E_normalization.E01_term_mapper import TermMapper
from E_normalization.E02_disambiguator import Disambiguator
from F_evaluation.F05_extraction_analysis import run_analysis

# Module constants
PIPELINE_VERSION = "0.7"


class Orchestrator:
    """
    Main pipeline orchestrator.

    All configuration is loaded from config.yaml.

    Stages:
        1. Parse PDF -> DocumentGraph
        2. Generate candidates (syntax + lexicon)
        3. Validate with Claude
        4. Log results to corpus_log/
    """

    # Default config path (all other paths loaded from config)
    DEFAULT_CONFIG = "/Users/frederictetard/Projects/ese/corpus_metadata/corpus_config/config.yaml"

    def __init__(
        self,
        log_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        config_path: Optional[str] = None,
        run_id: Optional[str] = None,
        skip_validation: Optional[bool] = None,
        enable_haiku_screening: Optional[bool] = None,
        heuristics_config: Optional[HeuristicsConfig] = None,
    ):
        # Load configuration from YAML
        self.config_path = config_path or self.DEFAULT_CONFIG
        self.config = self._load_config(self.config_path)

        # Extract paths from config
        paths = self.config.get("paths", {})
        base_path = paths.get("base", "/Users/frederictetard/Projects/ese")

        # Set paths from config (with parameter override)
        self.log_dir = Path(log_dir or Path(base_path) / paths.get("logs", "corpus_log"))
        self.output_dir = Path(output_dir) if output_dir else None
        self.pdf_dir = Path(base_path) / paths.get("pdf_input", "Pdfs")
        self.gold_json = str(Path(base_path) / paths.get("gold_json", "gold_data/papers_gold.json"))

        # Extract API settings from config
        api_cfg = self.config.get("api", {}).get("claude", {})
        val_cfg = api_cfg.get("validation", {})

        # Set model from config (with parameter override)
        self.model = model or val_cfg.get("model", "claude-sonnet-4-20250514")
        self.batch_delay_ms = api_cfg.get("batch_delay_ms", 100)

        # Set validation settings from config (with parameter override)
        validation_cfg = self.config.get("validation", {}).get("abbreviation", {})
        self.skip_validation = skip_validation if skip_validation is not None else not validation_cfg.get("enabled", True)

        # Set heuristics settings from config (with parameter override)
        heur_cfg = self.config.get("heuristics", {})
        self.enable_haiku_screening = enable_haiku_screening if enable_haiku_screening is not None else heur_cfg.get("enable_haiku_screening", False)

        self.run_id = run_id or generate_run_id("RUN")

        # Load heuristics config from YAML
        self.heuristics = heuristics_config or HeuristicsConfig.from_yaml(self.config_path)

        # Extract lexicon paths from config
        lexicons = self.config.get("lexicons", {})
        dict_path = Path(base_path) / paths.get("dictionaries", "ouput_datasources")

        # Parser
        self.parser = PDFToDocGraphParser()
        self.table_extractor = TableExtractor()

        # Extract generator settings from config
        gen_cfg = self.config.get("generators", {})
        regex_cfg = gen_cfg.get("regex_pattern", {})
        lexicon_cfg = gen_cfg.get("lexicon", {})

        # Build lexicon generator config from YAML
        lexicon_gen_config = {
            "run_id": self.run_id,
            "abbrev_lexicon_path": str(dict_path / lexicons.get("abbreviation_general", "2025_08_abbreviation_general.json")),
            "disease_lexicon_path": str(dict_path / lexicons.get("disease_lexicon", "2025_08_lexicon_disease.json")),
            "orphanet_lexicon_path": str(dict_path / lexicons.get("orphanet_diseases", "2025_08_orphanet_diseases.json")),
            "rare_disease_acronyms_path": str(dict_path / lexicons.get("rare_disease_acronyms", "2025_08_rare_disease_acronyms.json")),
            "umls_abbrev_path": str(dict_path / lexicons.get("umls_biological", "2025_08_umls_biological_abbreviations_v5.tsv")),
            "umls_clinical_path": str(dict_path / lexicons.get("umls_clinical", "2025_08_umls_clinical_abbreviations_v5.tsv")),
            "anca_lexicon_path": str(dict_path / lexicons.get("disease_lexicon_anca", "disease_lexicon_anca.json")),
            "igan_lexicon_path": str(dict_path / lexicons.get("disease_lexicon_igan", "disease_lexicon_igan.json")),
            "pah_lexicon_path": str(dict_path / lexicons.get("disease_lexicon_pah", "disease_lexicon_pah.json")),
            "context_window": lexicon_cfg.get("context_window", 300),
        }

        # Generators (all 5 strategies)
        self.generators = [
            AbbrevSyntaxCandidateGenerator(config={"run_id": self.run_id}),   # C01: Syntax/Schwartz-Hearst
            GlossaryTableCandidateGenerator(config={"run_id": self.run_id}),  # C01b: Glossary tables
            RegexCandidateGenerator(config={                                    # C02: Rigid patterns
                "run_id": self.run_id,
                "enabled_types": regex_cfg.get("enabled_types", ["TRIAL_ID", "COMPOUND_ID", "DOI", "PMID", "PMCID"]),
            }),
            LayoutCandidateGenerator(config={"run_id": self.run_id}),         # C03: Layout heuristics
            RegexLexiconGenerator(config=lexicon_gen_config),                 # C04: Lexicon/FlashText
        ]

        # Validation (Claude) - reads config from YAML
        if not self.skip_validation:
            self.claude_client = ClaudeClient(
                api_key=api_key,
                model=self.model,
                config_path=self.config_path,
            )
            self.llm_engine = LLMEngine(
                llm_client=self.claude_client,
                model=self.model,
                run_id=self.run_id,
                max_tokens=val_cfg.get("max_tokens", 450),
                temperature=val_cfg.get("temperature", 0.0),
                top_p=val_cfg.get("top_p", 1.0),
            )
        else:
            self.claude_client = None
            self.llm_engine = None

        # Logger - writes to corpus_log/
        self.logger = ValidationLogger(
            log_dir=str(self.log_dir),
            run_id=self.run_id,
        )

        # Extract normalization settings from config
        norm_cfg = self.config.get("normalization", {})
        term_mapper_cfg = norm_cfg.get("term_mapper", {})
        disambig_cfg = norm_cfg.get("disambiguator", {})

        # Normalization (E01) - canonicalize long forms + attach standard IDs
        self.term_mapper = TermMapper(config={
            "mapping_file_path": str(dict_path / lexicons.get("abbreviation_general", "2025_08_abbreviation_general.json")),
            "enable_fuzzy_matching": term_mapper_cfg.get("enable_fuzzy_matching", False),
            "fuzzy_cutoff": term_mapper_cfg.get("fuzzy_cutoff", 0.90),
            "fill_long_form_for_orphans": term_mapper_cfg.get("fill_long_form_for_orphans", False),
        })

        # Disambiguation (E02) - resolve ambiguous abbreviations using context
        self.disambiguator = Disambiguator(config={
            "min_context_score": disambig_cfg.get("min_context_score", 2),
            "min_margin": disambig_cfg.get("min_margin", 1),
            "fill_long_form_for_orphans": disambig_cfg.get("fill_long_form_for_orphans", True),
        })

        print(f"Orchestrator v{PIPELINE_VERSION} initialized")
        print(f"  Run ID: {self.run_id}")
        print(f"  Config: {self.config_path}")
        print(f"  Model: {self.model}")
        print(f"  Log dir: {self.log_dir}")
        print(f"  Validation: {'ENABLED' if not self.skip_validation else 'DISABLED'}")
        print(f"  Haiku screening: {'ENABLED' if self.enable_haiku_screening else 'OFF'}")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            print(f"[WARN] Config file not found: {config_path}, using defaults")
            return {}

        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[WARN] Failed to load config from {config_path}: {e}")
            return {}

    def _export_extracted_text(self, pdf_path: Path, doc) -> None:
        """
        Export extracted text from parsed PDF to a text file.

        The text file is saved in the same folder as the PDF with format:
        <pdf_stem>_<timestamp>.txt
        """
        # Build output path: same folder as PDF, name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_path = pdf_path.parent / f"{pdf_path.stem}_{timestamp}.txt"

        # Extract text from all blocks
        text_lines = []
        current_page = None

        for block in doc.iter_linear_blocks(skip_header_footer=False):
            # Add page separator when page changes
            if block.page_num != current_page:
                if current_page is not None:
                    text_lines.append("")  # Empty line between pages
                text_lines.append(f"--- Page {block.page_num} ---")
                text_lines.append("")
                current_page = block.page_num

            # Add block text
            text = (block.text or "").strip()
            if text:
                text_lines.append(text)

        # Write to file
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(text_lines))
            print(f"  Extracted text: {txt_path.name}")
        except Exception as e:
            print(f"  [WARN] Failed to export text: {e}")

    def process_pdf(
        self,
        pdf_path: str,
        batch_delay_ms: Optional[float] = None,
    ) -> List[ExtractedEntity]:
        """
        Process a single PDF through the full pipeline.
        """
        # Use config value if not specified
        if batch_delay_ms is None:
            batch_delay_ms = self.batch_delay_ms

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

        # Export extracted text to file
        self._export_extracted_text(pdf_path, doc)

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
        # 2. SF appears 2+ times in document (strict for precision)
        def should_keep(c: Candidate) -> bool:
            if c.generator_type != GeneratorType.LEXICON_MATCH:
                return True
            sf_upper = c.short_form.upper()
            if sf_upper in corroborated_sfs:
                return True
            # Strict: require SF appears at least twice (1C)
            if word_counts.get(sf_upper, 0) >= 2:
                return True
            return False

        needs_validation: List[Candidate] = [c for c in unique_candidates if should_keep(c)]

        # ========================================
        # UPSTREAM REDUCTION: Reduce LEXICON_MATCH before LLM (1A, 1B)
        # ========================================

        # Use centralized config for allowlists
        ALLOWED_2LETTER_SFS = self.heuristics.allowed_2letter_sfs
        ALLOWED_MIXED_CASE = self.heuristics.allowed_mixed_case

        def is_valid_sf_form(sf: str, context: str = "") -> bool:
            """1B: Filter SF by form - reject non-abbreviation patterns."""
            sf_upper = sf.upper()

            # Allow 2-letter uppercase if in allowlist
            if len(sf) == 2 and sf_upper in ALLOWED_2LETTER_SFS:
                return True

            # Allow known mixed-case/software abbreviations
            if sf_upper in ALLOWED_MIXED_CASE:
                return True

            # Special case: Ig/IG only if near immunoglobulin context
            if sf_upper == 'IG':
                ctx_lower = context.lower()
                if any(x in ctx_lower for x in ['igg', 'iga', 'igm', 'ige', 'immunoglobulin']):
                    return True
                return False  # Reject ambiguous IG without context

            # Reject if mostly lowercase and longer than 6 chars (likely a word)
            if len(sf) > 6 and sf.islower():
                return False
            # Reject if all lowercase and > 4 chars (e.g., "protein", "study")
            if sf.islower() and len(sf) > 4:
                return False
            # Reject if looks like a regular capitalized word (e.g., "Medications")
            if len(sf) > 6 and sf[0].isupper() and sf[1:].islower():
                return False
            return True

        def score_lf_quality(c: Candidate, full_text: str) -> int:
            """1A: Score LF quality for dedup ranking."""
            score = 0
            lf = (c.long_form or "").lower()
            sf = c.short_form

            # LF appears near SF in document (highest priority)
            if lf and sf in full_text:
                # Check if LF appears within 200 chars of SF
                sf_positions = [m.start() for m in re.finditer(re.escape(sf), full_text)]
                for pos in sf_positions[:5]:  # Check first 5 occurrences
                    window = full_text[max(0, pos-200):pos+200].lower()
                    if lf in window:
                        score += 100
                        break

            # LF appears anywhere in document
            if lf and lf in full_text.lower():
                score += 50

            # Prefer shorter, more "definitional" LFs
            if lf:
                # Penalize very long LFs (likely not definitions)
                if len(lf) > 60:
                    score -= 20
                # Penalize LFs with verbs/articles
                if any(w in lf.split() for w in ['the', 'a', 'an', 'is', 'are', 'was']):
                    score -= 10

            # Prefer candidates from trusted lexicons
            if c.provenance and c.provenance.lexicon_source:
                if 'umls' in c.provenance.lexicon_source.lower():
                    score += 30

            return score

        # 1A: Dedup LEXICON_MATCH by SF - keep top 2 per SF
        lexicon_by_sf: Dict[str, List[Candidate]] = {}
        non_lexicon: List[Candidate] = []
        sf_form_rejected = 0

        for c in needs_validation:
            if c.generator_type == GeneratorType.LEXICON_MATCH:
                sf_upper = c.short_form.upper()
                ctx = c.context_text or ""
                # 1B: Apply SF form filter (with context for IG special case)
                if not is_valid_sf_form(c.short_form, ctx):
                    sf_form_rejected += 1
                    continue
                if sf_upper not in lexicon_by_sf:
                    lexicon_by_sf[sf_upper] = []
                lexicon_by_sf[sf_upper].append(c)
            else:
                non_lexicon.append(c)

        # Keep top 2 LFs per SF based on quality score
        deduped_lexicon: List[Candidate] = []
        for sf, candidates in lexicon_by_sf.items():
            if len(candidates) <= 2:
                deduped_lexicon.extend(candidates)
            else:
                # Score and rank by LF quality
                scored = [(c, score_lf_quality(c, full_text)) for c in candidates]
                scored.sort(key=lambda x: -x[1])  # Highest score first
                deduped_lexicon.extend([c for c, _ in scored[:2]])

        lexicon_before = sum(len(v) for v in lexicon_by_sf.values()) + sf_form_rejected
        lexicon_after = len(deduped_lexicon)
        print(f"  LEXICON_MATCH reduction: {lexicon_before} -> {lexicon_after} "
              f"(dedup: {lexicon_before - sf_form_rejected - lexicon_after}, "
              f"form filter: {sf_form_rejected})")

        needs_validation = non_lexicon + deduped_lexicon
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

        # Use centralized config for common words
        COMMON_WORDS = self.heuristics.common_words

        # ========================================
        # PASO A: Stats whitelist - deterministic (no LLM)
        # These are auto-approved if numeric evidence is present
        # ========================================
        STATS_ABBREVS = self.heuristics.stats_abbrevs  # {SF: canonical_LF}

        # ========================================
        # PASO B: Country codes - deterministic (no LLM)
        # ========================================
        COUNTRY_ABBREVS = self.heuristics.country_abbrevs  # {SF: canonical_LF}

        # Blacklist: words that look like abbreviations but aren't
        # These will be auto-rejected even if in lexicon
        SF_BLACKLIST = self.heuristics.sf_blacklist

        # ========================================
        # PASO C: Hyphenated abbreviations - search in full text
        # These are often missed by standard generators
        # ========================================
        HYPHENATED_ABBREVS = self.heuristics.hyphenated_abbrevs

        # Direct text search for abbreviations often missed by lexicon
        # These are searched in full_text if not already found
        DIRECT_SEARCH_ABBREVS = self.heuristics.direct_search_abbrevs

        def has_numeric_evidence(context: str, sf: str) -> bool:
            """Check if SF appears with numeric evidence (digits, %, =, :)."""
            if not context:
                return False
            ctx = context.lower()
            sf_lower = sf.lower()

            # Find SF position and check Â±30 chars window
            idx = ctx.find(sf_lower)
            if idx == -1:
                return False

            window_start = max(0, idx - 30)
            window_end = min(len(ctx), idx + len(sf) + 30)
            window = ctx[window_start:window_end]

            # Check for numeric patterns
            import re
            # Digits, %, =, :, or typical stat patterns
            if re.search(r'\d', window):
                return True
            if '%' in window or '=' in window:
                return True
            # Patterns like "95% CI" or "OR 1.2"
            if re.search(rf'{sf_lower}\s*[:=]?\s*[\d\.\-]', window):
                return True
            if re.search(rf'[\d\.]+\s*-\s*[\d\.]+', window):  # Range like 1.2-3.4
                return True

            return False

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

        # Initialize heuristics counters for logging
        counters = HeuristicsCounters()

        for c in needs_validation:
            sf_upper = c.short_form.upper()
            ctx = c.context_text or ""

            # ========================================
            # PASO B: Auto-reject blacklisted SFs (before other checks)
            # ========================================
            if sf_upper in SF_BLACKLIST:
                entity = _create_auto_entity(
                    c, ValidationStatus.REJECTED, 0.95,
                    "Blacklisted: not a valid abbreviation in this context",
                    ["auto_rejected_blacklist"], {"auto": "blacklist"}
                )
                auto_results.append((c, entity))
                counters.blacklisted_fp_count += 1
                auto_rejected_count += 1
                continue

            # ========================================
            # Contextual rejection for ambiguous SFs (Option B)
            # ========================================
            if not check_context_match(c.short_form, ctx, self.heuristics):
                entity = _create_auto_entity(
                    c, ValidationStatus.REJECTED, 0.90,
                    "Rejected: ambiguous SF without required medical context",
                    ["auto_rejected_context"], {"auto": "context_mismatch"}
                )
                auto_results.append((c, entity))
                counters.context_rejected += 1
                auto_rejected_count += 1
                continue

            # ========================================
            # Exclude trial IDs if configured (NCT\d+)
            # ========================================
            if check_trial_id(c.short_form, self.heuristics):
                entity = _create_auto_entity(
                    c, ValidationStatus.REJECTED, 0.90,
                    "Excluded: trial identifier (NCT number)",
                    ["auto_rejected_trial_id"], {"auto": "trial_id_excluded"}
                )
                auto_results.append((c, entity))
                counters.trial_id_excluded += 1
                auto_rejected_count += 1
                continue

            # ========================================
            # PASO A: Auto-approve stats with numeric evidence (NO LLM)
            # ========================================
            if sf_upper in STATS_ABBREVS and has_numeric_evidence(ctx, c.short_form):
                canonical_lf = STATS_ABBREVS.get(sf_upper)
                entity = _create_auto_entity(
                    c, ValidationStatus.VALIDATED, 0.90,
                    f"Stats abbreviation with numeric evidence",
                    ["auto_approved_stats"],
                    {"auto": "stats_whitelist", "canonical_lf": canonical_lf}
                )
                if canonical_lf:
                    entity = entity.model_copy(update={"long_form": canonical_lf})
                auto_results.append((c, entity))
                counters.recovered_by_stats_whitelist += 1
                auto_approved_count += 1
                continue

            # ========================================
            # PASO B: Auto-approve country codes (NO LLM)
            # ========================================
            if sf_upper in COUNTRY_ABBREVS:
                canonical_lf = COUNTRY_ABBREVS.get(sf_upper)
                entity = _create_auto_entity(
                    c, ValidationStatus.VALIDATED, 0.90,
                    f"Country code abbreviation",
                    ["auto_approved_country"],
                    {"auto": "country_code", "canonical_lf": canonical_lf}
                )
                if canonical_lf:
                    entity = entity.model_copy(update={"long_form": canonical_lf})
                auto_results.append((c, entity))
                counters.recovered_by_country_code += 1
                auto_approved_count += 1
                continue

            # Auto-reject: Common English words
            if sf_upper in COMMON_WORDS:
                entity = _create_auto_entity(
                    c, ValidationStatus.REJECTED, 0.95,
                    "Common English word, not an abbreviation",
                    ["auto_rejected"], {"auto": "rejected_common_word"}
                )
                auto_results.append((c, entity))
                counters.common_word_rejected += 1
                auto_rejected_count += 1
                continue

            # Needs LLM validation
            llm_candidates.append(c)

        print(f"\n[3/4] Validating candidates with Claude...")
        print(f"  Corroborated SFs: {len(corroborated_sfs)}")
        print(f"  Frequent SFs (2+ occurrences): {frequent_sfs}")
        print(f"  Filtered (lexicon-only, rare): {filtered_count}")
        print(f"  Auto-approved stats (PASO A): {counters.recovered_by_stats_whitelist}")
        print(f"  Auto-approved country (PASO B): {counters.recovered_by_country_code}")
        print(f"  Auto-rejected blacklist: {counters.blacklisted_fp_count}")
        print(f"  Auto-rejected context: {counters.context_rejected}")
        print(f"  Auto-rejected trial IDs: {counters.trial_id_excluded}")
        print(f"  Auto-rejected common words: {counters.common_word_rejected}")
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

        # Hybrid validation: batch for high-precision sources, individual for rest
        # SYNTAX_PATTERN + GLOSSARY_TABLE = explicit pairs (high precision, batch OK)
        # LEXICON_MATCH = FP-prone, needs individual validation
        explicit_candidates = []
        lexicon_candidates = []

        for c in llm_candidates:
            gen = c.generator_type.value if c.generator_type else ""
            if "syntax" in gen.lower() or "glossary" in gen.lower():
                explicit_candidates.append(c)
            else:
                lexicon_candidates.append(c)

        print(f"  Batch (explicit pairs): {len(explicit_candidates)}")
        print(f"  Individual (lexicon): {len(lexicon_candidates)}")

        # ========================================
        # HAIKU FAST-REJECT: Screen lexicon candidates before Sonnet
        # (disabled by default - enable with enable_haiku_screening=True)
        # ========================================
        haiku_rejected: List[ExtractedEntity] = []
        if self.enable_haiku_screening and lexicon_candidates:
            print(f"\n  Haiku screening {len(lexicon_candidates)} lexicon candidates...")
            haiku_start = time.time()

            needs_review, haiku_rejected = self.llm_engine.fast_reject_batch(
                lexicon_candidates,
                haiku_model="claude-3-5-haiku-20241022",
                batch_size=20,
            )

            haiku_time = time.time() - haiku_start
            print(f"    Haiku rejected: {len(haiku_rejected)}")
            print(f"    Needs Sonnet review: {len(needs_review)}")
            print(f"    Haiku time: {haiku_time:.2f}s")

            # Log Haiku rejections
            for entity in haiku_rejected:
                # Find matching candidate for logging
                candidate = next(
                    (c for c in lexicon_candidates if str(c.id) == str(entity.candidate_id)),
                    None
                )
                if candidate:
                    self.logger.log_validation(
                        candidate=candidate,
                        entity=entity,
                        llm_response=entity.raw_llm_response,
                        elapsed_ms=haiku_time * 1000 / len(lexicon_candidates),
                    )
                results.append(entity)

            # Update lexicon_candidates to only those needing Sonnet
            lexicon_candidates = needs_review

        # Batch validate explicit pairs (high precision, ~10x speedup)
        if explicit_candidates:
            try:
                val_start = time.time()
                batch_results = self.llm_engine.verify_candidates_batch(
                    explicit_candidates, batch_size=10
                )
                elapsed_ms = (time.time() - val_start) * 1000

                for candidate, entity in zip(explicit_candidates, batch_results):
                    self.logger.log_validation(
                        candidate=candidate,
                        entity=entity,
                        llm_response=entity.raw_llm_response,
                        elapsed_ms=elapsed_ms / len(explicit_candidates),
                    )
                    results.append(entity)
            except Exception as e:
                print(f"  [WARN] Batch error, falling back to individual: {e}")
                for candidate in explicit_candidates:
                    try:
                        entity = self.llm_engine.verify_candidate(candidate)
                        self.logger.log_validation(
                            candidate=candidate, entity=entity,
                            llm_response=entity.raw_llm_response, elapsed_ms=0,
                        )
                        results.append(entity)
                    except Exception as e2:
                        self.logger.log_error(candidate, str(e2))

        # Individual validation for lexicon matches (FP-prone, needs full context)
        batch_size = 15
        for batch_start in range(0, len(lexicon_candidates), batch_size):
            batch_end = min(batch_start + batch_size, len(lexicon_candidates))
            batch = lexicon_candidates[batch_start:batch_end]

            print(f"  Lexicon progress: {batch_end}/{len(lexicon_candidates)}")

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
            if batch_delay_ms > 0 and batch_end < len(lexicon_candidates):
                time.sleep(batch_delay_ms / 1000)

        print(f"  Time: {time.time() - start:.2f}s")

        # ========================================
        # PASO C: Detect hyphenated abbreviations in full text
        # These are often missed by standard generators
        # ========================================
        # Get all SFs already found
        found_sfs = {r.short_form.upper() for r in results if r.status == ValidationStatus.VALIDATED}

        for hyph_sf, hyph_lf in HYPHENATED_ABBREVS.items():
            if hyph_sf.upper() in found_sfs:
                continue  # Already found

            # Search for hyphenated SF in full text (case-insensitive)
            import re
            # Match the hyphenated pattern with word boundaries
            pattern = rf'\b{re.escape(hyph_sf)}\b'
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                # Found! Create a synthetic entity
                context_start = max(0, match.start() - 100)
                context_end = min(len(full_text), match.end() + 100)
                context_snippet = full_text[context_start:context_end]

                ctx_hash = hash_string(context_snippet)
                primary = EvidenceSpan(
                    text=context_snippet,
                    location=Coordinate(page_num=1),  # Default to page 1
                    scope_ref=ctx_hash,
                    start_char_offset=match.start() - context_start,
                    end_char_offset=match.end() - context_start,
                )

                prov = ProvenanceMetadata(
                    pipeline_version=PIPELINE_VERSION,
                    run_id=self.run_id,
                    doc_fingerprint="hyphenated_detection",
                    generator_name=GeneratorType.LEXICON_MATCH,
                    rule_version="hyphenated_detector:v1.0",
                    lexicon_source="orchestrator:hyphenated",
                )

                import uuid
                entity = ExtractedEntity(
                    candidate_id=uuid.uuid4(),
                    doc_id=str(pdf_path.stem) if hasattr(pdf_path, 'stem') else str(pdf_path),
                    field_type=FieldType.DEFINITION_PAIR,
                    short_form=match.group(),  # Preserve original case
                    long_form=hyph_lf,
                    primary_evidence=primary,
                    supporting_evidence=[],
                    status=ValidationStatus.VALIDATED,
                    confidence_score=0.85,
                    rejection_reason=None,
                    validation_flags=["auto_approved_hyphenated"],
                    provenance=prov,
                    raw_llm_response={"auto": "hyphenated_detector"},
                )
                results.append(entity)
                counters.recovered_by_hyphen += 1

        if counters.recovered_by_hyphen > 0:
            print(f"  Hyphenated detected (PASO C): {counters.recovered_by_hyphen}")

        # ========================================
        # Direct text search for missed abbreviations (UK, RR, CC BY, etc.)
        # ========================================
        for direct_sf, direct_lf in DIRECT_SEARCH_ABBREVS.items():
            if direct_sf.upper() in found_sfs:
                continue  # Already found

            # Search for SF in full text
            # Use word boundary for single words, exact match for multi-word
            if ' ' in direct_sf:
                # Multi-word: exact match
                pattern = re.escape(direct_sf)
            else:
                # Single word: word boundary
                pattern = rf'\b{re.escape(direct_sf)}\b'

            match = re.search(pattern, full_text)
            if match:
                # Found! Create entity
                context_start = max(0, match.start() - 100)
                context_end = min(len(full_text), match.end() + 100)
                context_snippet = full_text[context_start:context_end]

                ctx_hash = hash_string(context_snippet)
                primary = EvidenceSpan(
                    text=context_snippet,
                    location=Coordinate(page_num=1),
                    scope_ref=ctx_hash,
                    start_char_offset=match.start() - context_start,
                    end_char_offset=match.end() - context_start,
                )

                prov = ProvenanceMetadata(
                    pipeline_version=PIPELINE_VERSION,
                    run_id=self.run_id,
                    doc_fingerprint="direct_search",
                    generator_name=GeneratorType.LEXICON_MATCH,
                    rule_version="direct_search:v1.0",
                    lexicon_source="orchestrator:direct_search",
                )

                entity = ExtractedEntity(
                    candidate_id=uuid.uuid4(),
                    doc_id=str(pdf_path.stem) if hasattr(pdf_path, 'stem') else str(pdf_path),
                    field_type=FieldType.DEFINITION_PAIR,
                    short_form=match.group(),
                    long_form=direct_lf,
                    primary_evidence=primary,
                    supporting_evidence=[],
                    status=ValidationStatus.VALIDATED,
                    confidence_score=0.85,
                    rejection_reason=None,
                    validation_flags=["auto_approved_direct_search"],
                    provenance=prov,
                    raw_llm_response={"auto": "direct_search"},
                )
                results.append(entity)
                found_sfs.add(direct_sf.upper())
                counters.recovered_by_direct_search += 1

        if counters.recovered_by_direct_search > 0:
            print(f"  Direct search detected (UK, RR, CC BY): {counters.recovered_by_direct_search}")

        # ========================================
        # PASO D: LLM Extractor SF-only
        # Find abbreviations NOT in lexicon using LLM
        # Guardrails: SF must be exact substring in text
        # ========================================
        print(f"\n  Running LLM SF-only extractor (PASO D)...")

        # Get all SFs already VALIDATED (don't skip REJECTED - they might be valid SF-only)
        already_found_sfs = {r.short_form.upper() for r in results if r.status == ValidationStatus.VALIDATED}

        # For LLM extraction, only skip VALIDATED (not REJECTED - they might be valid SFs with wrong LF)
        # This allows us to rescue SF-only even if the pair was rejected
        all_processed_sfs = already_found_sfs.copy()

        # Sample chunks of text for LLM analysis (to stay within context limits)
        chunk_size = 3000
        text_chunks = []
        for i in range(0, min(len(full_text), 15000), chunk_size):
            text_chunks.append(full_text[i:i+chunk_size])

        # Prompt for SF-only extraction
        sf_extraction_prompt = """You are an expert at identifying medical/scientific abbreviations in text.

Given the following text, identify ALL abbreviations that appear. An abbreviation is typically:
- 2-10 characters long
- Contains uppercase letters (e.g., "FDA", "eGFR", "IL-6", "C3a")
- May contain numbers or hyphens
- Represents a longer term/concept

IMPORTANT RULES:
1. Only return abbreviations that EXACTLY appear in the text (case-sensitive)
2. Do NOT include common words (the, and, for, etc.)
3. Do NOT include author names, months, or country names
4. Include multi-word abbreviations like "CC BY" if present
5. Include mixed-case abbreviations like "eGFR", "IgA", "C3a"

Already found (DO NOT include these): {already_found}

Text to analyze:
{text}

Return a JSON array of abbreviation strings found. Example: ["DOI", "CC BY", "IgA", "C3a"]
Return ONLY the JSON array, nothing else."""

        llm_sf_candidates = set()
        llm_errors = 0

        for i, chunk in enumerate(text_chunks):
            try:
                prompt = sf_extraction_prompt.format(
                    already_found=", ".join(sorted(already_found_sfs)[:50]),
                    text=chunk
                )

                response = self.claude_client.complete_json_any(
                    system_prompt="You are an abbreviation extraction assistant. Return only valid JSON arrays.",
                    user_prompt=prompt,
                    model=self.model,
                    temperature=0.0,
                    max_tokens=500,
                    top_p=1.0,
                )

                # Response is already parsed (list or dict)
                if isinstance(response, list):
                    for sf in response:
                        if isinstance(sf, str) and sf.strip():
                            llm_sf_candidates.add(sf.strip())
                elif isinstance(response, dict) and "abbreviations" in response:
                    for sf in response.get("abbreviations", []):
                        if isinstance(sf, str) and sf.strip():
                            llm_sf_candidates.add(sf.strip())
            except Exception as e:
                llm_errors += 1

        print(f"    LLM chunks processed: {len(text_chunks)}, errors: {llm_errors}")
        print(f"    LLM candidates found: {len(llm_sf_candidates)}")
        if llm_sf_candidates:
            print(f"    Candidates: {sorted(llm_sf_candidates)[:20]}")

        # Validate and add LLM-found SFs
        llm_filtered_already = 0
        llm_filtered_blacklist = 0
        llm_filtered_not_in_text = 0
        llm_filtered_form = 0
        for sf_candidate in llm_sf_candidates:
            sf_upper = sf_candidate.upper()

            # Skip if already processed
            if sf_upper in all_processed_sfs:
                llm_filtered_already += 1
                continue

            # Skip common words and blacklist
            if sf_upper in COMMON_WORDS or sf_upper in SF_BLACKLIST:
                llm_filtered_blacklist += 1
                continue

            # Guardrail: SF must appear exactly in text
            if sf_candidate not in full_text:
                # Try case-insensitive match
                pattern = rf'\b{re.escape(sf_candidate)}\b'
                match = re.search(pattern, full_text, re.IGNORECASE)
                if not match:
                    llm_filtered_not_in_text += 1
                    continue
                sf_candidate = match.group()  # Use actual case from text

            # Validate SF form (basic sanity check)
            if len(sf_candidate) < 2 or len(sf_candidate) > 15:
                llm_filtered_form += 1
                continue
            if not any(c.isupper() for c in sf_candidate):
                llm_filtered_form += 1
                continue

            # Create entity (SF-only, no LF)
            # Find context around the SF
            idx = full_text.find(sf_candidate)
            if idx == -1:
                idx = full_text.lower().find(sf_candidate.lower())

            context_start = max(0, idx - 100)
            context_end = min(len(full_text), idx + len(sf_candidate) + 100)
            context_snippet = full_text[context_start:context_end]

            ctx_hash = hash_string(context_snippet)
            primary = EvidenceSpan(
                text=context_snippet,
                location=Coordinate(page_num=1),
                scope_ref=ctx_hash,
                start_char_offset=idx - context_start if idx >= context_start else 0,
                end_char_offset=idx - context_start + len(sf_candidate) if idx >= context_start else len(sf_candidate),
            )

            prov = ProvenanceMetadata(
                pipeline_version=PIPELINE_VERSION,
                run_id=self.run_id,
                doc_fingerprint="llm_sf_extraction",
                generator_name=GeneratorType.LEXICON_MATCH,
                rule_version="llm_sf_extractor:v1.0",
                lexicon_source="orchestrator:llm_sf_only",
            )

            entity = ExtractedEntity(
                candidate_id=uuid.uuid4(),
                doc_id=str(pdf_path.stem) if hasattr(pdf_path, 'stem') else str(pdf_path),
                field_type=FieldType.SHORT_FORM_ONLY,  # SF-only, no LF
                short_form=sf_candidate,
                long_form=None,  # No LF - avoid hallucination
                primary_evidence=primary,
                supporting_evidence=[],
                status=ValidationStatus.VALIDATED,
                confidence_score=0.75,  # Lower confidence for LLM-extracted
                rejection_reason=None,
                validation_flags=["llm_sf_extracted"],
                provenance=prov,
                raw_llm_response={"auto": "llm_sf_extractor"},
            )
            results.append(entity)
            all_processed_sfs.add(sf_upper)
            counters.recovered_by_llm_sf_only += 1

        print(f"  LLM SF-only extracted (PASO D): {counters.recovered_by_llm_sf_only}")
        if llm_sf_candidates:
            print(f"    Filtered: already={llm_filtered_already}, blacklist={llm_filtered_blacklist}, not_in_text={llm_filtered_not_in_text}, form={llm_filtered_form}")

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

        # Log heuristics counters for debugging
        counters.log_summary()

        # Export results
        self._export_results(pdf_path, results, unique_candidates, counters)

        # Print validated abbreviations
        validated = [r for r in results if r.status == ValidationStatus.VALIDATED]
        if validated:
            print(f"\nValidated abbreviations ({len(validated)}):")
            for v in validated:
                lf = v.long_form or "(no expansion)"
                print(f"  * {v.short_form} -> {lf}")

        return results

    def _export_results(
        self,
        pdf_path: Path,
        results: List[ExtractedEntity],
        candidates: List[Candidate],
        counters: Optional[HeuristicsCounters] = None,
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
            "heuristics_counters": counters.to_dict() if counters else None,
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

        # Run F06 extraction analysis (screen output only)
        run_analysis(export_data, self.gold_json)

    def process_folder(
        self,
        folder_path: Optional[str] = None,
        batch_delay_ms: float = 100,
    ) -> Dict[str, List[ExtractedEntity]]:
        """
        Process all PDFs in a folder.
        
        Returns dict of {pdf_name: [entities]}
        """
        folder = Path(folder_path or self.pdf_dir)
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
                print(f"  [WARN] ERROR processing {pdf_path.name}: {e}")
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
# corpus_metadata/corpus_extraction/orchestrator.py
"""
Orchestrator v0.8: PDF -> Parse -> Generate -> Validate -> Log

All configuration is loaded from config.yaml - no hardcoded parameters.

Usage:
    python orchestrator.py

Requires:
    - ANTHROPIC_API_KEY in .env file (or set in config.yaml)
    - pip install anthropic pyyaml python-dotenv
"""

from __future__ import annotations

# =============================================================================
# SILENCE WARNINGS FIRST (must be before library imports)
# =============================================================================
import os
import warnings

# Suppress at environment level for subprocesses
os.environ["PYTHONWARNINGS"] = (
    "ignore::UserWarning,ignore::FutureWarning,ignore::DeprecationWarning"
)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

_WARNING_FILTERS = [
    (UserWarning, r".*W036.*"),
    (UserWarning, r".*matcher.*does not have any patterns.*"),
    (UserWarning, r".*InconsistentVersionWarning.*"),
    (UserWarning, r".*max_size.*deprecated.*"),
    (FutureWarning, r".*max_size.*deprecated.*"),
    (DeprecationWarning, r".*max_size.*"),
]

for _cat, _pat in _WARNING_FILTERS:
    warnings.filterwarnings("ignore", category=_cat, message=_pat)

warnings.filterwarnings("ignore", module=r"sklearn\.base")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"spacy\.language")
warnings.filterwarnings("ignore", category=UserWarning, module=r"transformers")
# =============================================================================

import heapq
import json
import re
import sys
import time
import uuid
from collections import Counter
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
# =============================================================================

# Ensure imports work
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from A_core.A01_domain_models import (
    Candidate,
    Coordinate,
    EvidenceSpan,
    ExtractedEntity,
    FieldType,
    GeneratorType,
    ProvenanceMetadata,
    ValidationStatus,
)
from A_core.A03_provenance import generate_run_id, hash_string
from A_core.A04_heuristics_config import (
    HeuristicsConfig,
    HeuristicsCounters,
    check_context_match,
    check_trial_id,
)
from A_core.A05_disease_models import (
    DiseaseCandidate,
    DiseaseExportDocument,
    DiseaseExportEntry,
    ExtractedDisease,
)
from B_parsing.B01_pdf_to_docgraph import PDFToDocGraphParser
from B_parsing.B03_table_extractor import TableExtractor
from C_generators.C01_strategy_abbrev import AbbrevSyntaxCandidateGenerator
from C_generators.C02_strategy_regex import RegexCandidateGenerator
from C_generators.C03_strategy_layout import LayoutCandidateGenerator
from C_generators.C04_strategy_flashtext import RegexLexiconGenerator
from C_generators.C05_strategy_glossary import GlossaryTableCandidateGenerator
from C_generators.C06_strategy_disease import DiseaseDetector
from D_validation.D02_llm_engine import ClaudeClient, LLMEngine
from D_validation.D03_validation_logger import ValidationLogger
from E_normalization.E01_term_mapper import TermMapper
from E_normalization.E02_disambiguator import Disambiguator
from E_normalization.E03_disease_normalizer import DiseaseNormalizer
from F_evaluation.F05_extraction_analysis import run_analysis

PIPELINE_VERSION = "0.8"


# =============================================================================
# HELPER FUNCTIONS (module-level for reuse)
# =============================================================================


def extract_context_snippet(
    full_text: str, match_start: int, match_end: int, window: int = 100
) -> str:
    """Extract context snippet around a match position."""
    start = max(0, match_start - window)
    end = min(len(full_text), match_end + window)
    return full_text[start:end]


def has_numeric_evidence(context: str, sf: str) -> bool:
    """Check if SF appears with numeric evidence (digits, %, =, :)."""
    if not context:
        return False
    ctx = context.lower()
    sf_lower = sf.lower()

    idx = ctx.find(sf_lower)
    if idx == -1:
        return False

    window = ctx[max(0, idx - 30) : min(len(ctx), idx + len(sf) + 30)]

    # Check for numeric patterns
    if re.search(r"\d", window) or "%" in window or "=" in window:
        return True
    if re.search(rf"{re.escape(sf_lower)}\s*[:=]?\s*[\d.\-]", window):
        return True
    if re.search(r"[\d.]+\s*-\s*[\d.]+", window):
        return True
    return False


def is_valid_sf_form(
    sf: str, context: str, allowed_2letter: set, allowed_mixed: set
) -> bool:
    """Filter SF by form - reject non-abbreviation patterns."""
    sf_upper = sf.upper()

    if len(sf) == 2 and sf_upper in allowed_2letter:
        return True
    if sf_upper in allowed_mixed:
        return True

    # Special case: IG only if near immunoglobulin context
    if sf_upper == "IG":
        ctx_lower = context.lower()
        return any(
            x in ctx_lower for x in ["igg", "iga", "igm", "ige", "immunoglobulin"]
        )

    # Reject lowercase words
    if sf.islower() and len(sf) > 4:
        return False
    if len(sf) > 6 and sf.islower():
        return False
    # Reject capitalized words (e.g., "Medications")
    if len(sf) > 6 and sf[0].isupper() and sf[1:].islower():
        return False
    return True


def score_lf_quality(
    candidate: Candidate, full_text: str, full_text_lower: Optional[str] = None
) -> int:
    """Score LF quality for dedup ranking.

    Args:
        candidate: The candidate to score
        full_text: Original full text
        full_text_lower: Pre-cached lowercase version (optional, avoids repeated .lower() calls)
    """
    score = 0
    lf = (candidate.long_form or "").lower()
    sf = candidate.short_form

    # Use cached lowercase or compute once
    text_lower = full_text_lower if full_text_lower is not None else full_text.lower()

    if lf and sf in full_text:
        # Check if LF appears within 200 chars of SF - use islice to avoid materializing all matches
        for m in islice(re.finditer(re.escape(sf), full_text), 5):
            window_start = max(0, m.start() - 200)
            window_end = m.start() + 200
            window = text_lower[window_start:window_end]
            if lf in window:
                score += 100
                break

    if lf and lf in text_lower:
        score += 50

    if lf:
        if len(lf) > 60:
            score -= 20
        lf_words = set(lf.split())
        if lf_words & {"the", "a", "an", "is", "are", "was"}:
            score -= 10

    if candidate.provenance and candidate.provenance.lexicon_source:
        if "umls" in candidate.provenance.lexicon_source.lower():
            score += 30

    return score


# =============================================================================
# ORCHESTRATOR CLASS
# =============================================================================


class Orchestrator:
    """
    Main pipeline orchestrator.

    Stages:
        1. Parse PDF -> DocumentGraph
        2. Generate candidates (syntax + lexicon)
        3. Validate with Claude
        4. Log results to corpus_log/
    """

    DEFAULT_CONFIG = (
        "/Users/frederictetard/Projects/ese/corpus_metadata/corpus_config/config.yaml"
    )

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
        self.config_path = config_path or self.DEFAULT_CONFIG
        self.config = self._load_config(self.config_path)

        # Extract paths from config
        paths = self.config.get("paths", {})
        base_path = paths.get("base", "/Users/frederictetard/Projects/ese")

        self.log_dir = Path(
            log_dir or Path(base_path) / paths.get("logs", "corpus_log")
        )
        self.output_dir = Path(output_dir) if output_dir else None
        self.pdf_dir = Path(base_path) / paths.get("pdf_input", "Pdfs")
        self.gold_json = str(
            Path(base_path) / paths.get("gold_json", "gold_data/papers_gold.json")
        )

        # API settings
        api_cfg = self.config.get("api", {}).get("claude", {})
        val_cfg = api_cfg.get("validation", {})

        self.model = model or val_cfg.get("model", "claude-sonnet-4-20250514")
        self.batch_delay_ms = api_cfg.get("batch_delay_ms", 100)

        # Validation settings
        validation_cfg = self.config.get("validation", {}).get("abbreviation", {})
        self.skip_validation = (
            skip_validation
            if skip_validation is not None
            else not validation_cfg.get("enabled", True)
        )

        # Heuristics settings
        heur_cfg = self.config.get("heuristics", {})
        self.enable_haiku_screening = (
            enable_haiku_screening
            if enable_haiku_screening is not None
            else heur_cfg.get("enable_haiku_screening", False)
        )

        self.run_id = run_id or generate_run_id("RUN")
        self.heuristics = heuristics_config or HeuristicsConfig.from_yaml(
            self.config_path
        )

        # Initialize components
        self._init_components(paths, base_path, api_key, val_cfg)

        print(f"Orchestrator v{PIPELINE_VERSION} initialized")
        print(f"  Run ID: {self.run_id}")
        print(f"  Config: {self.config_path}")
        print(f"  Model: {self.model}")
        print(f"  Log dir: {self.log_dir}")
        print(f"  Validation: {'ENABLED' if not self.skip_validation else 'DISABLED'}")
        print(
            f"  Haiku screening: {'ENABLED' if self.enable_haiku_screening else 'OFF'}"
        )

    def _init_components(
        self, paths: dict, base_path: str, api_key: Optional[str], val_cfg: dict
    ) -> None:
        """Initialize parser, generators, validation, and normalization components."""
        lexicons = self.config.get("lexicons", {})
        dict_path = Path(base_path) / paths.get("dictionaries", "ouput_datasources")

        # Parser
        self.parser = PDFToDocGraphParser()
        self.table_extractor = TableExtractor()

        # Generators
        gen_cfg = self.config.get("generators", {})
        regex_cfg = gen_cfg.get("regex_pattern", {})
        lexicon_cfg = gen_cfg.get("lexicon", {})

        lexicon_gen_config = {
            "run_id": self.run_id,
            "abbrev_lexicon_path": str(
                dict_path
                / lexicons.get(
                    "abbreviation_general", "2025_08_abbreviation_general.json"
                )
            ),
            "disease_lexicon_path": str(
                dict_path
                / lexicons.get("disease_lexicon", "2025_08_lexicon_disease.json")
            ),
            "orphanet_lexicon_path": str(
                dict_path
                / lexicons.get("orphanet_diseases", "2025_08_orphanet_diseases.json")
            ),
            "rare_disease_acronyms_path": str(
                dict_path
                / lexicons.get(
                    "rare_disease_acronyms", "2025_08_rare_disease_acronyms.json"
                )
            ),
            "umls_abbrev_path": str(
                dict_path
                / lexicons.get(
                    "umls_biological", "2025_08_umls_biological_abbreviations_v5.tsv"
                )
            ),
            "umls_clinical_path": str(
                dict_path
                / lexicons.get(
                    "umls_clinical", "2025_08_umls_clinical_abbreviations_v5.tsv"
                )
            ),
            "anca_lexicon_path": str(
                dict_path
                / lexicons.get("disease_lexicon_anca", "disease_lexicon_anca.json")
            ),
            "igan_lexicon_path": str(
                dict_path
                / lexicons.get("disease_lexicon_igan", "disease_lexicon_igan.json")
            ),
            "pah_lexicon_path": str(
                dict_path
                / lexicons.get("disease_lexicon_pah", "disease_lexicon_pah.json")
            ),
            "context_window": lexicon_cfg.get("context_window", 300),
        }

        self.generators = [
            AbbrevSyntaxCandidateGenerator(config={"run_id": self.run_id}),
            GlossaryTableCandidateGenerator(config={"run_id": self.run_id}),
            RegexCandidateGenerator(
                config={
                    "run_id": self.run_id,
                    "enabled_types": regex_cfg.get(
                        "enabled_types",
                        ["TRIAL_ID", "COMPOUND_ID", "DOI", "PMID", "PMCID"],
                    ),
                }
            ),
            LayoutCandidateGenerator(config={"run_id": self.run_id}),
            RegexLexiconGenerator(config=lexicon_gen_config),
        ]

        # Validation
        if not self.skip_validation:
            self.claude_client = ClaudeClient(
                api_key=api_key, model=self.model, config_path=self.config_path
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

        # Logger
        self.logger = ValidationLogger(log_dir=str(self.log_dir), run_id=self.run_id)

        # Normalization
        norm_cfg = self.config.get("normalization", {})
        term_mapper_cfg = norm_cfg.get("term_mapper", {})
        disambig_cfg = norm_cfg.get("disambiguator", {})

        self.term_mapper = TermMapper(
            config={
                "mapping_file_path": str(
                    dict_path
                    / lexicons.get(
                        "abbreviation_general", "2025_08_abbreviation_general.json"
                    )
                ),
                "enable_fuzzy_matching": term_mapper_cfg.get(
                    "enable_fuzzy_matching", False
                ),
                "fuzzy_cutoff": term_mapper_cfg.get("fuzzy_cutoff", 0.90),
                "fill_long_form_for_orphans": term_mapper_cfg.get(
                    "fill_long_form_for_orphans", False
                ),
            }
        )

        self.disambiguator = Disambiguator(
            config={
                "min_context_score": disambig_cfg.get("min_context_score", 2),
                "min_margin": disambig_cfg.get("min_margin", 1),
                "fill_long_form_for_orphans": disambig_cfg.get(
                    "fill_long_form_for_orphans", True
                ),
            }
        )

        # Disease detection components
        disease_cfg = self.config.get("disease_detection", {})
        self.enable_disease_detection = disease_cfg.get("enabled", True)

        if self.enable_disease_detection:
            self.disease_detector = DiseaseDetector(
                config={
                    "run_id": self.run_id,
                    "lexicon_base_path": str(dict_path),
                    "enable_general_lexicon": disease_cfg.get(
                        "enable_general_lexicon", True
                    ),
                    "enable_orphanet": disease_cfg.get("enable_orphanet", True),
                    "enable_scispacy": disease_cfg.get("enable_scispacy", True),
                    "context_window": disease_cfg.get("context_window", 300),
                }
            )
            self.disease_normalizer = DiseaseNormalizer()
        else:
            self.disease_detector = None
            self.disease_normalizer = None

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

    # =========================================================================
    # ENTITY CREATION HELPERS
    # =========================================================================

    def _create_entity_from_candidate(
        self,
        candidate: Candidate,
        status: ValidationStatus,
        confidence: float,
        reason: str,
        flags: List[str],
        raw_response: Dict,
        long_form_override: Optional[str] = None,
    ) -> ExtractedEntity:
        """Create ExtractedEntity from a Candidate (for auto-approve/reject)."""
        context = (candidate.context_text or "").strip()
        ctx_hash = hash_string(context) if context else "no_context"
        primary = EvidenceSpan(
            text=context,
            location=candidate.context_location,
            scope_ref=ctx_hash,
            start_char_offset=0,
            end_char_offset=len(context),
        )
        return ExtractedEntity(
            candidate_id=candidate.id,
            doc_id=candidate.doc_id,
            field_type=candidate.field_type,
            short_form=candidate.short_form.strip(),
            long_form=long_form_override
            or (candidate.long_form.strip() if candidate.long_form else None),
            primary_evidence=primary,
            supporting_evidence=[],
            status=status,
            confidence_score=confidence,
            rejection_reason=reason if status == ValidationStatus.REJECTED else None,
            validation_flags=flags,
            provenance=candidate.provenance,
            raw_llm_response=raw_response,
        )

    def _create_entity_from_search(
        self,
        doc_id: str,
        full_text: str,
        match: re.Match,
        long_form: Optional[str],
        field_type: FieldType,
        confidence: float,
        flags: List[str],
        rule_version: str,
        lexicon_source: str,
    ) -> ExtractedEntity:
        """Create ExtractedEntity from a text search match."""
        context_snippet = extract_context_snippet(full_text, match.start(), match.end())
        ctx_hash = hash_string(context_snippet)

        primary = EvidenceSpan(
            text=context_snippet,
            location=Coordinate(page_num=1),
            scope_ref=ctx_hash,
            start_char_offset=match.start() - max(0, match.start() - 100),
            end_char_offset=match.end() - max(0, match.start() - 100),
        )

        prov = ProvenanceMetadata(
            pipeline_version=PIPELINE_VERSION,
            run_id=self.run_id,
            doc_fingerprint=lexicon_source,
            generator_name=GeneratorType.LEXICON_MATCH,
            rule_version=rule_version,
            lexicon_source=f"orchestrator:{lexicon_source}",
        )

        return ExtractedEntity(
            candidate_id=uuid.uuid4(),
            doc_id=doc_id,
            field_type=field_type,
            short_form=match.group(),
            long_form=long_form,
            primary_evidence=primary,
            supporting_evidence=[],
            status=ValidationStatus.VALIDATED,
            confidence_score=confidence,
            rejection_reason=None,
            validation_flags=flags,
            provenance=prov,
            raw_llm_response={"auto": lexicon_source},
        )

    # =========================================================================
    # PIPELINE STAGES
    # =========================================================================

    def _parse_pdf(self, pdf_path: Path):
        """Stage 1: Parse PDF into DocumentGraph."""
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

        self._export_extracted_text(pdf_path, doc)
        return doc

    def _generate_candidates(self, doc) -> Tuple[List[Candidate], str]:
        """Stage 2: Generate and deduplicate candidates."""
        print("\n[2/4] Generating candidates...")
        start = time.time()

        all_candidates = []
        for gen in self.generators:
            candidates = gen.extract(doc)
            print(f"  {gen.generator_type.value}: {len(candidates)} candidates")
            all_candidates.extend(candidates)

        # Deduplicate
        seen = set()
        unique_candidates = []
        for c in all_candidates:
            key = (c.short_form.upper(), (c.long_form or "").lower())
            if key not in seen:
                seen.add(key)
                unique_candidates.append(c)

        # Build full text
        full_text = " ".join(
            block.text for block in doc.iter_linear_blocks() if block.text
        )

        print(f"  Total unique: {len(unique_candidates)}")
        print(f"  Time: {time.time() - start:.2f}s")

        return unique_candidates, full_text

    def _filter_candidates(
        self,
        candidates: List[Candidate],
        full_text: str,
    ) -> Tuple[List[Candidate], set, Counter, int]:
        """Filter and reduce candidates before validation."""
        # Corroborated SFs (found by non-lexicon generators)
        corroborated_sfs = {
            c.short_form.upper()
            for c in candidates
            if c.generator_type != GeneratorType.LEXICON_MATCH
        }

        # Count SF occurrences
        abbrev_pattern = r"\b[A-Za-z][A-Za-z0-9]{1,11}\b"
        all_tokens = re.findall(abbrev_pattern, full_text)
        word_counts = Counter(
            t.upper() for t in all_tokens if any(c.isupper() for c in t)
        )

        # Keep lexicon matches if corroborated or frequent
        def should_keep(c: Candidate) -> bool:
            if c.generator_type != GeneratorType.LEXICON_MATCH:
                return True
            sf_upper = c.short_form.upper()
            return sf_upper in corroborated_sfs or word_counts.get(sf_upper, 0) >= 2

        needs_validation = [c for c in candidates if should_keep(c)]

        # Reduce lexicon matches (dedup by SF, filter by form)
        allowed_2letter = self.heuristics.allowed_2letter_sfs
        allowed_mixed = self.heuristics.allowed_mixed_case

        lexicon_by_sf: Dict[str, List[Candidate]] = {}
        non_lexicon = []
        sf_form_rejected = 0

        for c in needs_validation:
            if c.generator_type == GeneratorType.LEXICON_MATCH:
                ctx = c.context_text or ""
                if not is_valid_sf_form(
                    c.short_form, ctx, allowed_2letter, allowed_mixed
                ):
                    sf_form_rejected += 1
                    continue
                sf_upper = c.short_form.upper()
                lexicon_by_sf.setdefault(sf_upper, []).append(c)
            else:
                non_lexicon.append(c)

        # Keep top 2 LFs per SF (cache lowercase for performance)
        full_text_lower = full_text.lower()
        deduped_lexicon = []
        for sf, cands in lexicon_by_sf.items():
            if len(cands) <= 2:
                deduped_lexicon.extend(cands)
            else:
                # Use heapq.nlargest instead of full sort for O(n) vs O(n log n)
                top_2 = heapq.nlargest(
                    2,
                    cands,
                    key=lambda c: score_lf_quality(c, full_text, full_text_lower),
                )
                deduped_lexicon.extend(top_2)

        lexicon_before = sum(len(v) for v in lexicon_by_sf.values()) + sf_form_rejected
        print(
            f"  LEXICON_MATCH reduction: {lexicon_before} -> {len(deduped_lexicon)} "
            f"(dedup: {lexicon_before - sf_form_rejected - len(deduped_lexicon)}, form filter: {sf_form_rejected})"
        )

        return (
            non_lexicon + deduped_lexicon,
            corroborated_sfs,
            word_counts,
            len(candidates) - len(non_lexicon + deduped_lexicon),
        )

    def _apply_heuristics(
        self,
        candidates: List[Candidate],
        counters: HeuristicsCounters,
    ) -> Tuple[List[Tuple[Candidate, ExtractedEntity]], List[Candidate]]:
        """Apply auto-approve/reject heuristics. Returns (auto_results, llm_candidates)."""
        auto_results = []
        llm_candidates = []

        for c in candidates:
            sf_upper = c.short_form.upper()
            ctx = c.context_text or ""

            # Auto-reject blacklisted
            if sf_upper in self.heuristics.sf_blacklist:
                entity = self._create_entity_from_candidate(
                    c,
                    ValidationStatus.REJECTED,
                    0.95,
                    "Blacklisted: not a valid abbreviation",
                    ["auto_rejected_blacklist"],
                    {"auto": "blacklist"},
                )
                auto_results.append((c, entity))
                counters.blacklisted_fp_count += 1
                continue

            # Auto-reject context mismatch
            if not check_context_match(c.short_form, ctx, self.heuristics):
                entity = self._create_entity_from_candidate(
                    c,
                    ValidationStatus.REJECTED,
                    0.90,
                    "Rejected: ambiguous SF without required context",
                    ["auto_rejected_context"],
                    {"auto": "context_mismatch"},
                )
                auto_results.append((c, entity))
                counters.context_rejected += 1
                continue

            # Auto-reject trial IDs
            if check_trial_id(c.short_form, self.heuristics):
                entity = self._create_entity_from_candidate(
                    c,
                    ValidationStatus.REJECTED,
                    0.90,
                    "Excluded: trial identifier (NCT number)",
                    ["auto_rejected_trial_id"],
                    {"auto": "trial_id_excluded"},
                )
                auto_results.append((c, entity))
                counters.trial_id_excluded += 1
                continue

            # Auto-approve stats with numeric evidence
            if sf_upper in self.heuristics.stats_abbrevs and has_numeric_evidence(
                ctx, c.short_form
            ):
                canonical_lf = self.heuristics.stats_abbrevs.get(sf_upper)
                entity = self._create_entity_from_candidate(
                    c,
                    ValidationStatus.VALIDATED,
                    0.90,
                    "Stats abbreviation with numeric evidence",
                    ["auto_approved_stats"],
                    {"auto": "stats_whitelist"},
                    long_form_override=canonical_lf,
                )
                auto_results.append((c, entity))
                counters.recovered_by_stats_whitelist += 1
                continue

            # Auto-approve country codes
            if sf_upper in self.heuristics.country_abbrevs:
                canonical_lf = self.heuristics.country_abbrevs.get(sf_upper)
                entity = self._create_entity_from_candidate(
                    c,
                    ValidationStatus.VALIDATED,
                    0.90,
                    "Country code abbreviation",
                    ["auto_approved_country"],
                    {"auto": "country_code"},
                    long_form_override=canonical_lf,
                )
                auto_results.append((c, entity))
                counters.recovered_by_country_code += 1
                continue

            # Auto-reject common words
            if sf_upper in self.heuristics.common_words:
                entity = self._create_entity_from_candidate(
                    c,
                    ValidationStatus.REJECTED,
                    0.95,
                    "Common English word, not an abbreviation",
                    ["auto_rejected"],
                    {"auto": "rejected_common_word"},
                )
                auto_results.append((c, entity))
                counters.common_word_rejected += 1
                continue

            llm_candidates.append(c)

        return auto_results, llm_candidates

    def _validate_with_llm(
        self,
        llm_candidates: List[Candidate],
        batch_delay_ms: float,
    ) -> List[ExtractedEntity]:
        """Validate candidates using LLM."""
        assert self.llm_engine is not None, (
            "LLM engine must be initialized for validation"
        )
        results = []

        # Split by generator type
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

        # Haiku screening (optional)
        if self.enable_haiku_screening and lexicon_candidates:
            print(
                f"\n  Haiku screening {len(lexicon_candidates)} lexicon candidates..."
            )
            haiku_start = time.time()
            needs_review, haiku_rejected = self.llm_engine.fast_reject_batch(
                lexicon_candidates,
                haiku_model="claude-3-5-haiku-20241022",
                batch_size=20,
            )
            haiku_time = time.time() - haiku_start
            print(
                f"    Haiku rejected: {len(haiku_rejected)}, Needs review: {len(needs_review)}, Time: {haiku_time:.2f}s"
            )

            for entity in haiku_rejected:
                candidate = next(
                    (
                        c
                        for c in lexicon_candidates
                        if str(c.id) == str(entity.candidate_id)
                    ),
                    None,
                )
                if candidate:
                    self.logger.log_validation(
                        candidate,
                        entity,
                        entity.raw_llm_response,
                        haiku_time * 1000 / len(lexicon_candidates),
                    )
                results.append(entity)

            lexicon_candidates = needs_review

        # Batch validate explicit pairs
        if explicit_candidates:
            try:
                val_start = time.time()
                batch_results = self.llm_engine.verify_candidates_batch(
                    explicit_candidates, batch_size=10
                )
                elapsed_ms = (time.time() - val_start) * 1000

                for candidate, entity in zip(explicit_candidates, batch_results):
                    self.logger.log_validation(
                        candidate,
                        entity,
                        entity.raw_llm_response,
                        elapsed_ms / len(explicit_candidates),
                    )
                    results.append(entity)
            except Exception as e:
                print(f"  [WARN] Batch error, falling back to individual: {e}")
                for candidate in explicit_candidates:
                    try:
                        entity = self.llm_engine.verify_candidate(candidate)
                        self.logger.log_validation(
                            candidate, entity, entity.raw_llm_response, 0
                        )
                        results.append(entity)
                    except Exception as e2:
                        self.logger.log_error(candidate, str(e2))

        # Batch validation for lexicon matches (10-15x faster than individual calls)
        if lexicon_candidates:
            try:
                val_start = time.time()
                batch_results = self.llm_engine.verify_candidates_batch(
                    lexicon_candidates, batch_size=15
                )
                elapsed_ms = (time.time() - val_start) * 1000

                for candidate, entity in zip(lexicon_candidates, batch_results):
                    self.logger.log_validation(
                        candidate,
                        entity,
                        entity.raw_llm_response,
                        elapsed_ms / len(lexicon_candidates),
                    )
                    results.append(entity)
            except Exception as e:
                print(f"  [WARN] Batch error, falling back to individual: {e}")
                for candidate in lexicon_candidates:
                    try:
                        entity = self.llm_engine.verify_candidate(candidate)
                        self.logger.log_validation(
                            candidate, entity, entity.raw_llm_response, 0
                        )
                        results.append(entity)
                    except Exception as e2:
                        self.logger.log_error(candidate, str(e2))

        return results

    def _search_missing_abbreviations(
        self,
        doc_id: str,
        full_text: str,
        found_sfs: set,
        counters: HeuristicsCounters,
    ) -> List[ExtractedEntity]:
        """Search for abbreviations missed by generators (PASO C/D)."""
        results = []

        # PASO C: Hyphenated abbreviations
        for hyph_sf, hyph_lf in self.heuristics.hyphenated_abbrevs.items():
            if hyph_sf.upper() in found_sfs:
                continue
            pattern = rf"\b{re.escape(hyph_sf)}\b"
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                entity = self._create_entity_from_search(
                    doc_id,
                    full_text,
                    match,
                    hyph_lf,
                    FieldType.DEFINITION_PAIR,
                    0.85,
                    ["auto_approved_hyphenated"],
                    "hyphenated_detector:v1.0",
                    "hyphenated",
                )
                results.append(entity)
                found_sfs.add(hyph_sf.upper())
                counters.recovered_by_hyphen += 1

        if counters.recovered_by_hyphen > 0:
            print(f"  Hyphenated detected (PASO C): {counters.recovered_by_hyphen}")

        # Direct search abbreviations
        for direct_sf, direct_lf in self.heuristics.direct_search_abbrevs.items():
            if direct_sf.upper() in found_sfs:
                continue
            pattern = (
                re.escape(direct_sf)
                if " " in direct_sf
                else rf"\b{re.escape(direct_sf)}\b"
            )
            match = re.search(pattern, full_text)
            if match:
                entity = self._create_entity_from_search(
                    doc_id,
                    full_text,
                    match,
                    direct_lf,
                    FieldType.DEFINITION_PAIR,
                    0.85,
                    ["auto_approved_direct_search"],
                    "direct_search:v1.0",
                    "direct_search",
                )
                results.append(entity)
                found_sfs.add(direct_sf.upper())
                counters.recovered_by_direct_search += 1

        if counters.recovered_by_direct_search > 0:
            print(f"  Direct search detected: {counters.recovered_by_direct_search}")

        return results

    def _extract_sf_only_with_llm(
        self,
        doc_id: str,
        full_text: str,
        found_sfs: set,
        counters: HeuristicsCounters,
    ) -> List[ExtractedEntity]:
        """PASO D: Extract SF-only using LLM."""
        print("\n  Running LLM SF-only extractor (PASO D)...")
        if self.claude_client is None:
            return []
        results = []

        # Sample chunks
        chunk_size = 3000
        text_chunks = [
            full_text[i : i + chunk_size]
            for i in range(0, min(len(full_text), 15000), chunk_size)
        ]

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

        for chunk in text_chunks:
            try:
                prompt = sf_extraction_prompt.format(
                    already_found=", ".join(sorted(found_sfs)[:50]), text=chunk
                )
                response = self.claude_client.complete_json_any(
                    system_prompt="You are an abbreviation extraction assistant. Return only valid JSON arrays.",
                    user_prompt=prompt,
                    model=self.model,
                    temperature=0.0,
                    max_tokens=500,
                    top_p=1.0,
                )
                if isinstance(response, list):
                    for sf in response:
                        if isinstance(sf, str) and sf.strip():
                            llm_sf_candidates.add(sf.strip())
                elif isinstance(response, dict) and "abbreviations" in response:
                    for sf in response.get("abbreviations", []):
                        if isinstance(sf, str) and sf.strip():
                            llm_sf_candidates.add(sf.strip())
            except Exception:
                llm_errors += 1

        print(
            f"    LLM chunks: {len(text_chunks)}, errors: {llm_errors}, candidates: {len(llm_sf_candidates)}"
        )

        # Validate and add LLM-found SFs
        blacklist = self.heuristics.common_words | self.heuristics.sf_blacklist

        for sf_candidate in llm_sf_candidates:
            sf_upper = sf_candidate.upper()

            if sf_upper in found_sfs or sf_upper in blacklist:
                continue

            # Verify SF exists in text
            if sf_candidate not in full_text:
                match = re.search(
                    rf"\b{re.escape(sf_candidate)}\b", full_text, re.IGNORECASE
                )
                if not match:
                    continue
                sf_candidate = match.group()

            # Basic form validation
            if (
                len(sf_candidate) < 2
                or len(sf_candidate) > 15
                or not any(c.isupper() for c in sf_candidate)
            ):
                continue

            # Find context
            idx = full_text.find(sf_candidate)
            if idx == -1:
                idx = full_text.lower().find(sf_candidate.lower())

            context_snippet = extract_context_snippet(
                full_text, idx, idx + len(sf_candidate)
            )
            ctx_hash = hash_string(context_snippet)

            primary = EvidenceSpan(
                text=context_snippet,
                location=Coordinate(page_num=1),
                scope_ref=ctx_hash,
                start_char_offset=max(0, idx - (idx - 100 if idx > 100 else 0)),
                end_char_offset=max(0, idx - (idx - 100 if idx > 100 else 0))
                + len(sf_candidate),
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
                doc_id=doc_id,
                field_type=FieldType.SHORT_FORM_ONLY,
                short_form=sf_candidate,
                long_form=None,
                primary_evidence=primary,
                supporting_evidence=[],
                status=ValidationStatus.VALIDATED,
                confidence_score=0.75,
                rejection_reason=None,
                validation_flags=["llm_sf_extracted"],
                provenance=prov,
                raw_llm_response={"auto": "llm_sf_extractor"},
            )
            results.append(entity)
            found_sfs.add(sf_upper)
            counters.recovered_by_llm_sf_only += 1

        print(f"  LLM SF-only extracted (PASO D): {counters.recovered_by_llm_sf_only}")
        return results

    def _normalize_results(
        self, results: List[ExtractedEntity], full_text: str
    ) -> List[ExtractedEntity]:
        """Stage 3.5: Normalize and disambiguate results.

        Args:
            results: List of entities to normalize
            full_text: Pre-built full document text (avoids rebuilding)
        """
        print("\n[3.5/4] Normalizing & disambiguating...")

        # Normalize
        normalized_count = 0
        for i, entity in enumerate(results):
            normalized = self.term_mapper.normalize(entity)
            if "normalized" in (normalized.validation_flags or []):
                normalized_count += 1
            results[i] = normalized

        # Disambiguate
        results = self.disambiguator.resolve(results, full_text)
        disambiguated_count = sum(
            1 for r in results if "disambiguated" in (r.validation_flags or [])
        )

        print(f"  Normalized: {normalized_count}")
        print(f"  Disambiguated: {disambiguated_count}")

        return results

    # =========================================================================
    # MAIN PROCESS METHOD
    # =========================================================================

    def process_pdf(
        self, pdf_path: str | Path, batch_delay_ms: Optional[float] = None
    ) -> List[ExtractedEntity]:
        """Process a single PDF through the full pipeline."""
        delay: float = (
            batch_delay_ms
            if batch_delay_ms is not None
            else (self.batch_delay_ms or 100.0)
        )

        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        print(f"\n{'=' * 60}")
        print(f"Processing: {pdf_path_obj.name}")
        print(f"{'=' * 60}")

        # Stage 1: Parse
        doc = self._parse_pdf(pdf_path_obj)

        # Stage 2: Generate candidates
        unique_candidates, full_text = self._generate_candidates(doc)

        # Early exit if validation disabled
        if self.skip_validation:
            print("\n[3/4] Validation SKIPPED")
            self._export_results(pdf_path_obj, [], unique_candidates)
            return []

        # Filter candidates
        needs_validation, corroborated_sfs, word_counts, filtered_count = (
            self._filter_candidates(unique_candidates, full_text)
        )

        # Apply heuristics
        counters = HeuristicsCounters()
        auto_results, llm_candidates = self._apply_heuristics(
            needs_validation, counters
        )

        # Print stats
        frequent_sfs = sum(
            1
            for c in unique_candidates
            if c.generator_type == GeneratorType.LEXICON_MATCH
            and c.short_form.upper() not in corroborated_sfs
            and word_counts.get(c.short_form.upper(), 0) >= 2
        )

        print("\n[3/4] Validating candidates with Claude...")
        print(f"  Corroborated SFs: {len(corroborated_sfs)}")
        print(f"  Frequent SFs (2+): {frequent_sfs}")
        print(f"  Filtered (lexicon-only, rare): {filtered_count}")
        print(f"  Auto-approved stats: {counters.recovered_by_stats_whitelist}")
        print(f"  Auto-approved country: {counters.recovered_by_country_code}")
        print(f"  Auto-rejected blacklist: {counters.blacklisted_fp_count}")
        print(f"  Auto-rejected context: {counters.context_rejected}")
        print(f"  Auto-rejected trial IDs: {counters.trial_id_excluded}")
        print(f"  Auto-rejected common words: {counters.common_word_rejected}")
        print(f"  Candidates for LLM: {len(llm_candidates)}")

        start = time.time()

        # Collect results
        results: List[ExtractedEntity] = []

        # Log auto results
        for candidate, entity in auto_results:
            self.logger.log_validation(candidate, entity, entity.raw_llm_response, 0)
            results.append(entity)

        # LLM validation
        llm_results = self._validate_with_llm(llm_candidates, delay)
        results.extend(llm_results)

        print(f"  Time: {time.time() - start:.2f}s")

        # Search for missing abbreviations
        doc_id = str(pdf_path_obj.stem)
        found_sfs = {
            r.short_form.upper()
            for r in results
            if r.status == ValidationStatus.VALIDATED
        }

        search_results = self._search_missing_abbreviations(
            doc_id, full_text, found_sfs, counters
        )
        results.extend(search_results)

        # LLM SF-only extraction
        sf_only_results = self._extract_sf_only_with_llm(
            doc_id, full_text, found_sfs, counters
        )
        results.extend(sf_only_results)

        # Normalize (pass full_text to avoid rebuilding)
        results = self._normalize_results(results, full_text)

        # Disease detection (parallel pipeline)
        disease_results: List[ExtractedDisease] = []
        if self.enable_disease_detection and self.disease_detector is not None:
            disease_results = self._process_diseases(doc, pdf_path_obj)

        # Stage 4: Summary & Export
        print("\n[4/4] Writing summary...")
        self.logger.write_summary()
        self.logger.print_summary()
        counters.log_summary()

        self._export_results(pdf_path_obj, results, unique_candidates, counters)

        # Export disease results
        if disease_results:
            self._export_disease_results(pdf_path_obj, disease_results)

        # Print validated
        validated = [r for r in results if r.status == ValidationStatus.VALIDATED]
        if validated:
            print(f"\nValidated abbreviations ({len(validated)}):")
            for v in validated:
                print(f"  * {v.short_form} -> {v.long_form or '(no expansion)'}")

        # Print validated diseases
        validated_diseases = [
            d for d in disease_results if d.status == ValidationStatus.VALIDATED
        ]
        if validated_diseases:
            print(f"\nValidated diseases ({len(validated_diseases)}):")
            for d in validated_diseases[:10]:  # Show first 10
                codes = []
                if d.icd10_code:
                    codes.append(f"ICD-10:{d.icd10_code}")
                if d.orpha_code:
                    codes.append(f"ORPHA:{d.orpha_code}")
                code_str = f" [{', '.join(codes)}]" if codes else ""
                print(f"  * {d.preferred_label}{code_str}")
            if len(validated_diseases) > 10:
                print(f"  ... and {len(validated_diseases) - 10} more")

        return results

    # =========================================================================
    # DISEASE DETECTION METHODS
    # =========================================================================

    def _process_diseases(
        self, doc, pdf_path: Path
    ) -> List[ExtractedDisease]:
        """
        Process document for disease mentions.

        Returns validated disease entities.
        """
        if self.disease_detector is None:
            return []

        print("\n[3b/4] Detecting disease mentions...")
        start = time.time()

        # Generate disease candidates
        candidates = self.disease_detector.extract(doc)
        print(f"  Disease candidates: {len(candidates)}")

        if not candidates:
            print(f"  Time: {time.time() - start:.2f}s")
            return []

        # Auto-validate based on source (specialized lexicons are high trust)
        results: List[ExtractedDisease] = []
        for candidate in candidates:
            # Specialized lexicons (PAH, ANCA, IgAN) are auto-validated
            is_specialized = "specialized" in candidate.generator_type.value

            entity = self._create_disease_entity(
                candidate=candidate,
                status=ValidationStatus.VALIDATED,
                confidence=candidate.initial_confidence + candidate.confidence_boost,
                flags=["auto_validated_lexicon"] if is_specialized else ["lexicon_match"],
            )
            results.append(entity)

        # Normalize diseases
        if self.disease_normalizer is not None:
            results = self.disease_normalizer.normalize_batch(results)

        validated_count = len(
            [r for r in results if r.status == ValidationStatus.VALIDATED]
        )
        print(f"  Validated diseases: {validated_count}")
        print(f"  Time: {time.time() - start:.2f}s")

        return results

    def _create_disease_entity(
        self,
        candidate: DiseaseCandidate,
        status: ValidationStatus,
        confidence: float,
        flags: List[str],
    ) -> ExtractedDisease:
        """Create ExtractedDisease from a DiseaseCandidate."""
        context = (candidate.context_text or "").strip()
        ctx_hash = hash_string(context) if context else "no_context"

        primary = EvidenceSpan(
            text=context,
            location=candidate.context_location,
            scope_ref=ctx_hash,
            start_char_offset=0,
            end_char_offset=len(context),
        )

        # Extract primary codes from identifiers
        icd10 = None
        icd11 = None
        snomed = None
        mondo = None
        orpha = None
        umls = None
        mesh = None

        for ident in candidate.identifiers:
            if ident.system in ("ICD-10", "ICD-10-CM") and not icd10:
                icd10 = ident.code
            elif ident.system == "ICD-11" and not icd11:
                icd11 = ident.code
            elif ident.system in ("SNOMED-CT", "SNOMED") and not snomed:
                snomed = ident.code
            elif ident.system == "MONDO" and not mondo:
                mondo = ident.code
            elif ident.system in ("ORPHA", "Orphanet") and not orpha:
                orpha = ident.code
            elif ident.system == "UMLS" and not umls:
                umls = ident.code
            elif ident.system == "MeSH" and not mesh:
                mesh = ident.code

        return ExtractedDisease(
            candidate_id=candidate.id,
            doc_id=candidate.doc_id,
            matched_text=candidate.matched_text,
            preferred_label=candidate.preferred_label,
            abbreviation=candidate.abbreviation,
            identifiers=candidate.identifiers,
            icd10_code=icd10,
            icd11_code=icd11,
            snomed_code=snomed,
            mondo_id=mondo,
            orpha_code=orpha,
            umls_cui=umls,
            mesh_id=mesh,
            primary_evidence=primary,
            supporting_evidence=[],
            status=status,
            confidence_score=min(1.0, confidence),
            validation_flags=flags,
            is_rare_disease=candidate.is_rare_disease,
            disease_category=candidate.disease_category,
            provenance=candidate.provenance,
        )

    def _export_disease_results(
        self, pdf_path: Path, results: List[ExtractedDisease]
    ) -> None:
        """Export disease detection results to separate JSON file."""
        out_dir = self.output_dir or pdf_path.parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        validated = [r for r in results if r.status == ValidationStatus.VALIDATED]
        rejected = [r for r in results if r.status == ValidationStatus.REJECTED]
        ambiguous = [r for r in results if r.status == ValidationStatus.AMBIGUOUS]

        # Build export entries
        disease_entries: List[DiseaseExportEntry] = []
        for entity in validated:
            codes = {
                "icd10": entity.icd10_code,
                "icd11": entity.icd11_code,
                "snomed": entity.snomed_code,
                "mondo": entity.mondo_id,
                "orpha": entity.orpha_code,
                "umls": entity.umls_cui,
                "mesh": entity.mesh_id,
            }

            all_identifiers = [
                {"system": i.system, "code": i.code, "display": i.display}
                for i in entity.identifiers
            ]

            entry = DiseaseExportEntry(
                matched_text=entity.matched_text,
                preferred_label=entity.preferred_label,
                abbreviation=entity.abbreviation,
                confidence=entity.confidence_score,
                is_rare_disease=entity.is_rare_disease,
                category=entity.disease_category,
                codes=codes,
                all_identifiers=all_identifiers,
                context=entity.primary_evidence.text if entity.primary_evidence else None,
                page=entity.primary_evidence.location.page_num
                if entity.primary_evidence
                else None,
                lexicon_source=entity.provenance.lexicon_source
                if entity.provenance
                else None,
                validation_flags=entity.validation_flags,
            )
            disease_entries.append(entry)

        # Build export document
        export_doc = DiseaseExportDocument(
            run_id=self.run_id,
            timestamp=datetime.now().isoformat(),
            document=pdf_path.name,
            document_path=str(pdf_path.absolute()),
            pipeline_version=PIPELINE_VERSION,
            total_candidates=len(results),
            total_validated=len(validated),
            total_rejected=len(rejected),
            total_ambiguous=len(ambiguous),
            diseases=disease_entries,
        )

        # Write to file
        out_file = out_dir / f"diseases_{pdf_path.stem}_{timestamp}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(export_doc.model_dump_json(indent=2))

        print(f"  Disease export: {out_file.name}")

    # =========================================================================
    # EXPORT METHODS
    # =========================================================================

    def _export_extracted_text(self, pdf_path: Path, doc) -> None:
        """Export extracted text to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_path = pdf_path.parent / f"{pdf_path.stem}_{timestamp}.txt"

        text_lines = []
        current_page = None

        for block in doc.iter_linear_blocks(skip_header_footer=False):
            if block.page_num != current_page:
                if current_page is not None:
                    text_lines.append("")
                text_lines.append(f"--- Page {block.page_num} ---")
                text_lines.append("")
                current_page = block.page_num

            text = (block.text or "").strip()
            if text:
                text_lines.append(text)

        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(text_lines))
            print(f"  Extracted text: {txt_path.name}")
        except Exception as e:
            print(f"  [WARN] Failed to export text: {e}")

    def _export_results(
        self,
        pdf_path: Path,
        results: List[ExtractedEntity],
        candidates: List[Candidate],
        counters: Optional[HeuristicsCounters] = None,
    ) -> None:
        """Export results to JSON."""
        out_dir = self.output_dir or pdf_path.parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        export_data = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "document": pdf_path.name,
            "document_path": str(pdf_path),
            "total_candidates": len(candidates),
            "total_validated": sum(
                1 for r in results if r.status == ValidationStatus.VALIDATED
            ),
            "total_rejected": sum(
                1 for r in results if r.status == ValidationStatus.REJECTED
            ),
            "total_ambiguous": sum(
                1 for r in results if r.status == ValidationStatus.AMBIGUOUS
            ),
            "heuristics_counters": counters.to_dict() if counters else None,
            "abbreviations": [],
        }

        for entity in results:
            if entity.status == ValidationStatus.VALIDATED:
                lexicon_ids = None
                if entity.provenance.lexicon_ids:
                    lexicon_ids = [
                        {"source": lid.source, "id": lid.id}
                        for lid in entity.provenance.lexicon_ids
                    ]

                export_data["abbreviations"].append(
                    {
                        "short_form": entity.short_form,
                        "long_form": entity.long_form,
                        "confidence": entity.confidence_score,
                        "field_type": entity.field_type.value,
                        "page": entity.primary_evidence.location.page_num
                        if entity.primary_evidence
                        else None,
                        "context_text": entity.primary_evidence.text
                        if entity.primary_evidence
                        else None,
                        "lexicon_source": entity.provenance.lexicon_source,
                        "lexicon_ids": lexicon_ids,
                    }
                )

        out_file = out_dir / f"abbreviations_{pdf_path.stem}_{timestamp}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"\n  Exported: {out_file}")
        run_analysis(export_data, self.gold_json)

    def process_folder(
        self, folder_path: Optional[str] = None, batch_delay_ms: float = 100
    ) -> Dict[str, List[ExtractedEntity]]:
        """Process all PDFs in a folder."""
        folder = Path(folder_path or self.pdf_dir)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")

        pdf_files = sorted(folder.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in {folder}")
            return {}

        print(f"\n{'#' * 60}")
        print(f"BATCH PROCESSING: {len(pdf_files)} PDFs")
        print(f"Folder: {folder}")
        print(f"{'#' * 60}")

        all_results = {}

        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"\n[{i}/{len(pdf_files)}] {pdf_path.name}")
            try:
                all_results[pdf_path.name] = self.process_pdf(
                    str(pdf_path), batch_delay_ms=batch_delay_ms
                )
            except Exception as e:
                print(f"  [WARN] ERROR: {e}")
                all_results[pdf_path.name] = []

        print(f"\n{'#' * 60}")
        print("BATCH COMPLETE")
        print(f"{'#' * 60}")
        print(f"PDFs processed: {len(all_results)}")

        total_validated = sum(
            sum(1 for r in results if r.status == ValidationStatus.VALIDATED)
            for results in all_results.values()
        )
        print(f"Total validated abbreviations: {total_validated}")
        print(f"Log directory: {self.log_dir}")

        return all_results


def main():
    """Process all PDFs in the default folder."""
    Orchestrator().process_folder()


if __name__ == "__main__":
    main()

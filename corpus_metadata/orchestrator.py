# corpus_metadata/orchestrator.py
"""
Orchestrator v0.8: PDF -> Parse -> Generate -> Validate -> Log

All configuration is loaded from config.yaml - no hardcoded parameters.

Usage:
    python orchestrator.py

Requires:
    - ANTHROPIC_API_KEY in .env file (or set in config.yaml)
    - pip install anthropic pyyaml python-dotenv

REFACTORED: StageTimer and warning suppression extracted to orchestrator_utils.py
"""

from __future__ import annotations

# =============================================================================
# SILENCE WARNINGS FIRST (must be before library imports)
# =============================================================================
from dotenv import load_dotenv

load_dotenv()  # Load .env file from current directory or parent

from orchestrator_utils import setup_warning_suppression, StageTimer
setup_warning_suppression()
# =============================================================================

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

from Z_utils.Z05_path_utils import get_base_path

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
# =============================================================================

# Ensure imports work
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Configure logging first (before slow imports)
from A_core.A00_logging import (
    configure_logging,
    get_logger,
    INFO,
)

# Initialize module logger
logger = get_logger(__name__)

# Print startup message before slow imports (scispacy takes ~30s to load)
print("Starting pipeline... (loading NLP models, this may take a moment)")

from A_core.A01_domain_models import (
    Candidate,
    ExtractedEntity,
    ValidationStatus,
)
from A_core.A03_provenance import generate_run_id
from A_core.A04_heuristics_config import (
    HeuristicsConfig,
    HeuristicsCounters,
)
from A_core.A16_pipeline_metrics import PipelineMetrics
from A_core.A05_disease_models import ExtractedDisease
from A_core.A06_drug_models import ExtractedDrug
from A_core.A19_gene_models import ExtractedGene
from A_core.A09_pharma_models import ExtractedPharma
from A_core.A10_author_models import ExtractedAuthor
from A_core.A11_citation_models import ExtractedCitation
from A_core.A07_feasibility_models import FeasibilityCandidate, NERCandidate
from A_core.A08_document_metadata_models import DocumentMetadata
from A_core.A17_care_pathway_models import CarePathway
from A_core.A18_recommendation_models import RecommendationSet

# Import refactored modules
from H_pipeline.H01_component_factory import ComponentFactory
from H_pipeline.H02_abbreviation_pipeline import AbbreviationPipeline
from H_pipeline.H03_visual_integration import VisualPipelineIntegration
from I_extraction.I01_entity_processors import EntityProcessor
from I_extraction.I02_feasibility_processor import FeasibilityProcessor
from J_export.J01_export_handlers import ExportManager

# Clinical intelligence extractors
from C_generators.C17_flowchart_graph_extractor import FlowchartGraphExtractor
from C_generators.C12_guideline_recommendation_extractor import GuidelineRecommendationExtractor

PIPELINE_VERSION = "0.8"


class Orchestrator:
    """
    Main pipeline orchestrator for clinical document metadata extraction.

    This class coordinates the entire extraction pipeline, processing PDF documents
    through multiple stages to extract structured metadata including abbreviations,
    drugs, diseases, feasibility information, and more.

    Pipeline Stages:
        1. Parse PDF -> DocumentGraph (structure extraction)
        2. Generate candidates (syntax patterns + lexicon matching)
        3. Validate candidates with Claude LLM
        4. Normalize and deduplicate entities
        5. Detect domain-specific entities (drugs, diseases, etc.)
        6. Export results to JSON

    Attributes:
        config: Configuration dictionary loaded from YAML.
        run_id: Unique identifier for this pipeline run.
        log_dir: Directory for validation logs.
        pdf_dir: Directory containing input PDFs.
        model: Claude model to use for validation.

    Example:
        >>> orchestrator = Orchestrator(config_path="config.yaml")
        >>> orchestrator.process_folder()
    """

    DEFAULT_CONFIG = (
        "/Users/frederictetard/Projects/ese/corpus_metadata/G_config/config.yaml"
    )

    def __init__(
        self,
        log_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        config_path: Optional[str] = None,
        run_id: Optional[str] = None,
        gold_json: Optional[str] = None,
        heuristics_config: Optional[HeuristicsConfig] = None,
    ) -> None:
        """
        Initialize the orchestrator with configuration.

        Args:
            log_dir: Directory for validation logs. Defaults to config value.
            output_dir: Directory for output files. Defaults to PDF subdirectories.
            model: Claude model for validation. Defaults to config value.
            api_key: Anthropic API key. Defaults to environment variable.
            config_path: Path to config.yaml file.
            run_id: Unique run identifier. Auto-generated if not provided.
            gold_json: Path to gold standard JSON for evaluation.
            heuristics_config: Pre-loaded heuristics configuration.
        """
        self._logger = get_logger(__name__)

        self.config_path = config_path or self.DEFAULT_CONFIG
        self.config = self._load_config(self.config_path)

        # Load extraction pipeline settings directly from config.yaml
        self._load_extraction_settings()

        # Extract paths from config (uses CORPUS_BASE_PATH env var or auto-detects)
        paths = self.config.get("paths", {})
        base_path = get_base_path(self.config)

        self.log_dir = Path(
            log_dir or Path(base_path) / paths.get("logs", "corpus_log")
        )
        self.output_dir = Path(output_dir) if output_dir else None
        self.pdf_dir = Path(base_path) / paths.get("pdf_input", "Pdfs")
        self.gold_json = gold_json or str(
            Path(base_path) / paths.get("gold_json", "gold_data/papers_gold_v2.json")
        )

        # API settings
        api_cfg = self.config.get("api", {}).get("claude", {})
        val_cfg = api_cfg.get("validation", {})

        self.model = model or val_cfg.get("model", "claude-sonnet-4-20250514")
        self.batch_delay_ms = api_cfg.get("batch_delay_ms", 100)

        self.run_id = run_id or generate_run_id("RUN")
        self.heuristics = heuristics_config or HeuristicsConfig.from_yaml(
            self.config_path
        )

        # Initialize components using factory
        self._init_components(api_key)

        # Log initialization (both console and file)
        self._logger.info(f"Orchestrator v{PIPELINE_VERSION} initialized")
        self._logger.info(f"Run ID: {self.run_id}")
        self._logger.info(f"Config: {self.config_path}")
        self._logger.info(f"Model: {self.model}")

        # Console output for user visibility
        print(f"\nOrchestrator v{PIPELINE_VERSION} initialized")
        print(f"  Run ID: {self.run_id}")
        print(f"  Config: {self.config_path}")
        print(f"  Model:  {self.model}")
        print(f"  Logs:   {self.log_dir}")
        self._print_extraction_config()

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

    def _load_extraction_settings(self) -> None:
        """Load extraction pipeline settings from config.yaml."""
        pipeline = self.config.get("extraction_pipeline", {})
        preset = pipeline.get("preset")
        extractors = pipeline.get("extractors", {})
        options = pipeline.get("options", {})

        # Define presets (override individual flags if preset is set)
        PRESETS = {
            "drugs_only": {
                "drugs": True, "diseases": False, "genes": False, "abbreviations": False,
                "feasibility": False, "pharma_companies": False, "authors": False,
                "citations": False, "document_metadata": False, "tables": False,
            },
            "diseases_only": {
                "drugs": False, "diseases": True, "genes": False, "abbreviations": False,
                "feasibility": False, "pharma_companies": False, "authors": False,
                "citations": False, "document_metadata": False, "tables": False,
            },
            "abbreviations_only": {
                "drugs": False, "diseases": False, "genes": False, "abbreviations": True,
                "feasibility": False, "pharma_companies": False, "authors": False,
                "citations": False, "document_metadata": False, "tables": False,
            },
            "feasibility_only": {
                "drugs": False, "diseases": False, "genes": False, "abbreviations": False,
                "feasibility": True, "pharma_companies": False, "authors": False,
                "citations": False, "document_metadata": False, "tables": False,
            },
            "entities_only": {
                "drugs": True, "diseases": True, "genes": True, "abbreviations": True,
                "feasibility": False, "pharma_companies": False, "authors": False,
                "citations": False, "document_metadata": False, "tables": False,
            },
            "clinical_entities": {
                "drugs": True, "diseases": True, "genes": False, "abbreviations": False,
                "feasibility": False, "pharma_companies": False, "authors": False,
                "citations": False, "document_metadata": False, "tables": False,
            },
            "metadata_only": {
                "drugs": False, "diseases": False, "genes": False, "abbreviations": False,
                "feasibility": False, "pharma_companies": False, "authors": True,
                "citations": True, "document_metadata": True, "tables": False,
            },
            "standard": {
                "drugs": True, "diseases": True, "genes": True, "abbreviations": True,
                "feasibility": True, "pharma_companies": False, "authors": False,
                "citations": False, "document_metadata": False, "tables": True,
                "care_pathways": True, "recommendations": True, "figures": True,
            },
            "all": {
                "drugs": True, "diseases": True, "genes": True, "abbreviations": True,
                "feasibility": True, "pharma_companies": True, "authors": True,
                "citations": True, "document_metadata": True, "tables": True,
                "care_pathways": True, "recommendations": True, "figures": True,
            },
            "minimal": {
                "drugs": False, "diseases": False, "genes": False, "abbreviations": True,
                "feasibility": False, "pharma_companies": False, "authors": False,
                "citations": False, "document_metadata": False, "tables": False,
            },
            "images_only": {
                "drugs": False, "diseases": False, "genes": False, "abbreviations": False,
                "feasibility": False, "pharma_companies": False, "authors": False,
                "citations": False, "document_metadata": False, "tables": True,
                "care_pathways": False, "recommendations": False, "figures": True,
            },
            "tables_only": {
                "drugs": False, "diseases": False, "genes": False, "abbreviations": False,
                "feasibility": False, "pharma_companies": False, "authors": False,
                "citations": False, "document_metadata": False, "tables": True,
                "care_pathways": False, "recommendations": False, "figures": False,
            },
        }

        # Apply preset if set, otherwise use individual flags
        self.active_preset = preset
        if preset and preset in PRESETS:
            preset_config = PRESETS[preset]
            self.extract_drugs = preset_config["drugs"]
            self.extract_diseases = preset_config["diseases"]
            self.extract_genes = preset_config.get("genes", True)
            self.extract_abbreviations = preset_config["abbreviations"]
            self.extract_feasibility = preset_config["feasibility"]
            self.extract_pharma = preset_config["pharma_companies"]
            self.extract_authors = preset_config["authors"]
            self.extract_citations = preset_config["citations"]
            self.extract_doc_metadata = preset_config["document_metadata"]
            self.extract_tables = preset_config["tables"]
            self.extract_care_pathways = preset_config.get("care_pathways", False)
            self.extract_recommendations = preset_config.get("recommendations", False)
            self.extract_figures = preset_config.get("figures", False)
        else:
            # Use individual extractor flags
            self.active_preset = None
            self.extract_drugs = extractors.get("drugs", True)
            self.extract_diseases = extractors.get("diseases", True)
            self.extract_genes = extractors.get("genes", True)
            self.extract_abbreviations = extractors.get("abbreviations", True)
            self.extract_feasibility = extractors.get("feasibility", True)
            self.extract_pharma = extractors.get("pharma_companies", False)
            self.extract_authors = extractors.get("authors", False)
            self.extract_citations = extractors.get("citations", False)
            self.extract_doc_metadata = extractors.get("document_metadata", False)
            self.extract_tables = extractors.get("tables", True)
            self.extract_care_pathways = extractors.get("care_pathways", True)
            self.extract_recommendations = extractors.get("recommendations", True)
            self.extract_figures = extractors.get("figures", True)

        # Processing options (always read from options, not affected by preset)
        self.use_llm_validation = options.get("use_llm_validation", True)
        self.use_llm_feasibility = options.get("use_llm_feasibility", True)
        self.use_vlm_tables = options.get("use_vlm_tables", False)
        self.use_normalization = options.get("use_normalization", True)
        self.use_epi_enricher = options.get("use_epi_enricher", True)
        self.use_zeroshot_bioner = options.get("use_zeroshot_bioner", True)
        self.use_biomedical_ner = options.get("use_biomedical_ner", True)
        self.use_patient_journey = options.get("use_patient_journey", True)
        self.use_registry_extraction = options.get("use_registry_extraction", True)
        self.use_genetic_extraction = options.get("use_genetic_extraction", True)

    def _print_extraction_config(self) -> None:
        """Print extraction configuration summary."""
        # ANSI color codes
        GREEN = "\033[92m"
        RED = "\033[91m"
        CYAN = "\033[96m"
        RESET = "\033[0m"

        print(f"\n  {CYAN}Extraction Pipeline Configuration:{RESET}")
        print("  " + "-" * 40)

        # Show active preset if set
        if self.active_preset:
            print(f"  PRESET: {CYAN}{self.active_preset}{RESET}")
        else:
            print("  PRESET: (custom - using individual flags)")

        # Extractors
        print(f"\n  {CYAN}EXTRACTORS:{RESET}")
        extractors = [
            ("drugs", self.extract_drugs),
            ("diseases", self.extract_diseases),
            ("genes", self.extract_genes),
            ("abbreviations", self.extract_abbreviations),
            ("feasibility", self.extract_feasibility),
            ("pharma_companies", self.extract_pharma),
            ("authors", self.extract_authors),
            ("citations", self.extract_citations),
            ("document_metadata", self.extract_doc_metadata),
            ("tables", self.extract_tables),
            ("care_pathways", self.extract_care_pathways),
            ("recommendations", self.extract_recommendations),
            ("figures", self.extract_figures),
        ]
        for name, enabled in extractors:
            if enabled:
                status = f"{GREEN}ON{RESET}"
            else:
                status = f"{RED}OFF{RESET}"
            print(f"    {name:<20} {status}")

        # Options
        print(f"\n  {CYAN}OPTIONS:{RESET}")
        options = [
            ("use_llm_validation", self.use_llm_validation),
            ("use_llm_feasibility", self.use_llm_feasibility),
            ("use_vlm_tables", self.use_vlm_tables),
            ("use_normalization", self.use_normalization),
            ("use_epi_enricher", self.use_epi_enricher),
            ("use_zeroshot_bioner", self.use_zeroshot_bioner),
            ("use_biomedical_ner", self.use_biomedical_ner),
            ("use_patient_journey", self.use_patient_journey),
            ("use_registry_extraction", self.use_registry_extraction),
            ("use_genetic_extraction", self.use_genetic_extraction),
        ]
        for name, enabled in options:
            if enabled:
                status = f"{GREEN}ON{RESET}"
            else:
                status = f"{RED}OFF{RESET}"
            print(f"    {name:<20} {status}")

        print("  " + "-" * 40)

    def _init_components(self, api_key: Optional[str]) -> None:
        """Initialize all pipeline components using the factory."""
        # Create component factory
        self.factory = ComponentFactory(
            config=self.config,
            run_id=self.run_id,
            pipeline_version=PIPELINE_VERSION,
            log_dir=self.log_dir,
            api_key=api_key,
        )

        # Create core components
        self.parser = self.factory.create_parser()
        self.table_extractor = self.factory.create_table_extractor()
        self.generators = self.factory.create_generators()

        # Create Claude client and LLM engine
        self.claude_client = self.factory.create_claude_client(self.model)
        self.llm_engine = self.factory.create_llm_engine(self.claude_client, self.model)
        self.vlm_table_extractor = self.factory.create_vlm_table_extractor(self.claude_client)

        # Create validation logger
        self.logger = self.factory.create_validation_logger()

        # Create normalization components
        self.term_mapper = self.factory.create_term_mapper()
        self.disambiguator = self.factory.create_disambiguator()
        self.deduplicator = self.factory.create_deduplicator()

        # Create entity detectors
        self.disease_detector = self.factory.create_disease_detector()
        self.disease_normalizer = self.factory.create_disease_normalizer()
        self.drug_detector = self.factory.create_drug_detector()
        self.gene_detector = self.factory.create_gene_detector()
        self.pharma_detector = self.factory.create_pharma_detector()
        self.author_detector = self.factory.create_author_detector()
        self.citation_detector = self.factory.create_citation_detector()

        # Create feasibility components
        self.feasibility_detector = self.factory.create_feasibility_detector()
        self.llm_feasibility_extractor = self.factory.create_llm_feasibility_extractor(
            self.claude_client
        )

        # Create enrichers
        self.epi_enricher = self.factory.create_epi_enricher()
        self.zeroshot_bioner = self.factory.create_zeroshot_bioner()
        self.biomedical_ner = self.factory.create_biomedical_ner()
        self.patient_journey_enricher = self.factory.create_patient_journey_enricher()
        self.registry_enricher = self.factory.create_registry_enricher()
        self.genetic_enricher = self.factory.create_genetic_enricher()

        # Create document metadata strategy
        self.doc_metadata_strategy = self.factory.create_doc_metadata_strategy(
            self.claude_client, self.model
        )

        # Create NCT enricher
        self.nct_enricher = self.factory.create_nct_enricher()

        # Load rare disease lookup
        self.rare_disease_lookup = self.factory.load_rare_disease_lookup()

        # Create clinical intelligence extractors
        self.flowchart_extractor = FlowchartGraphExtractor(
            llm_client=self.claude_client,
            llm_model=self.model,
        ) if self.claude_client else None

        self.recommendation_extractor = GuidelineRecommendationExtractor(
            llm_client=self.claude_client,
            llm_model=self.model,
        ) if self.claude_client else None

        # Create abbreviation pipeline
        self.abbreviation_pipeline = AbbreviationPipeline(
            run_id=self.run_id,
            pipeline_version=PIPELINE_VERSION,
            parser=self.parser,
            table_extractor=self.table_extractor,
            generators=self.generators,
            heuristics=self.heuristics,
            term_mapper=self.term_mapper,
            disambiguator=self.disambiguator,
            deduplicator=self.deduplicator,
            logger=self.logger,
            claude_client=self.claude_client,
            llm_engine=self.llm_engine,
            vlm_table_extractor=self.vlm_table_extractor,
            nct_enricher=self.nct_enricher,
            rare_disease_lookup=self.rare_disease_lookup,
            use_vlm_tables=self.use_vlm_tables,
            use_normalization=self.use_normalization,
            model=self.model,
        )

        # Create entity processor
        self.entity_processor = EntityProcessor(
            run_id=self.run_id,
            pipeline_version=PIPELINE_VERSION,
            disease_detector=self.disease_detector,
            disease_normalizer=self.disease_normalizer,
            drug_detector=self.drug_detector,
            gene_detector=self.gene_detector,
            pharma_detector=self.pharma_detector,
            author_detector=self.author_detector,
            citation_detector=self.citation_detector,
        )

        # Create feasibility processor
        self.feasibility_processor = FeasibilityProcessor(
            run_id=self.run_id,
            feasibility_detector=self.feasibility_detector,
            llm_feasibility_extractor=self.llm_feasibility_extractor,
            epi_enricher=self.epi_enricher,
            zeroshot_bioner=self.zeroshot_bioner,
            biomedical_ner=self.biomedical_ner,
            patient_journey_enricher=self.patient_journey_enricher,
            registry_enricher=self.registry_enricher,
            genetic_enricher=self.genetic_enricher,
        )

        # Create export manager
        self.export_manager = ExportManager(
            run_id=self.run_id,
            pipeline_version=PIPELINE_VERSION,
            output_dir=self.output_dir,
            gold_json=self.gold_json,
            claude_client=self.claude_client,
        )

        # Create visual pipeline integration
        self.visual_integration = VisualPipelineIntegration(self.config)

        # Print unified lexicon summary
        self._print_unified_lexicon_summary()

    def _print_unified_lexicon_summary(self) -> None:
        """Print unified summary of all loaded lexicons across all detectors."""
        # Collect stats from all sources
        all_stats: List[tuple] = []
        seen_files: set = set()

        # From C04 (abbreviation generator)
        for gen in self.generators:
            if hasattr(gen, "_lexicon_stats"):
                for name, count, filename in gen._lexicon_stats:
                    if name in ("ANCA disease", "IgAN disease", "PAH disease"):
                        continue
                    if name in ("Abbreviations", "Clinical research", "UMLS biological", "UMLS clinical"):
                        cat = "Abbreviation"
                    elif name == "Rare disease acronyms":
                        cat = "Disease"
                    else:
                        cat = "Other"
                    if filename not in seen_files:
                        all_stats.append((name, count, filename, cat))
                        seen_files.add(filename)

        # From C07 (drug detector)
        if self.drug_detector and hasattr(self.drug_detector, "_lexicon_stats"):
            for name, count, filename in self.drug_detector._lexicon_stats:
                key = f"Drug:{filename}"
                if key not in seen_files:
                    all_stats.append((name, count, filename, "Drug"))
                    seen_files.add(key)

        # From C06 (disease detector)
        if self.disease_detector and hasattr(self.disease_detector, "_lexicon_stats"):
            for name, count, filename in self.disease_detector._lexicon_stats:
                display_name = name.replace("Specialized ", "")
                key = f"Disease:{filename}"
                if key not in seen_files:
                    all_stats.append((display_name, count, filename, "Disease"))
                    seen_files.add(key)

        if not all_stats:
            return

        # Group by category with explicit order
        categories: list[tuple[str, list]] = [
            ("Abbreviation", []),
            ("Drug", []),
            ("Disease", []),
            ("Other", []),
        ]
        cat_dict = {name: items for name, items in categories}

        for name, count, filename, cat in all_stats:
            if cat in cat_dict:
                cat_dict[cat].append((name, count, filename))

        # Calculate totals
        total = sum(count for _, count, _, _ in all_stats if count > 1)
        file_count = len([s for s in all_stats if s[1] > 0])

        print(f"\nLexicons loaded: {file_count} sources, {total:,} entries")
        print("-" * 70)

        for cat_name, items in categories:
            if not items:
                continue
            cat_total = sum(count for _, count, _ in items if count > 1)
            print(f"  {cat_name} ({cat_total:,} entries)")
            for name, count, filename in items:
                if count > 1:
                    print(f"    * {name:<26} {count:>8,}  {filename}")
                else:
                    print(f"    * {name:<26} {'enabled':>8}  {filename}")
        print()

    def _get_output_dir(self, pdf_path: Path) -> Path:
        """Get output directory for a PDF file."""
        return self.export_manager.get_output_dir(pdf_path)

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

        # Create metrics for this document (single source of truth)
        doc_id = str(pdf_path_obj.stem)
        metrics = PipelineMetrics(run_id=self.run_id, doc_id=doc_id)

        # Initialize stage timer
        timer = StageTimer()
        timer.start("total")

        print(f"\n{'=' * 60}")
        print(f"Processing: {pdf_path_obj.name}")
        print(f"{'=' * 60}")

        # Stage 1: Parse PDF
        timer.start("1. PDF Parsing")
        output_dir = self._get_output_dir(pdf_path_obj)
        doc = self.abbreviation_pipeline.parse_pdf(pdf_path_obj, output_dir)
        parse_time = timer.stop("1. PDF Parsing")
        print(f"  ⏱  {parse_time:.1f}s")

        # Export extracted text
        self.export_manager.export_extracted_text(pdf_path_obj, doc)

        # Build full_text with page markers (needed for VLM page detection in recommendations)
        text_lines = []
        current_page = None
        for block in doc.iter_linear_blocks():
            if block.page_num != current_page:
                if current_page is not None:
                    text_lines.append("")
                text_lines.append(f"--- Page {block.page_num} ---")
                text_lines.append("")
                current_page = block.page_num
            text = (block.text or "").strip()
            if text:
                text_lines.append(text)
        full_text_with_pages = "\n".join(text_lines)
        full_text = full_text_with_pages  # Will be overwritten by abbreviation pipeline if enabled

        # Stage 2-4: Abbreviation extraction
        results: List[ExtractedEntity] = []
        unique_candidates: List[Candidate] = []
        counters = HeuristicsCounters()

        if self.extract_abbreviations:
            timer.start("2. Candidate Generation")
            unique_candidates, full_text = self.abbreviation_pipeline.generate_candidates(doc)
            # Abbreviation pipeline returns text without page markers, but we keep full_text_with_pages
            gen_time = timer.stop("2. Candidate Generation")
            print(f"  ⏱  {gen_time:.1f}s")

            # Update generation metrics
            metrics.generation.generated_candidates = len(unique_candidates)
            metrics.generation.unique_short_forms = len({c.short_form.upper() for c in unique_candidates})

            if not self.use_llm_validation:
                print("\n[3/12] Validation SKIPPED")
            else:
                # Filter candidates
                needs_validation, corroborated_sfs, word_counts, filtered_count, sf_form_rejected = (
                    self.abbreviation_pipeline.filter_candidates(unique_candidates, full_text)
                )

                # Update generation metrics with filtered count
                metrics.generation.filtered_lexicon_only = filtered_count

                # Include sf_form_rejected from filter_candidates in counters
                # These are candidates filtered by is_valid_sf_form() before apply_heuristics
                counters.form_filter_rejected += sf_form_rejected

                # Apply heuristics
                auto_results, llm_candidates = self.abbreviation_pipeline.apply_heuristics(
                    needs_validation, counters
                )

                # Update heuristics metrics from counters
                auto_approved = len([e for _, e in auto_results if e.status == ValidationStatus.VALIDATED])
                auto_rejected_total = (
                    counters.blacklisted_fp_count +
                    counters.context_rejected +
                    counters.trial_id_excluded +
                    counters.common_word_rejected +
                    counters.form_filter_rejected
                )
                # total_processed includes sf_form_rejected since they were processed (filtered)
                metrics.heuristics.total_processed = len(needs_validation) + sf_form_rejected
                metrics.heuristics.auto_approved = auto_approved
                metrics.heuristics.auto_rejected = auto_rejected_total
                metrics.heuristics.sent_to_llm = len(llm_candidates)
                metrics.heuristics.approved_by_stats_whitelist = counters.recovered_by_stats_whitelist
                metrics.heuristics.approved_by_country_code = counters.recovered_by_country_code
                metrics.heuristics.rejected_by_blacklist = counters.blacklisted_fp_count
                metrics.heuristics.rejected_by_context = counters.context_rejected
                metrics.heuristics.rejected_by_trial_id = counters.trial_id_excluded
                metrics.heuristics.rejected_by_common_word = counters.common_word_rejected

                # Print stats
                from A_core.A01_domain_models import GeneratorType
                frequent_sfs = sum(
                    1
                    for c in unique_candidates
                    if c.generator_type == GeneratorType.LEXICON_MATCH
                    and c.short_form.upper() not in corroborated_sfs
                    and word_counts.get(c.short_form.upper(), 0) >= 2
                )

                print("\n[3/12] Validating candidates with Claude...")
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

                timer.start("3. LLM Validation")

                # Log auto results
                for candidate, entity in auto_results:
                    self.logger.log_validation(candidate, entity, entity.raw_llm_response, 0)
                    results.append(entity)

                # LLM validation
                llm_results = self.abbreviation_pipeline.validate_with_llm(llm_candidates, delay)
                results.extend(llm_results)

                # Update validation metrics
                llm_approved = sum(1 for r in llm_results if r.status == ValidationStatus.VALIDATED)
                llm_rejected = sum(1 for r in llm_results if r.status == ValidationStatus.REJECTED)
                llm_ambiguous = sum(1 for r in llm_results if r.status == ValidationStatus.AMBIGUOUS)
                metrics.validation.total_validated = len(llm_results)
                metrics.validation.llm_approved = llm_approved
                metrics.validation.llm_rejected = llm_rejected
                metrics.validation.llm_ambiguous = llm_ambiguous
                metrics.validation.llm_calls = len(llm_candidates)

                val_time = timer.stop("3. LLM Validation")
                print(f"  ⏱  {val_time:.1f}s")

                # Search for missing abbreviations
                found_sfs = {
                    r.short_form.upper()
                    for r in results
                    if r.status == ValidationStatus.VALIDATED
                }

                search_results = self.abbreviation_pipeline.search_missing_abbreviations(
                    doc_id, full_text, found_sfs, counters
                )
                results.extend(search_results)

                # LLM SF-only extraction
                sf_only_results = self.abbreviation_pipeline.extract_sf_only_with_llm(
                    doc_id, full_text, found_sfs, counters, delay_ms=delay
                )
                results.extend(sf_only_results)

                # Update SF-only metrics
                metrics.validation.sf_only_extracted = len(search_results)
                metrics.validation.sf_only_from_llm = len(sf_only_results)

            # Normalize
            timer.start("4. Normalization")
            results = self.abbreviation_pipeline.normalize_results(results, full_text)
            norm_time = timer.stop("4. Normalization")
            print(f"  ⏱  {norm_time:.1f}s")
        else:
            print("\n[Abbreviation detection] SKIPPED (disabled in config)")

        # Stage 5-9: Entity extraction
        disease_results: List[ExtractedDisease] = []
        if self.extract_diseases:
            timer.start("5. Disease Detection")
            disease_results = self.entity_processor.process_diseases(doc, pdf_path_obj)
            disease_time = timer.stop("5. Disease Detection")
            print(f"  ⏱  {disease_time:.1f}s")
        else:
            print("\n[Disease detection] SKIPPED (disabled in config)")

        gene_results: List[ExtractedGene] = []
        if self.extract_genes:
            timer.start("6. Gene Detection")
            gene_results = self.entity_processor.process_genes(doc, pdf_path_obj)
            gene_time = timer.stop("6. Gene Detection")
            print(f"  ⏱  {gene_time:.1f}s")
        else:
            print("\n[Gene detection] SKIPPED (disabled in config)")

        drug_results: List[ExtractedDrug] = []
        if self.extract_drugs:
            timer.start("7. Drug Detection")
            drug_results = self.entity_processor.process_drugs(doc, pdf_path_obj)
            drug_time = timer.stop("7. Drug Detection")
            print(f"  ⏱  {drug_time:.1f}s")
        else:
            print("\n[Drug detection] SKIPPED (disabled in config)")

        pharma_results: List[ExtractedPharma] = []
        if self.extract_pharma:
            timer.start("8. Pharma Detection")
            pharma_results = self.entity_processor.process_pharma(doc, pdf_path_obj)
            pharma_time = timer.stop("8. Pharma Detection")
            print(f"  ⏱  {pharma_time:.1f}s")
        else:
            print("\n[Pharma detection] SKIPPED (disabled in config)")

        author_results: List[ExtractedAuthor] = []
        if self.extract_authors:
            timer.start("9a. Author Detection")
            author_results = self.entity_processor.process_authors(doc, pdf_path_obj, full_text)
            author_time = timer.stop("9a. Author Detection")
            print(f"  ⏱  {author_time:.1f}s")
        else:
            print("\n[Author detection] SKIPPED (disabled in config)")

        citation_results: List[ExtractedCitation] = []
        if self.extract_citations:
            timer.start("9b. Citation Detection")
            citation_results = self.entity_processor.process_citations(doc, pdf_path_obj, full_text)
            citation_time = timer.stop("9b. Citation Detection")
            print(f"  ⏱  {citation_time:.1f}s")
        else:
            print("\n[Citation detection] SKIPPED (disabled in config)")

        # Stage 10: Feasibility extraction
        feasibility_results: List[FeasibilityCandidate | NERCandidate] = []
        if self.extract_feasibility:
            timer.start("10. Feasibility")
            feasibility_results = self.feasibility_processor.process(doc, pdf_path_obj, full_text)
            feas_time = timer.stop("10. Feasibility")
            print(f"  ⏱  {feas_time:.1f}s")
        else:
            print("\n[Feasibility extraction] SKIPPED (disabled in config)")

        # Stage 10a: Care pathway extraction from flowchart figures
        care_pathway_results: List[CarePathway] = []
        if self.extract_care_pathways and self.flowchart_extractor:
            timer.start("10a. Care Pathways")
            print("\n[10a/12] Extracting care pathways from flowchart figures...")
            care_pathway_results = self._extract_care_pathways(doc, pdf_path_obj)
            pathway_time = timer.stop("10a. Care Pathways")
            print(f"  Extracted {len(care_pathway_results)} care pathways")
            print(f"  ⏱  {pathway_time:.1f}s")
        else:
            print("\n[Care pathway extraction] SKIPPED (disabled in config)")

        # Stage 10b: Guideline recommendation extraction
        recommendation_results: List[RecommendationSet] = []
        if self.extract_recommendations and self.recommendation_extractor:
            timer.start("10b. Recommendations")
            print("\n[10b/12] Extracting guideline recommendations...")
            # Use full_text_with_pages for VLM page detection
            recommendation_results = self._extract_recommendations(doc, pdf_path_obj, full_text_with_pages)
            rec_time = timer.stop("10b. Recommendations")
            total_recs = sum(len(rs.recommendations) for rs in recommendation_results)
            print(f"  Extracted {len(recommendation_results)} recommendation sets ({total_recs} recommendations)")
            print(f"  ⏱  {rec_time:.1f}s")
        else:
            print("\n[Recommendation extraction] SKIPPED (disabled in config)")

        # Stage 10c: Visual extraction (tables and figures as images)
        visual_result = None
        self._last_visual_result = None  # Store for summary printing
        if (self.extract_tables or self.extract_figures) and self.visual_integration.enabled:
            timer.start("10c. Visual Extraction")
            print("\n[10c/12] Extracting visuals (tables and figures)...")
            try:
                visual_result = self.visual_integration.extract(str(pdf_path_obj))
                if visual_result:
                    self._last_visual_result = visual_result
                    print(f"    + Extracted {len(visual_result.visuals)} visuals "
                          f"({visual_result.tables_detected} tables, {visual_result.figures_detected} figures)")
            except Exception as e:
                print(f"    [WARN] Visual extraction failed: {e}")
            visual_time = timer.stop("10c. Visual Extraction")
            print(f"  ⏱  {visual_time:.1f}s")
        else:
            print("\n[Visual extraction] SKIPPED (disabled in config)")

        # Stage 11: Document metadata extraction
        doc_metadata: Optional[DocumentMetadata] = None
        if self.extract_doc_metadata:
            timer.start("11. Doc Metadata")
            doc_metadata = self._process_document_metadata(
                doc, pdf_path_obj, full_text[:5000]
            )
            meta_time = timer.stop("11. Doc Metadata")
            print(f"  ⏱  {meta_time:.1f}s")
        else:
            print("\n[Document metadata] SKIPPED (disabled in config)")

        # Stage 12: Summary & Export
        timer.start("12. Export")
        print("\n[12/12] Writing summary...")
        self.logger.write_summary()
        self.logger.print_summary()
        counters.log_summary()

        # Export abbreviation results (only if extraction was enabled)
        if self.extract_abbreviations:
            self.export_manager.export_results(
                pdf_path_obj, results, unique_candidates, counters,
                disease_results=disease_results if disease_results else None,
                drug_results=drug_results if drug_results else None,
                pharma_results=pharma_results if pharma_results else None,
            )

        if disease_results:
            self.export_manager.export_disease_results(pdf_path_obj, disease_results)

        if self.extract_genes:
            self.export_manager.export_gene_results(pdf_path_obj, gene_results)

        if drug_results:
            self.export_manager.export_drug_results(pdf_path_obj, drug_results)

        if pharma_results:
            self.export_manager.export_pharma_results(pdf_path_obj, pharma_results)

        if author_results:
            self.export_manager.export_author_results(pdf_path_obj, author_results)

        if citation_results:
            self.export_manager.export_citation_results(pdf_path_obj, citation_results)

        if feasibility_results:
            feasibility_only = [r for r in feasibility_results if isinstance(r, FeasibilityCandidate)]
            self.export_manager.export_feasibility_results(pdf_path_obj, feasibility_only, doc)

        if doc is not None:
            self.export_manager.export_images(pdf_path_obj, doc)

        if doc is not None and self.extract_tables:
            self.export_manager.export_tables(pdf_path_obj, doc)

        if doc_metadata:
            self.export_manager.export_document_metadata(pdf_path_obj, doc_metadata)

        if care_pathway_results:
            self.export_manager.export_care_pathways(pdf_path_obj, care_pathway_results)

        if recommendation_results:
            self.export_manager.export_recommendations(pdf_path_obj, recommendation_results)

        if visual_result:
            exported_paths = self.visual_integration.export(
                visual_result,
                output_dir,
                doc_name=pdf_path_obj.stem,
                export_images=True,
            )
            print(f"  Visual exports: {len(exported_paths)} files")

        # Update export metrics
        validated_count = sum(1 for r in results if r.status == ValidationStatus.VALIDATED)
        rejected_count = sum(1 for r in results if r.status == ValidationStatus.REJECTED)
        ambiguous_count = sum(1 for r in results if r.status == ValidationStatus.AMBIGUOUS)
        metrics.export.validated = validated_count
        metrics.export.rejected = rejected_count
        metrics.export.ambiguous = ambiguous_count

        # Entity type breakdown for export metrics
        if disease_results:
            metrics.export.by_entity_type["disease"] = sum(
                1 for d in disease_results if d.status == ValidationStatus.VALIDATED
            )
        if gene_results:
            metrics.export.by_entity_type["gene"] = sum(
                1 for g in gene_results if g.status == ValidationStatus.VALIDATED
            )
        if drug_results:
            metrics.export.by_entity_type["drug"] = sum(
                1 for d in drug_results if d.status == ValidationStatus.VALIDATED
            )

        # Validate metrics invariants
        invariant_errors = metrics.validate_invariants()
        if invariant_errors:
            self._logger.warning(f"Metrics invariant violations: {invariant_errors}")
            for err in invariant_errors:
                print(f"  [WARN] Metrics: {err}")

        # Print metrics summary
        print(f"\nPIPELINE METRICS ({doc_id}):")
        print(f"  Generated:      {metrics.generation.generated_candidates}")
        print(f"  Auto-approved:  {metrics.heuristics.auto_approved}")
        print(f"  Auto-rejected:  {metrics.heuristics.auto_rejected}")
        print(f"  Sent to LLM:    {metrics.heuristics.sent_to_llm}")
        print(f"  LLM approved:   {metrics.validation.llm_approved}")
        print(f"  LLM rejected:   {metrics.validation.llm_rejected}")
        print(f"  Exported:       {metrics.export.validated} validated / {metrics.export.rejected} rejected")

        # Print validated results summary (only for enabled extractors)
        self._print_validated_summary(
            results, disease_results, gene_results, drug_results,
            pharma_results, feasibility_results, doc_metadata,
            care_pathway_results, recommendation_results,
            extract_diseases=self.extract_diseases,
            extract_genes=self.extract_genes,
            extract_drugs=self.extract_drugs,
            extract_pharma=self.extract_pharma,
            extract_feasibility=self.extract_feasibility,
            extract_care_pathways=self.extract_care_pathways,
            extract_recommendations=self.extract_recommendations,
        )

        # Stop export timer and total timer
        timer.stop("12. Export")
        timer.stop("total")

        # Print timing summary
        timer.print_summary()

        return results

    def _process_document_metadata(
        self, doc, pdf_path: Path, content_sample: str
    ) -> Optional[DocumentMetadata]:
        """Extract document metadata including classification and descriptions."""
        if self.doc_metadata_strategy is None:
            return None

        print("\n[11/12] Extracting document metadata...")

        try:
            metadata = self.doc_metadata_strategy.extract(
                file_path=str(pdf_path),
                doc_graph=doc,
                doc_id=pdf_path.stem,
                content_sample=content_sample,
            )
            return metadata
        except Exception as e:
            print(f"  [WARN] Document metadata extraction failed: {e}")
            return None

    def _extract_care_pathways(
        self, doc, pdf_path: Path
    ) -> List[CarePathway]:
        """Extract care pathways from flowchart/algorithm figures."""
        from B_parsing.B02_doc_graph import ImageType

        if not self.flowchart_extractor:
            return []

        pathways: List[CarePathway] = []

        # Look for flowchart images in the document
        for img in doc.iter_images():
            # Only process flowcharts (patient flow diagrams, treatment algorithms)
            if img.image_type != ImageType.FLOWCHART:
                continue

            if not img.image_base64:
                continue

            try:
                # Extract care pathway from the flowchart
                pathway = self.flowchart_extractor.extract_care_pathway(
                    image_base64=img.image_base64,
                    ocr_text=img.ocr_text,
                    caption=img.caption,
                    figure_id=f"Figure_p{img.page_num}",
                    page_num=img.page_num,
                )

                if pathway and pathway.nodes:
                    pathways.append(pathway)
                    print(f"    + Page {img.page_num}: {len(pathway.nodes)} nodes, {len(pathway.edges)} edges")

            except Exception as e:
                print(f"    [WARN] Care pathway extraction failed for page {img.page_num}: {e}")

        return pathways

    def _extract_recommendations(
        self, doc, pdf_path: Path, full_text: str
    ) -> List[RecommendationSet]:
        """Extract guideline recommendations from text and tables."""
        if not self.recommendation_extractor:
            return []

        # Set PDF path for VLM-based extraction
        self.recommendation_extractor.pdf_path = str(pdf_path)

        recommendation_sets: List[RecommendationSet] = []

        # Extract from full text (LLM-based extraction)
        # Pass full text to extractor - it handles truncation internally for LLM context
        # while using full text for VLM page detection
        try:
            text_recs = self.recommendation_extractor.extract_from_text(
                text=full_text,
                source=pdf_path.stem,
                use_llm=True,
            )
            if text_recs and text_recs.recommendations:
                recommendation_sets.append(text_recs)
                print(f"    + Text extraction: {len(text_recs.recommendations)} recommendations")
        except Exception as e:
            print(f"    [WARN] Text recommendation extraction failed: {e}")

        # Extract from tables (pattern-based extraction)
        for table in doc.iter_tables():
            if not table.logical_rows:
                continue

            # Check if table looks like a recommendation table
            # (has treatment/recommendation columns)
            headers = table.logical_rows[0] if table.logical_rows else []
            headers_lower = [str(h).lower() for h in headers]

            is_rec_table = any(
                keyword in " ".join(headers_lower)
                for keyword in ["recommend", "treatment", "therapy", "dose", "indication"]
            )

            if not is_rec_table:
                continue

            try:
                table_data = {
                    "title": table.caption or f"Table page {table.page_num}",
                    "headers": headers,
                    "rows": table.logical_rows[1:] if len(table.logical_rows) > 1 else [],
                }

                table_recs = self.recommendation_extractor.extract_from_table(
                    table_data=table_data,
                    source=f"table_p{table.page_num}",
                    page_num=table.page_num,
                )

                if table_recs and table_recs.recommendations:
                    recommendation_sets.append(table_recs)
                    print(f"    + Table page {table.page_num}: {len(table_recs.recommendations)} recommendations")

            except Exception as e:
                print(f"    [WARN] Table recommendation extraction failed for page {table.page_num}: {e}")

        return recommendation_sets

    def _print_validated_summary(
        self,
        results: List[ExtractedEntity],
        disease_results: List[ExtractedDisease],
        gene_results: List[ExtractedGene],
        drug_results: List[ExtractedDrug],
        pharma_results: List[ExtractedPharma],
        feasibility_results: List,
        doc_metadata: Optional[DocumentMetadata],
        care_pathway_results: Optional[List[CarePathway]] = None,
        recommendation_results: Optional[List[RecommendationSet]] = None,
        *,
        extract_diseases: bool = True,
        extract_genes: bool = True,
        extract_drugs: bool = True,
        extract_pharma: bool = True,
        extract_feasibility: bool = True,
        extract_care_pathways: bool = True,
        extract_recommendations: bool = True,
    ) -> None:
        """Print summary of validated entities.

        Only prints sections for extractors that were enabled.
        This prevents confusing output like "0 diseases" when disease
        extraction was intentionally disabled.

        Args:
            results: Validated abbreviation entities
            disease_results: Disease extraction results
            gene_results: Gene extraction results
            drug_results: Drug extraction results
            pharma_results: Pharma company extraction results
            feasibility_results: Feasibility extraction results
            doc_metadata: Document metadata
            care_pathway_results: Care pathway extraction results
            recommendation_results: Recommendation extraction results
            extract_diseases: Whether disease extractor was enabled
            extract_genes: Whether gene extractor was enabled
            extract_drugs: Whether drug extractor was enabled
            extract_pharma: Whether pharma extractor was enabled
            extract_feasibility: Whether feasibility extractor was enabled
            extract_care_pathways: Whether care pathway extractor was enabled
            extract_recommendations: Whether recommendation extractor was enabled
        """
        # Print validated abbreviations
        validated = [r for r in results if r.status == ValidationStatus.VALIDATED]
        if validated:
            print(f"\nValidated abbreviations ({len(validated)}):")
            for v in validated:
                src = ""
                if v.provenance and v.provenance.lexicon_source:
                    lex = v.provenance.lexicon_source
                    if lex.startswith("2025_08_"):
                        lex = lex.replace("2025_08_", "").replace(".json", "").replace(".tsv", "")
                    src = f" [{lex}]"
                print(f"  * {v.short_form} -> {v.long_form or '(no expansion)'}{src}")

        # Print validated diseases (only if extractor was enabled)
        if extract_diseases:
            validated_diseases = [d for d in disease_results if d.status == ValidationStatus.VALIDATED]
            if validated_diseases:
                print(f"\nValidated diseases ({len(validated_diseases)}):")
                for d in validated_diseases:
                    codes = []
                    if d.icd10_code:
                        codes.append(f"ICD-10:{d.icd10_code}")
                    if d.orpha_code:
                        codes.append(f"ORPHA:{d.orpha_code}")
                    code_str = f" [{', '.join(codes)}]" if codes else ""
                    print(f"  * {d.preferred_label}{code_str}")

        # Print validated genes (only if extractor was enabled)
        if extract_genes:
            validated_genes = [g for g in gene_results if g.status == ValidationStatus.VALIDATED]
            if validated_genes:
                print(f"\nValidated genes ({len(validated_genes)}):")
                for g in validated_genes:
                    codes = []
                    if g.hgnc_id:
                        codes.append(f"HGNC:{g.hgnc_id}")
                    if g.entrez_id:
                        codes.append(f"Entrez:{g.entrez_id}")
                    code_str = f" [{', '.join(codes)}]" if codes else ""
                    diseases = len(g.associated_diseases)
                    disease_str = f" ({diseases} diseases)" if diseases else ""
                    print(f"  * {g.hgnc_symbol}{code_str}{disease_str}")

        # Print validated drugs (only if extractor was enabled)
        if extract_drugs:
            validated_drugs = [drug for drug in drug_results if drug.status == ValidationStatus.VALIDATED]
            if validated_drugs:
                print(f"\nValidated drugs ({len(validated_drugs)}):")
                for drug in validated_drugs:
                    phase = f" ({drug.development_phase})" if drug.development_phase else ""
                    compound = f" [{drug.compound_id}]" if drug.compound_id else ""
                    print(f"  * {drug.preferred_name}{compound}{phase}")

        # Print validated pharma companies (only if extractor was enabled)
        if extract_pharma:
            validated_pharma = [p for p in pharma_results if p.status == ValidationStatus.VALIDATED]
            if validated_pharma:
                print(f"\nValidated pharma companies ({len(validated_pharma)}):")
                for p in validated_pharma:
                    hq = f" ({p.headquarters})" if p.headquarters else ""
                    print(f"  * {p.canonical_name}{hq}")

        # Print feasibility summary (only if extractor was enabled)
        if extract_feasibility and feasibility_results:
            print(f"\nFeasibility information ({len(feasibility_results)} items):")
            by_type: Dict[str, List] = {}
            for f in feasibility_results:
                if hasattr(f, 'field_type') and f.field_type is not None:
                    key = f.field_type.value.split("_")[0]
                elif hasattr(f, 'category'):
                    key = f.category.upper()
                else:
                    key = "OTHER"
                by_type.setdefault(key, []).append(f)
            for ftype, items in sorted(by_type.items()):
                print(f"  {ftype}: {len(items)} items")

        # Print document metadata summary
        if doc_metadata:
            print("\nDocument metadata:")
            if doc_metadata.classification:
                cls = doc_metadata.classification.primary_type
                print(f"  Type: {cls.code} - {cls.name} (conf: {cls.confidence:.2f})")
            if doc_metadata.description:
                print(f"  Title: {doc_metadata.description.title}")
            if doc_metadata.date_extraction and doc_metadata.date_extraction.primary_date:
                pd = doc_metadata.date_extraction.primary_date
                print(f"  Date: {pd.date.strftime('%Y-%m-%d')} (source: {pd.source.value})")

        # Print care pathway summary (only if extractor was enabled)
        if extract_care_pathways and care_pathway_results:
            total_nodes = sum(len(p.nodes) for p in care_pathway_results)
            total_edges = sum(len(p.edges) for p in care_pathway_results)
            print(f"\nCare pathways ({len(care_pathway_results)} pathways, {total_nodes} nodes, {total_edges} edges):")
            for p in care_pathway_results:
                drugs_str = f" [{', '.join(p.primary_drugs[:3])}]" if p.primary_drugs else ""
                print(f"  * {p.title or 'Unknown'}{drugs_str}")

        # Print recommendation summary (only if extractor was enabled)
        if extract_recommendations and recommendation_results:
            total_recs = sum(len(rs.recommendations) for rs in recommendation_results)
            print(f"\nGuideline recommendations ({len(recommendation_results)} sets, {total_recs} recommendations):")
            for rs in recommendation_results:
                print(f"  * {rs.guideline_name}: {len(rs.recommendations)} recommendations")
                for rec in rs.recommendations[:3]:  # Show first 3
                    action_short = (rec.action[:60] + "...") if len(rec.action) > 60 else rec.action
                    print(f"    - {rec.population}: {action_short}")
                if len(rs.recommendations) > 3:
                    print(f"    ... and {len(rs.recommendations) - 3} more")

        # Print visual extraction summary (only if extractor was enabled)
        if hasattr(self, '_last_visual_result') and self._last_visual_result:
            result = self._last_visual_result
            print(f"\nVisual extraction ({len(result.visuals)} visuals):")
            print(f"  Tables: {result.tables_detected}")
            print(f"  Figures: {result.figures_detected}")
            if result.tables_escalated > 0:
                print(f"  Tables escalated: {result.tables_escalated}")
            if result.vlm_enriched > 0:
                print(f"  VLM enriched: {result.vlm_enriched}")

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

        batch_start = time.time()
        doc_times: Dict[str, float] = {}

        print(f"\n{'#' * 60}")
        print(f"BATCH PROCESSING: {len(pdf_files)} PDFs")
        print(f"Folder: {folder}")
        print(f"{'#' * 60}")

        all_results = {}

        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"\n[{i}/{len(pdf_files)}] {pdf_path.name}")
            doc_start = time.time()
            try:
                all_results[pdf_path.name] = self.process_pdf(
                    str(pdf_path), batch_delay_ms=batch_delay_ms
                )
                doc_times[pdf_path.name] = time.time() - doc_start
            except Exception as e:
                print(f"  [WARN] ERROR: {e}")
                all_results[pdf_path.name] = []
                doc_times[pdf_path.name] = time.time() - doc_start

        batch_elapsed = time.time() - batch_start

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

        # Print batch timing summary
        print(f"\n{'─' * 50}")
        print("⏱  BATCH TIMING SUMMARY")
        print(f"{'─' * 50}")
        for doc_name, doc_time in doc_times.items():
            print(f"  {doc_name:<40} {doc_time:>6.1f}s")
        print(f"{'─' * 50}")
        print(f"  {'TOTAL BATCH TIME':<40} {batch_elapsed:>6.1f}s")
        if doc_times:
            avg_time = batch_elapsed / len(doc_times)
            print(f"  {'AVERAGE PER DOCUMENT':<40} {avg_time:>6.1f}s")
        print(f"{'─' * 50}")

        return all_results


def main() -> None:
    """
    Process all PDFs in the default folder.

    Initializes the logging system and runs the pipeline orchestrator
    on all PDF files found in the configured input directory.
    """
    # Configure logging for the pipeline run
    configure_logging(
        log_dir="logs",
        log_level=INFO,
        enable_file_logging=True,
        enable_console_logging=False,  # Use print() for console output
    )

    orchestrator = Orchestrator()
    orchestrator.process_folder()
    print("\nPipeline finished.")


if __name__ == "__main__":
    main()

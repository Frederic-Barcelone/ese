# corpus_metadata/H_pipeline/H01_component_factory.py
"""
Component Factory for pipeline initialization.

Centralizes the creation and configuration of all pipeline components
including parsers, generators, validators, normalizers, and detectors.
This factory pattern allows the orchestrator to focus on coordination
rather than component construction details.

Key Components:
    - ComponentFactory: Main factory class for creating pipeline components
    - Parser creation: PDFToDocGraphParser, TableExtractor
    - Generator creation: Syntax, regex, layout, flashtext, glossary generators
    - Validation: ClaudeClient, LLMEngine, ValidationLogger
    - Normalization: TermMapper, Disambiguator, Deduplicator
    - Entity detection: Disease, Drug, Gene, Pharma, Author, Citation detectors
    - Feasibility: FeasibilityDetector, LLMFeasibilityExtractor
    - Enrichment: EpiExtract, ZeroShotBioNER, BiomedicalNER, PubTator enrichers

Example:
    >>> from H_pipeline.H01_component_factory import ComponentFactory
    >>> factory = ComponentFactory(config, run_id, pipeline_version, log_dir)
    >>> parser = factory.create_parser()
    >>> disease_detector = factory.create_disease_detector()
    >>> claude_client = factory.create_claude_client()

Dependencies:
    - B_parsing: PDF parsing and table extraction
    - C_generators: Candidate generation strategies
    - D_validation: LLM validation engine
    - E_normalization: Term mapping and enrichment
    - Z_utils: Path utilities
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from Z_utils.Z05_path_utils import get_base_path

if TYPE_CHECKING:
    from B_parsing.B01_pdf_to_docgraph import PDFToDocGraphParser
    from B_parsing.B03_table_extractor import TableExtractor
    from C_generators.C06_strategy_disease import DiseaseDetector
    from C_generators.C07_strategy_drug import DrugDetector
    from C_generators.C16_strategy_gene import GeneDetector
    from C_generators.C18_strategy_pharma import PharmaCompanyDetector
    from C_generators.C13_strategy_author import AuthorDetector
    from C_generators.C14_strategy_citation import CitationDetector
    from C_generators.C08_strategy_feasibility import FeasibilityDetector
    from C_generators.C11_llm_feasibility import LLMFeasibilityExtractor
    from C_generators.C09_strategy_document_metadata import DocumentMetadataStrategy
    from C_generators.C15_vlm_table_extractor import VLMTableExtractor
    from D_validation.D02_llm_engine import ClaudeClient, LLMEngine
    from D_validation.D03_validation_logger import ValidationLogger
    from E_normalization.E01_term_mapper import TermMapper
    from E_normalization.E02_disambiguator import Disambiguator
    from E_normalization.E07_deduplicator import Deduplicator
    from E_normalization.E03_disease_normalizer import DiseaseNormalizer
    from E_normalization.E06_nct_enricher import NCTEnricher
    from E_normalization.E08_epi_extract_enricher import EpiExtractEnricher
    from E_normalization.E09_zeroshot_bioner import ZeroShotBioNEREnricher
    from E_normalization.E10_biomedical_ner_all import BiomedicalNEREnricher
    from E_normalization.E12_patient_journey_enricher import PatientJourneyEnricher
    from E_normalization.E13_registry_enricher import RegistryEnricher
    from E_normalization.E15_genetic_enricher import GeneticEnricher
    from E_normalization.E04_pubtator_enricher import DiseaseEnricher
    from E_normalization.E05_drug_enricher import DrugEnricher
    from E_normalization.E18_gene_enricher import GeneEnricher


class ComponentFactory:
    """
    Factory for creating and configuring pipeline components.

    Encapsulates all component initialization logic, allowing the orchestrator
    to remain focused on coordination rather than construction details.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        run_id: str,
        pipeline_version: str,
        log_dir: Path,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize the component factory.

        Args:
            config: Configuration dictionary loaded from YAML
            run_id: Unique identifier for this pipeline run
            pipeline_version: Version string for the pipeline
            log_dir: Directory for validation logs
            api_key: Optional API key for Claude client
        """
        self.config = config
        self.run_id = run_id
        self.pipeline_version = pipeline_version
        self.log_dir = log_dir
        self.api_key = api_key

        # Extract paths from config (uses CORPUS_BASE_PATH env var or auto-detects)
        self.paths = config.get("paths", {})
        self.base_path = str(get_base_path(config))
        self.dict_path = Path(self.base_path) / self.paths.get(
            "dictionaries", "ouput_datasources"
        )
        self.lexicons = config.get("lexicons", {})

        # API settings
        api_cfg = config.get("api", {}).get("claude", {})
        self.val_cfg = api_cfg.get("validation", {})
        self.default_model = self.val_cfg.get("model", "claude-sonnet-4-20250514")

        # Options from config
        pipeline_cfg = config.get("extraction_pipeline", {}).get("options", {})
        self.use_llm_validation = pipeline_cfg.get("use_llm_validation", True)
        self.use_llm_feasibility = pipeline_cfg.get("use_llm_feasibility", True)
        self.use_vlm_tables = pipeline_cfg.get("use_vlm_tables", False)
        self.use_epi_enricher = pipeline_cfg.get("use_epi_enricher", True)
        self.use_zeroshot_bioner = pipeline_cfg.get("use_zeroshot_bioner", True)
        self.use_biomedical_ner = pipeline_cfg.get("use_biomedical_ner", True)
        self.use_patient_journey = pipeline_cfg.get("use_patient_journey", True)
        self.use_registry_extraction = pipeline_cfg.get("use_registry_extraction", True)
        self.use_genetic_extraction = pipeline_cfg.get("use_genetic_extraction", True)
        self.use_pubtator_enrichment = pipeline_cfg.get("use_pubtator_enrichment", True)

    def create_parser(self) -> "PDFToDocGraphParser":
        """Create PDF parser component."""
        from B_parsing.B01_pdf_to_docgraph import PDFToDocGraphParser
        return PDFToDocGraphParser()

    def create_table_extractor(self) -> Optional["TableExtractor"]:
        """Create table extractor component.

        Returns None if Docling is not available (with warning).
        """
        # Check if Docling is available before attempting to create extractor
        try:
            from B_parsing.B28_docling_backend import DOCLING_AVAILABLE
            if not DOCLING_AVAILABLE:
                print(
                    "  [WARN] Docling not installed - table extraction DISABLED\n"
                    "         Install with: pip install docling docling-surya"
                )
                return None

            from B_parsing.B03_table_extractor import TableExtractor
            return TableExtractor()
        except ImportError as e:
            print(
                f"  [WARN] Table extraction unavailable: {e}\n"
                "         Install with: pip install docling docling-surya"
            )
            return None

    def create_generators(self) -> List[Any]:
        """Create all candidate generators."""
        from C_generators.C01_strategy_abbrev import AbbrevSyntaxCandidateGenerator
        from C_generators.C02_strategy_regex import RegexCandidateGenerator
        from C_generators.C03_strategy_layout import LayoutCandidateGenerator
        from C_generators.C04_strategy_flashtext import RegexLexiconGenerator
        from C_generators.C05_strategy_glossary import GlossaryTableCandidateGenerator

        gen_cfg = self.config.get("generators", {})
        regex_cfg = gen_cfg.get("regex_pattern", {})
        lexicon_cfg = gen_cfg.get("lexicon", {})

        lexicon_gen_config = {
            "run_id": self.run_id,
            "abbrev_lexicon_path": str(
                self.dict_path
                / self.lexicons.get(
                    "abbreviation_general", "2025_08_abbreviation_general.json"
                )
            ),
            "disease_lexicon_path": str(
                self.dict_path
                / self.lexicons.get("disease_lexicon", "2025_08_lexicon_disease.json")
            ),
            "orphanet_lexicon_path": str(
                self.dict_path
                / self.lexicons.get("orphanet_diseases", "2025_08_orphanet_diseases.json")
            ),
            "rare_disease_acronyms_path": str(
                self.dict_path
                / self.lexicons.get(
                    "rare_disease_acronyms", "2025_08_rare_disease_acronyms.json"
                )
            ),
            "umls_abbrev_path": str(
                self.dict_path
                / self.lexicons.get(
                    "umls_biological", "2025_08_umls_biological_abbreviations_v5.tsv"
                )
            ),
            "umls_clinical_path": str(
                self.dict_path
                / self.lexicons.get(
                    "umls_clinical", "2025_08_umls_clinical_abbreviations_v5.tsv"
                )
            ),
            "anca_lexicon_path": str(
                self.dict_path
                / self.lexicons.get("disease_lexicon_anca", "disease_lexicon_anca.json")
            ),
            "igan_lexicon_path": str(
                self.dict_path
                / self.lexicons.get("disease_lexicon_igan", "disease_lexicon_igan.json")
            ),
            "pah_lexicon_path": str(
                self.dict_path
                / self.lexicons.get("disease_lexicon_pah", "disease_lexicon_pah.json")
            ),
            "trial_acronyms_path": str(
                self.dict_path
                / self.lexicons.get("trial_acronyms", "trial_acronyms_lexicon.json")
            ),
            "pro_scales_path": str(
                self.dict_path / self.lexicons.get("pro_scales", "pro_scales_lexicon.json")
            ),
            "context_window": lexicon_cfg.get("context_window", 300),
        }

        return [
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

    def create_claude_client(
        self, model: Optional[str] = None
    ) -> Optional["ClaudeClient"]:
        """Create Claude API client if LLM validation is enabled."""
        if not self.use_llm_validation:
            return None

        from D_validation.D02_llm_engine import ClaudeClient
        return ClaudeClient(
            api_key=self.api_key,
            model=model or self.default_model,
            config_path=None,  # Config already loaded
        )

    def create_llm_engine(
        self, claude_client: Optional["ClaudeClient"], model: Optional[str] = None
    ) -> Optional["LLMEngine"]:
        """Create LLM validation engine."""
        if not self.use_llm_validation or claude_client is None:
            return None

        from D_validation.D02_llm_engine import LLMEngine
        return LLMEngine(
            llm_client=claude_client,
            model=model or self.default_model,
            run_id=self.run_id,
            max_tokens=self.val_cfg.get("max_tokens", 450),
            temperature=self.val_cfg.get("temperature", 0.0),
            top_p=self.val_cfg.get("top_p", 1.0),
        )

    def create_vlm_table_extractor(
        self, claude_client: Optional["ClaudeClient"]
    ) -> Optional["VLMTableExtractor"]:
        """Create VLM table extractor if enabled."""
        if not self.use_vlm_tables or claude_client is None:
            return None

        from C_generators.C15_vlm_table_extractor import VLMTableExtractor
        return VLMTableExtractor(
            llm_client=claude_client,
            llm_model=self.config.get("llm", {}).get("model", "claude-sonnet-4-20250514"),
            config={"run_id": self.run_id},
        )

    def create_validation_logger(self) -> "ValidationLogger":
        """Create validation logger."""
        from D_validation.D03_validation_logger import ValidationLogger
        return ValidationLogger(log_dir=str(self.log_dir), run_id=self.run_id)

    def create_term_mapper(self) -> "TermMapper":
        """Create term mapper for normalization."""
        from E_normalization.E01_term_mapper import TermMapper

        norm_cfg = self.config.get("normalization", {})
        term_mapper_cfg = norm_cfg.get("term_mapper", {})

        return TermMapper(
            config={
                "mapping_file_path": str(
                    self.dict_path
                    / self.lexicons.get(
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

    def create_disambiguator(self) -> "Disambiguator":
        """Create disambiguator for resolving ambiguous abbreviations."""
        from E_normalization.E02_disambiguator import Disambiguator

        norm_cfg = self.config.get("normalization", {})
        disambig_cfg = norm_cfg.get("disambiguator", {})

        return Disambiguator(
            config={
                "min_context_score": disambig_cfg.get("min_context_score", 2),
                "min_margin": disambig_cfg.get("min_margin", 1),
                "fill_long_form_for_orphans": disambig_cfg.get(
                    "fill_long_form_for_orphans", True
                ),
            }
        )

    def create_deduplicator(self) -> "Deduplicator":
        """Create deduplicator for merging duplicate entries."""
        from E_normalization.E07_deduplicator import Deduplicator

        norm_cfg = self.config.get("normalization", {})
        dedup_cfg = norm_cfg.get("deduplicator", {})

        return Deduplicator(
            config={
                "only_validated": dedup_cfg.get("only_validated", True),
                "store_alternatives": dedup_cfg.get("store_alternatives", True),
                "max_alternatives": dedup_cfg.get("max_alternatives", 5),
            }
        )

    def create_disease_detector(self) -> "DiseaseDetector":
        """Create disease detection component."""
        from C_generators.C06_strategy_disease import DiseaseDetector
        disease_cfg = self.config.get("disease_detection", {})
        return DiseaseDetector(
            config={
                "run_id": self.run_id,
                "lexicon_base_path": str(self.dict_path),
                "enable_general_lexicon": disease_cfg.get("enable_general_lexicon", True),
                "enable_orphanet": disease_cfg.get("enable_orphanet", True),
                "enable_mondo": disease_cfg.get("enable_mondo", True),
                "enable_rare_disease_acronyms": disease_cfg.get("enable_rare_disease_acronyms", True),
                "enable_scispacy": disease_cfg.get("enable_scispacy", True),
                "context_window": disease_cfg.get("context_window", 300),
            }
        )

    def create_disease_normalizer(self) -> "DiseaseNormalizer":
        """Create disease normalizer."""
        from E_normalization.E03_disease_normalizer import DiseaseNormalizer
        return DiseaseNormalizer()

    def create_drug_detector(self) -> "DrugDetector":
        """Create drug detection component."""
        from C_generators.C07_strategy_drug import DrugDetector
        return DrugDetector(
            config={
                "run_id": self.run_id,
                "lexicon_base_path": str(self.dict_path),
            }
        )

    def create_gene_detector(self) -> "GeneDetector":
        """Create gene detection component."""
        from C_generators.C16_strategy_gene import GeneDetector
        return GeneDetector(
            config={
                "run_id": self.run_id,
                "lexicon_base_path": str(self.dict_path),
            }
        )

    def create_pharma_detector(self) -> "PharmaCompanyDetector":
        """Create pharma company detection component."""
        from C_generators.C18_strategy_pharma import PharmaCompanyDetector
        return PharmaCompanyDetector(
            config={
                "run_id": self.run_id,
                "lexicon_base_path": str(self.dict_path),
            }
        )

    def create_author_detector(self) -> "AuthorDetector":
        """Create author detection component."""
        from C_generators.C13_strategy_author import AuthorDetector
        return AuthorDetector(
            config={
                "run_id": self.run_id,
                "pipeline_version": self.pipeline_version,
            }
        )

    def create_citation_detector(self) -> "CitationDetector":
        """Create citation detection component."""
        from C_generators.C14_strategy_citation import CitationDetector
        return CitationDetector(
            config={
                "run_id": self.run_id,
                "pipeline_version": self.pipeline_version,
            }
        )

    def create_feasibility_detector(self) -> "FeasibilityDetector":
        """Create feasibility detection component."""
        from C_generators.C08_strategy_feasibility import FeasibilityDetector
        return FeasibilityDetector(config={"run_id": self.run_id})

    def create_llm_feasibility_extractor(
        self, claude_client: Optional["ClaudeClient"]
    ) -> Optional["LLMFeasibilityExtractor"]:
        """Create LLM-based feasibility extractor if enabled."""
        if not self.use_llm_feasibility or claude_client is None:
            return None

        from C_generators.C11_llm_feasibility import LLMFeasibilityExtractor
        return LLMFeasibilityExtractor(
            llm_client=claude_client,
            llm_model=self.config.get("llm", {}).get("model", "claude-sonnet-4-20250514"),
            config={"run_id": self.run_id},
        )

    def create_epi_enricher(self) -> Optional["EpiExtractEnricher"]:
        """Create EpiExtract4GARD enricher if enabled."""
        if not self.use_epi_enricher:
            return None

        from E_normalization.E08_epi_extract_enricher import EpiExtractEnricher
        return EpiExtractEnricher(config={"run_id": self.run_id})

    def create_zeroshot_bioner(self) -> Optional["ZeroShotBioNEREnricher"]:
        """Create ZeroShotBioNER enricher if enabled."""
        if not self.use_zeroshot_bioner:
            return None

        from E_normalization.E09_zeroshot_bioner import ZeroShotBioNEREnricher
        return ZeroShotBioNEREnricher(config={"run_id": self.run_id})

    def create_biomedical_ner(self) -> Optional["BiomedicalNEREnricher"]:
        """Create BiomedicalNER enricher if enabled."""
        if not self.use_biomedical_ner:
            return None

        from E_normalization.E10_biomedical_ner_all import BiomedicalNEREnricher
        return BiomedicalNEREnricher(config={"run_id": self.run_id})

    def create_patient_journey_enricher(self) -> Optional["PatientJourneyEnricher"]:
        """Create patient journey enricher if enabled."""
        if not self.use_patient_journey:
            return None

        from E_normalization.E12_patient_journey_enricher import PatientJourneyEnricher
        return PatientJourneyEnricher(config={"run_id": self.run_id})

    def create_registry_enricher(self) -> Optional["RegistryEnricher"]:
        """Create registry enricher if enabled."""
        if not self.use_registry_extraction:
            return None

        from E_normalization.E13_registry_enricher import RegistryEnricher
        return RegistryEnricher(config={"run_id": self.run_id})

    def create_genetic_enricher(self) -> Optional["GeneticEnricher"]:
        """Create genetic enricher if enabled."""
        if not self.use_genetic_extraction:
            return None

        from E_normalization.E15_genetic_enricher import GeneticEnricher
        return GeneticEnricher(config={"run_id": self.run_id})

    def create_disease_enricher(self) -> Optional["DiseaseEnricher"]:
        """Create PubTator disease enricher if enabled."""
        if not self.use_pubtator_enrichment:
            return None

        from E_normalization.E04_pubtator_enricher import DiseaseEnricher
        return DiseaseEnricher(config={"run_id": self.run_id})

    def create_drug_enricher(self) -> Optional["DrugEnricher"]:
        """Create PubTator drug enricher if enabled."""
        if not self.use_pubtator_enrichment:
            return None

        from E_normalization.E05_drug_enricher import DrugEnricher
        return DrugEnricher(config={"run_id": self.run_id})

    def create_gene_enricher(self) -> Optional["GeneEnricher"]:
        """Create PubTator gene enricher if enabled."""
        if not self.use_pubtator_enrichment:
            return None

        from E_normalization.E18_gene_enricher import GeneEnricher
        return GeneEnricher(config={"run_id": self.run_id})

    def create_doc_metadata_strategy(
        self, claude_client: Optional["ClaudeClient"], model: Optional[str] = None
    ) -> "DocumentMetadataStrategy":
        """Create document metadata extraction strategy."""
        from C_generators.C09_strategy_document_metadata import DocumentMetadataStrategy

        doc_metadata_cfg = self.config.get("document_metadata", {})
        doc_types_path = doc_metadata_cfg.get(
            "document_types_path",
            str(self.dict_path / "2025_08_document_types.json")
        )

        return DocumentMetadataStrategy(
            document_types_path=doc_types_path if Path(doc_types_path).exists() else None,
            llm_client=claude_client,
            llm_model=model or self.default_model,
            run_id=self.run_id,
            pipeline_version=self.pipeline_version,
        )

    def create_nct_enricher(self) -> Optional["NCTEnricher"]:
        """Create NCT enricher if enabled."""
        nct_cfg = self.config.get("nct_enricher", {})
        if not nct_cfg.get("enabled", True):
            return None

        from E_normalization.E06_nct_enricher import NCTEnricher
        return NCTEnricher(nct_cfg)

    def load_rare_disease_lookup(self) -> Dict[str, str]:
        """Load rare disease acronyms for fallback lookup (SF->LF)."""
        from C_generators.C04_strategy_flashtext import (
            BAD_LONG_FORMS,
            WRONG_EXPANSION_BLACKLIST,
        )

        lookup: Dict[str, str] = {}
        rare_disease_path = self.dict_path / self.lexicons.get(
            "rare_disease_acronyms", "2025_08_rare_disease_acronyms.json"
        )

        if rare_disease_path.exists():
            try:
                data = json.loads(rare_disease_path.read_text(encoding="utf-8"))
                skipped_bad = 0
                for acronym, entry in data.items():
                    if isinstance(entry, dict) and entry.get("name"):
                        sf_key = acronym.strip().upper()
                        sf_lower = acronym.strip().lower()
                        lf_value = entry["name"].strip()
                        lf_lower = lf_value.lower()
                        # Filter out known wrong SF -> LF pairs and bad long forms
                        if (sf_lower, lf_lower) in WRONG_EXPANSION_BLACKLIST or lf_lower in BAD_LONG_FORMS:
                            skipped_bad += 1
                            continue
                        lookup[sf_key] = lf_value
                if skipped_bad > 0:
                    print(f"    [INFO] Skipped {skipped_bad} bad rare disease entries")
            except Exception as e:
                print(f"  [WARN] Failed to load rare disease lexicon: {e}")

        return lookup


__all__ = ["ComponentFactory"]

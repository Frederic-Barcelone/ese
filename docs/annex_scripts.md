# Annex: Script Inventory

> **Pipeline v0.8** — 199 Python scripts across 12 layers
> Generated 2026-02-07

---

## Layered Architecture Overview

### Naming Convention

Every script follows the pattern **`L##_descriptive_name.py`** where **L** is the layer letter and **##** is a two-digit sequence number. Numbering is not necessarily contiguous — gaps indicate deprecated or removed scripts. Two files share the number A13 (`ner_models` and `visual_models`), a legacy collision.

### Data Flow

```
PDF input
   │
   ▼
B_parsing ─────────► DocumentGraph (pages, blocks, tables, figures)
   │
   ▼
C_generators ──────► Candidates (high recall, noise acceptable)
   │
   ▼
D_validation ──────► Verified entities (LLM-filtered, high precision)
   │
   ▼
E_normalization ───► Enriched & deduplicated entities (ontology-mapped)
   │
   ▼
J_export ──────────► JSON output per entity type
```

**Supporting layers:**

| Layer | Role |
|-------|------|
| **A_core** | Domain models (Pydantic), interfaces, provenance, exceptions |
| **F_evaluation** | Gold standard loading, precision/recall/F1 scoring |
| **G_config** | `config.yaml` keys and type-safe access |
| **H_pipeline** | Component factory, abbreviation pipeline, merge resolution |
| **I_extraction** | High-level processors that orchestrate C → D → E for each entity |
| **K_tests** | 60 pytest files covering all layers |
| **Z_utils** | API client, text helpers, image utils, usage tracking, lexicon downloads |

### config.yaml Role

All runtime parameters live in `G_config/config.yaml`. Key sections: extraction presets, extractor toggles, API model tiers (17 call types routed to Haiku or Sonnet), path configuration, and cache settings. No CLI arguments — config.yaml is the single source of truth.

---

## A_core — Domain Models & Infrastructure (24 files)

Foundation layer defining all Pydantic models, abstract interfaces, provenance tracking, and shared utilities.

| # | File | Objective | Key Exports |
|---|------|-----------|-------------|
| A00 | `A00_logging.py` | Centralized logging with colored console output and rotating file logs | `get_logger`, `configure_logging`, `LogContext`, `timed`, `StepLogger`, `PipelineLogger` |
| A01 | `A01_domain_models.py` | Core Pydantic models for pipeline entities and audit trails | `PipelineStage`, `GeneratorType`, `ValidationStatus`, `EvidenceSpan`, `ProvenanceMetadata`, `Candidate`, `ExtractedEntity`, `AbbreviationCategory` |
| A02 | `A02_interfaces.py` | Abstract base classes for all pipeline components | `ExecutionContext`, `BaseExtractor`, `BaseCandidateGenerator`, `BaseVerifier`, `BaseNormalizer`, `BaseEnricher`, `BaseEvaluationMetric` |
| A03 | `A03_provenance.py` | Deterministic hashing and fingerprinting for reproducibility | `get_git_revision_hash`, `hash_bytes`, `hash_string`, `compute_doc_fingerprint`, `compute_prompt_hash`, `generate_run_id` |
| A04 | `A04_heuristics_config.py` | PASO abbreviation heuristics configuration and confidence calibration | `HeuristicsConfig`, `HeuristicsCounters`, `load_default_heuristics_config`, `calibrate_confidence`, `normalize_sf_key` |
| A05 | `A05_disease_models.py` | Disease entity models with ICD-10/SNOMED/ORPHA/MONDO codes | `DiseaseCandidate`, `ExtractedDisease`, `DiseaseIdentifier`, `DiseaseExportEntry`, `DiseaseExportDocument` |
| A06 | `A06_drug_models.py` | Drug entity models with RxNorm/MeSH/DrugBank and development phase | `DrugCandidate`, `ExtractedDrug`, `DrugIdentifier`, `DevelopmentPhase`, `DrugExportEntry` |
| A07 | `A07_feasibility_models.py` | Clinical trial feasibility models with computable criteria | `EligibilityCriterion`, `ScreeningFlow`, `StudyDesign`, `EpidemiologyData`, `FeasibilityCandidate`, `LabCriterion` |
| A08 | `A08_document_metadata_models.py` | Document-level metadata with classification and date extraction | `DocumentMetadata`, `PDFMetadata`, `DocumentClassification`, `DocumentDescription`, `DateExtractionResult` |
| A09 | `A09_pharma_models.py` | Pharmaceutical company entity models with HQ tracking | `PharmaCandidate`, `ExtractedPharma`, `PharmaExportEntry`, `PharmaExportDocument` |
| A10 | `A10_author_models.py` | Author/investigator models with ORCID identifiers | `AuthorCandidate`, `ExtractedAuthor`, `AuthorRoleType`, `AuthorExportEntry` |
| A11 | `A11_citation_models.py` | Citation/reference models with API validation support | `CitationCandidate`, `ExtractedCitation`, `CitationIdentifierType`, `CitationValidation` |
| A12 | `A12_exceptions.py` | Structured exception hierarchy for domain-specific errors | `ESEPipelineError`, `ConfigurationError`, `ParsingError`, `ExtractionError`, `APIError`, `RateLimitError` |
| A13a | `A13_ner_models.py` | Unified NER result models from multiple backends | `EntityCategory`, `NEREntity`, `NERResult`, `create_entity`, `merge_results` |
| A13b | `A13_visual_models.py` | Visual extraction models with VLM enrichment support | `VisualType`, `VisualCandidate`, `ExtractedVisual`, `TableStructure`, `CaptionCandidate`, `TriageResult` |
| A14 | `A14_extraction_result.py` | Universal extraction output contract with hash-based IDs | `EntityType`, `Provenance`, `ExtractionResult`, `compute_result_id`, `compute_regression_hash` |
| A15 | `A15_domain_profile.py` | Configurable domain profiles for extraction priors | `DomainProfile`, `ConfidenceAdjustments`, `load_domain_profile` (generic, nephrology, oncology, pulmonology) |
| A16 | `A16_pipeline_metrics.py` | Unified metrics tracking for pipeline observability | `PipelineMetrics`, `GenerationMetrics`, `ValidationMetrics`, `NormalizationMetrics`, `ExportMetrics` |
| A17 | `A17_care_pathway_models.py` | Clinical care pathway and treatment algorithm models | `CarePathwayNode`, `CarePathwayEdge`, `CarePathway`, `TaperSchedule` |
| A18 | `A18_recommendation_models.py` | Guideline recommendation models with evidence levels | `EvidenceLevel`, `RecommendationStrength`, `GuidelineRecommendation`, `RecommendationSet` |
| A19 | `A19_gene_models.py` | Gene entity models with HGNC/Entrez/Ensembl identifiers | `GeneCandidate`, `ExtractedGene`, `GeneIdentifier`, `GeneDiseaseLinkage`, `GeneExportEntry` |
| A20 | `A20_unicode_utils.py` | Unicode normalization for PDF text with mojibake handling | `HYPHENS_PATTERN`, `MOJIBAKE_MAP`, `normalize_sf`, `normalize_context`, `clean_long_form` |
| A21 | `A21_clinical_criteria.py` | Computable clinical eligibility criteria models | `LabCriterion`, `DiagnosisConfirmation`, `SeverityGrade`, `SEVERITY_GRADE_MAPPINGS` |
| A22 | `A22_logical_expressions.py` | Tree-based logical expressions for eligibility with SQL generation | `LogicalOperator`, `CriterionNode`, `LogicalExpression` |

---

## B_parsing — PDF Parsing & Document Structure (31 files)

Converts raw PDFs into structured `DocumentGraph` objects with layout detection, table extraction, figure extraction, and visual pipeline orchestration.

### Core PDF Parsing

| # | File | Objective | Key Exports |
|---|------|-----------|-------------|
| B01 | `B01_pdf_to_docgraph.py` | PDF to DocumentGraph parser with multi-column layout detection | `PDFToDocGraphParser`, `document_to_markdown`, `LayoutConfig` |
| B02 | `B02_doc_graph.py` | Pydantic data structures for parsed PDF document structure | `DocumentGraph`, `Page`, `TextBlock`, `Table`, `TableCell`, `ImageBlock`, `ContentRole` |

### Layout Detection & Reading Order

| # | File | Objective | Key Exports |
|---|------|-----------|-------------|
| B04 | `B04_column_ordering.py` | XY-Cut++ column layout detection and reading order | `order_page_blocks`, `detect_layout`, `LayoutConfig`, `LayoutType`, `ColumnOrderingMixin` |
| B05 | `B05_section_detector.py` | Layout-aware section detection for clinical documents | `SectionDetector`, `SectionInfo`, `SECTION_PATTERNS` |
| B18 | `B18_layout_models.py` | Pydantic models for layout-aware visual extraction | `LayoutPattern`, `VisualPosition`, `VisualZone`, `PageLayout` |
| B19 | `B19_layout_analyzer.py` | VLM-based layout analysis using Claude Vision | `analyze_page_layout`, `analyze_document_layouts`, `parse_vlm_response` |
| B25 | `B25_legacy_ordering.py` | Legacy block ordering for single/two-column layouts | `order_blocks_deterministically`, `is_two_column_page` |
| B29 | `B29_column_detection.py` | Column detection and gutter analysis utilities | `PageStats`, `Gutter`, `find_gutters`, `find_columns_by_clustering` |
| B30 | `B30_xy_cut_ordering.py` | XY-Cut++ recursive ordering algorithm | `xy_cut_order`, `order_body_bands`, `_find_best_cut` |

### Table Extraction

| # | File | Objective | Key Exports |
|---|------|-----------|-------------|
| B03 | `B03_table_extractor.py` | Table extraction using Docling TableFormer model | `TableExtractor`, `extract_tables_to_json`, `populate_document_graph` |
| B27 | `B27_table_validation.py` | Table validation and false positive filtering | `is_valid_table`, `is_prose_text_table`, `is_definition_table` |
| B28 | `B28_docling_backend.py` | Docling-based table extraction backend | `DoclingTableExtractor`, `DOCLING_AVAILABLE` |

### Figure Extraction

| # | File | Objective | Key Exports |
|---|------|-----------|-------------|
| B09 | `B09_pdf_native_figures.py` | Native PDF figure extraction using PyMuPDF | `extract_embedded_figures`, `detect_vector_figures`, `detect_all_figures` |
| B24 | `B24_native_figure_extraction.py` | Native figure extraction integration for DocumentGraph | `apply_native_figure_extraction`, `classify_image_type` |

### Visual Pipeline

| # | File | Objective | Key Exports |
|---|------|-----------|-------------|
| B10 | `B10_caption_detector.py` | Caption detection and column layout analysis | `detect_all_captions`, `infer_page_columns`, `link_caption_to_figure` |
| B11 | `B11_extraction_resolver.py` | Deterministic figure/table resolver with caption anchoring | `ResolvedFigure`, `ResolvedTable`, `resolve_all`, `resolve_tables` |
| B12 | `B12_visual_pipeline.py` | Visual extraction pipeline orchestrator | `VisualExtractionPipeline`, `PipelineConfig`, `PipelineResult` |
| B13 | `B13_visual_detector.py` | Visual detector with FAST/ACCURATE tiering | `detect_tables_with_docling`, `detect_all_visuals`, `DetectionResult` |
| B14 | `B14_visual_renderer.py` | Visual renderer with point-based padding and adaptive DPI | `RenderConfig`, `render_visual`, `render_table_as_image`, `render_full_page_from_path` |
| B15 | `B15_caption_extractor.py` | Multi-source caption extraction with provenance tracking | `extract_caption_multisource`, `extract_all_captions_on_page`, `infer_column_layout` |
| B16 | `B16_triage.py` | Visual triage for VLM processing requirements | `TriageConfig`, `TriageResult`, `TriageDecision`, `triage_batch`, `get_vlm_candidates` |
| B17 | `B17_document_resolver.py` | Document-level resolution for visual pipeline | `resolve_document`, `scan_body_text_references`, `merge_multipage_visuals` |
| B20 | `B20_zone_expander.py` | Whitespace-based bounding box computation from VLM zones | `expand_zones_to_bboxes`, `compute_column_boundaries`, `find_whitespace_boundaries` |
| B21 | `B21_filename_generator.py` | Layout-aware filename generation for visual exports | `generate_visual_filename`, `sanitize_name`, `parse_visual_filename` |
| B22 | `B22_doclayout_detector.py` | DocLayout-YOLO visual detector for figures/tables | `detect_visuals_doclayout`, `generate_vlm_description`, `DOCLAYOUT_CATEGORIES` |
| B31 | `B31_vlm_detector.py` | VLM-assisted visual detection using Claude Vision | `detect_page_visuals`, `detect_document_visuals`, `VLM_DETECTION_PROMPT` |

### Text Analysis Utilities

| # | File | Objective | Key Exports |
|---|------|-----------|-------------|
| B06 | `B06_confidence.py` | Feature-based confidence scoring for entity extraction | `ConfidenceCalculator`, `ContradictionDetector`, `UnifiedConfidenceCalculator` |
| B07 | `B07_negation.py` | Negation and assertion detection for clinical text | `NegationDetector`, `AssertionClassifier`, `is_negated` |
| B08 | `B08_eligibility_parser.py` | Eligibility criteria logical expression parser | `EligibilityParser`, `LogicalExpression`, `parse_eligibility` |
| B23 | `B23_text_helpers.py` | Text normalization, cleaning, and pattern utilities | `normalize_repeated_text`, `clean_text`, `table_to_markdown`, `bbox_overlaps` |
| B26 | `B26_repetition_inference.py` | Header/footer detection using repetition across pages | `infer_repeated_headers_footers`, `is_running_header` |

---

## C_generators — Candidate Generation (35 files)

High-recall extraction layer producing candidates via lexicons (FlashText), regex, layout analysis, and LLM/VLM. False positives are acceptable — precision comes from D_validation.

### Core Entity Strategies

| # | File | Objective | Key Exports |
|---|------|-----------|-------------|
| C06 | `C06_strategy_disease.py` | Multi-layer disease detection with FlashText and FP filtering | `DiseaseDetector` |
| C07 | `C07_strategy_drug.py` | Multi-layer drug detection with prioritized lexicons | `DrugDetector` |
| C16 | `C16_strategy_gene.py` | Multi-layer gene detection with Orphadata and HGNC | `GeneDetector` |
| C13 | `C13_strategy_author.py` | Author/investigator name detection with roles and affiliations | `AuthorDetector` |
| C14 | `C14_strategy_citation.py` | PMID, DOI, NCT identifier extraction from text | `CitationDetector` |
| C18 | `C18_strategy_pharma.py` | Pharmaceutical company mention detection via lexicon | `PharmaCompanyDetector` |

### General Extraction Strategies

| # | File | Objective | Key Exports |
|---|------|-----------|-------------|
| C00 | `C00_strategy_identifiers.py` | Standardized database identifier extraction | `IdentifierExtractor`, `IdentifierType`, `IDENTIFIER_PATTERNS` |
| C01 | `C01_strategy_abbrev.py` | Schwartz-Hearst abbreviation extraction with pattern variants | `SyntaxMatcherGenerator` |
| C02 | `C02_strategy_regex.py` | Rigid pattern matching for trial IDs, doses, dates | `RegexPatternGenerator`, `ReferenceType` |
| C03 | `C03_strategy_layout.py` | Spatial extraction using document coordinates and zones | `LayoutExtractor` |
| C04 | `C04_strategy_flashtext.py` | FlashText lexicon matching with scispacy NER integration | `RegexLexiconGenerator`, `LexiconLoaderMixin` |
| C05 | `C05_strategy_glossary.py` | SF/LF pair extraction from glossary tables | `GlossaryTableCandidateGenerator` |
| C09 | `C09_strategy_document_metadata.py` | File system and PDF metadata with LLM classification | `DocumentMetadataExtractor` |

### Abbreviation Utilities

| # | File | Objective | Key Exports |
|---|------|-----------|-------------|
| C20 | `C20_abbrev_patterns.py` | Pattern constants and helpers for abbreviation extraction | `_ABBREV_TOKEN_RE`, `_is_likely_author_initial`, `_looks_like_short_form` |
| C21 | `C21_noise_filters.py` | Noise filtering constants for abbreviation extraction | `OBVIOUS_NOISE`, `WRONG_EXPANSION_BLACKLIST`, `BAD_LONG_FORMS`, `LexiconEntry` |
| C22 | `C22_lexicon_loaders.py` | Mixin methods for loading lexicons into FlashText | `LexiconLoaderMixin` |
| C23 | `C23_inline_definition_detector.py` | Inline abbreviation definition pattern detection | `InlineDefinitionDetectorMixin` |

### False Positive Filters

| # | File | Objective | Key Exports |
|---|------|-----------|-------------|
| C24 | `C24_disease_fp_filter.py` | Confidence-based disease false positive filtering | `DiseaseFalsePositiveFilter`, `CHROMOSOME_PATTERNS` |
| C25 | `C25_drug_fp_filter.py` | Drug FP filtering with curated exclusion sets | `DrugFalsePositiveFilter`, `DRUG_ABBREVIATIONS` |
| C26 | `C26_drug_fp_constants.py` | Curated constant sets for drug FP filtering | `BACTERIA_ORGANISMS`, `BIOLOGICAL_ENTITIES`, `COMMON_WORDS`, `CONSUMER_DRUG_VARIANTS` |
| C34 | `C34_gene_fp_filter.py` | Context-aware gene false positive filtering | `GeneFalsePositiveFilter`, `STATISTICAL_TERMS` |

### Feasibility Extraction

| # | File | Objective | Key Exports |
|---|------|-----------|-------------|
| C08 | `C08_strategy_feasibility.py` | Pattern-based clinical trial feasibility extraction | `FeasibilityDetector` |
| C11 | `C11_llm_feasibility.py` | LLM-based structured feasibility extraction with quote verification | `LLMFeasibilityExtractor`, `FeasibilityResponseParserMixin` |
| C27 | `C27_feasibility_patterns.py` | Regex patterns for feasibility data extraction | `EPIDEMIOLOGY_ANCHORS`, `INCLUSION_MARKERS`, `ENDPOINT_PATTERNS` |
| C28 | `C28_feasibility_fp_filter.py` | Multi-layer feasibility false positive filtering | `FeasibilityFalsePositiveFilter` |
| C29 | `C29_feasibility_prompts.py` | LLM prompt templates for feasibility extraction | `SECTION_TARGETS`, extraction prompts |
| C30 | `C30_feasibility_response_parser.py` | Parse LLM JSON responses into feasibility candidates | `FeasibilityResponseParserMixin` |

### Guideline Recommendations

| # | File | Objective | Key Exports |
|---|------|-----------|-------------|
| C12 | `C12_guideline_recommendation_extractor.py` | Extract guideline recommendations with LoE/SoR grading | `GuidelineRecommendationExtractor` |
| C31 | `C31_recommendation_patterns.py` | Patterns and prompts for recommendation extraction | `ORGANIZATION_PATTERNS`, `EVIDENCE_PATTERNS`, `RECOMMENDATION_EXTRACTION_PROMPT` |
| C32 | `C32_recommendation_llm.py` | LLM-based recommendation extraction mixin | `LLMExtractionMixin` |
| C33 | `C33_recommendation_vlm.py` | VLM-based LoE/SoR extraction from tables | `VLMExtractionMixin` |

### Vision / VLM Strategies

| # | File | Objective | Key Exports |
|---|------|-----------|-------------|
| C10 | `C10_vision_image_analysis.py` | Claude Vision for CONSORT flowcharts and clinical figures | `VisionImageAnalyzer`, `PatientFlowData` |
| C15 | `C15_vlm_table_extractor.py` | VLM-based table structure extraction from images | `VLMTableExtractor`, `resize_image_for_vlm` |
| C17 | `C17_flowchart_graph_extractor.py` | Care pathway decision graph extraction from algorithms | `FlowchartGraphExtractor` |
| C19 | `C19_vlm_visual_enrichment.py` | VLM enrichment for figure/table classification | `VLMVisualEnricher`, `VLMConfig`, `VLMClassificationResult` |

---

## D_validation — LLM Verification (4 files)

High-precision filtering layer using Claude LLM to verify candidates from C_generators.

| # | File | Objective | Key Exports |
|---|------|-----------|-------------|
| D01 | `D01_prompt_registry.py` | Centralized versioned prompt store for validation tasks | `PromptTask`, `PromptBundle`, `PromptRegistry`, `compute_prompt_bundle_hash` |
| D02 | `D02_llm_engine.py` | LLM engine using Claude API for candidate verification | `ClaudeClient`, `LLMEngine`, `VerificationResult`, `resolve_model_tier`, `record_api_usage`, `MODEL_PRICING` |
| D03 | `D03_validation_logger.py` | Structured JSONL logging for validation results | `ValidationLogger` |
| D04 | `D04_quote_verifier.py` | Anti-hallucination verification for LLM-extracted quotes | `QuoteVerifier`, `NumericalVerifier`, `ExtractionVerifier`, `QuoteVerificationResult` |

---

## E_normalization — Enrichment & Deduplication (18 files)

Maps extracted entities to standard ontologies, enriches via external APIs, and deduplicates.

### Term Mapping & Disambiguation

| # | File | Objective | Key Exports |
|---|------|-----------|-------------|
| E01 | `E01_term_mapper.py` | Abbreviation normalization and canonical ID attachment | `TermMapper` |
| E02 | `E02_disambiguator.py` | Abbreviation disambiguation using document context voting | `Disambiguator` |
| E03 | `E03_disease_normalizer.py` | Disease normalization with therapeutic area categorization | `DiseaseNormalizer`, `CATEGORY_KEYWORDS` |

### External API Enrichment

| # | File | Objective | Key Exports |
|---|------|-----------|-------------|
| E04 | `E04_pubtator_enricher.py` | PubTator3 API for disease enrichment with MeSH codes | `PubTator3Client`, `DiseaseEnricher` |
| E05 | `E05_drug_enricher.py` | PubTator3 for drug enrichment with MeSH identifiers | `DrugEnricher` |
| E06 | `E06_nct_enricher.py` | ClinicalTrials.gov API for NCT identifier enrichment | `NCTEnricher`, `NCTTrialInfo`, `NCTClient` |
| E14 | `E14_citation_validator.py` | Citation validation via DOI/NCT/PMID external APIs | `CitationValidator`, `CitationValidationReport` |
| E18 | `E18_gene_enricher.py` | PubTator3 for gene enrichment with Entrez IDs | `GeneEnricher` |

### NER & Biomedical Enrichment

| # | File | Objective | Key Exports |
|---|------|-----------|-------------|
| E08 | `E08_epi_extract_enricher.py` | EpiExtract4GARD BioBERT model for rare disease epidemiology | `EpiExtractEnricher` |
| E09 | `E09_zeroshot_bioner.py` | Zero-shot BioNER for ADE and drug administration extraction | `ZeroShotBioNEREnricher`, `ENTITY_TYPES_TO_EXTRACT` |
| E10 | `E10_biomedical_ner_all.py` | d4data/biomedical-ner-all DistilBERT for 84 clinical entity types | `BiomedicalNEREnricher`, `ENTITY_CATEGORIES` |
| E15 | `E15_genetic_enricher.py` | Genetic variant and HPO phenotype extraction via regex | `GeneticEnricher`, `HGVS_CODING_PATTERN` |

### Feasibility-Specific Enrichment

| # | File | Objective | Key Exports |
|---|------|-----------|-------------|
| E12 | `E12_patient_journey_enricher.py` | Patient journey extraction for trial feasibility | `PatientJourneyEnricher`, `PATIENT_JOURNEY_LABELS` |
| E13 | `E13_registry_enricher.py` | Patient registry information for feasibility analysis | `RegistryEnricher`, `KNOWN_REGISTRIES` |

### Deduplication

| # | File | Objective | Key Exports |
|---|------|-----------|-------------|
| E07 | `E07_deduplicator.py` | Abbreviation deduplication with quality-based long form selection | `Deduplicator` |
| E11 | `E11_span_deduplicator.py` | Multi-source NER span deduplication with confidence selection | `SpanDeduplicator`, `NERSpan` |
| E16 | `E16_drug_combination_parser.py` | Drug combination string parsing into individual components | `parse_drug_combination`, `DrugComponent` |
| E17 | `E17_entity_deduplicator.py` | Generic deduplication for diseases, drugs, genes | `deduplicate_diseases`, `deduplicate_drugs`, `deduplicate_genes` |

---

## F_evaluation — Gold Standard Scoring (3 files)

Evaluation framework for measuring extraction accuracy against human-annotated gold standards.

| # | File | Objective | Key Exports |
|---|------|-----------|-------------|
| F01 | `F01_gold_loader.py` | Gold standard loader for annotated ground truth | `GoldLoader`, `GoldAnnotation`, `GoldStandard` |
| F02 | `F02_scorer.py` | Precision/recall/F1 scorer for extraction evaluation | `Scorer`, `ScorerConfig`, `ScoreReport` |
| F03 | `F03_evaluation_runner.py` | End-to-end evaluation runner against NLP4RARE, NLM-Gene, RareDisGene | Config constants: `RUN_NLP4RARE`, `RUN_NLM_GENE`, `RUN_RAREDIS_GENE`, `MAX_DOCS` |

---

## G_config — Configuration (1 file)

| # | File | Objective | Key Exports |
|---|------|-----------|-------------|
| G01 | `G01_config_keys.py` | Type-safe configuration key enums with defaults | `ConfigKeyBase`, `ConfigKey`, `CacheConfig`, `ParserConfig` |

---

## H_pipeline — Pipeline Orchestration (4 files)

Component factory, abbreviation pipeline with PASO heuristics, and merge resolution.

| # | File | Objective | Key Exports |
|---|------|-----------|-------------|
| H01 | `H01_component_factory.py` | Centralized component initialization for pipeline | `ComponentFactory` |
| H02 | `H02_abbreviation_pipeline.py` | Abbreviation extraction pipeline with PASO heuristic rules | `AbbreviationPipeline` |
| H03 | `H03_visual_integration.py` | Visual pipeline integration for tables and figures | `VisualPipelineIntegration`, `create_visual_integration` |
| H04 | `H04_merge_resolver.py` | Deterministic merge and conflict resolution for duplicates | `MergeConfig`, `MergeResolver`, `get_merge_resolver` |

---

## I_extraction — Entity Processors (2 files)

High-level processors that orchestrate the C → D → E flow for each entity type.

| # | File | Objective | Key Exports |
|---|------|-----------|-------------|
| I01 | `I01_entity_processors.py` | Entity extraction coordinator for diseases, drugs, genes, authors, citations | `EntityProcessor` |
| I02 | `I02_feasibility_processor.py` | Feasibility extraction processor with LLM and NER enrichers | `FeasibilityProcessor` |

---

## J_export — JSON Export (4 files)

Serialization layer converting internal models to JSON output files.

| # | File | Objective | Key Exports |
|---|------|-----------|-------------|
| J01 | `J01_export_handlers.py` | Central export manager for all pipeline results | `ExportManager`, `render_figure_with_padding` |
| J01a | `J01a_entity_exporters.py` | Export functions for diseases, genes, drugs, pharma, authors, citations | `export_disease_results`, `export_gene_results`, `export_drug_results`, `export_author_results` |
| J01b | `J01b_metadata_exporters.py` | Export functions for document metadata, care pathways, recommendations | `export_document_metadata`, `export_care_pathways`, `export_recommendations` |
| J02 | `J02_visual_export.py` | Visual extraction export for tables and figures | `visual_to_dict`, `export_visuals_to_json`, `export_tables_only`, `export_figures_only` |

---

## K_tests — Test Suite (60 files)

Comprehensive pytest suite covering all pipeline layers.

### A_core Tests

| # | File | Tests |
|---|------|-------|
| K01 | `K01_base_provenance.py` | BaseProvenanceMetadata generic class tests |
| K02 | `K02_disease_provenance_migration.py` | DiseaseProvenanceMetadata inheritance from BaseProvenanceMetadata |
| K04 | `K04_exceptions.py` | Exception hierarchy and error handling |
| K05 | `K05_extraction_result.py` | ExtractionResult immutability and ID stability |
| K06 | `K06_ner_models.py` | Unified NER data models |
| K51 | `K51_test_provenance.py` | Hashing, fingerprinting, run ID generation |
| K52 | `K52_test_author_models.py` | Author model validation and enums |
| K53 | `K53_test_citation_models.py` | Citation model validation |
| K54 | `K54_test_unicode_utils.py` | Unicode normalization and mojibake fixing |
| K55 | `K55_test_domain_profile.py` | Domain profiles and confidence adjustments |
| K56 | `K56_test_pipeline_metrics.py` | Pipeline metrics tracking and invariants |
| K57 | `K57_test_clinical_criteria.py` | Clinical criteria models and severity grades |

### B_parsing Tests

| # | File | Tests |
|---|------|-------|
| K07 | `K07_visual_models_layout.py` | Layout-aware fields in visual models |
| K08 | `K08_docling_backend.py` | Docling table extraction backend |
| K09 | `K09_filename_generator.py` | Layout-aware filename generation |
| K10 | `K10_layout_analyzer.py` | VLM layout analyzer response parsing |
| K11 | `K11_layout_e2e.py` | End-to-end layout-aware visual extraction |
| K12 | `K12_layout_models.py` | Layout pattern enums |
| K13 | `K13_visual_pipeline_integration.py` | Layout-aware visual pipeline integration |
| K14 | `K14_zone_expander.py` | Zone expander whitespace-based bbox computation |
| K17 | `K17_caption_extractor.py` | Multi-source caption extraction |
| K18 | `K18_detector.py` | Visual detector with FAST/ACCURATE tiering |
| K19 | `K19_image_extraction.py` | Image extraction coordinate and DPI handling |
| K20 | `K20_renderer.py` | Visual renderer with coordinate conversion |
| K21 | `K21_triage.py` | Visual triage logic |
| K22 | `K22_visual_models.py` | Visual extraction data models |

### C_generators Tests

| # | File | Tests |
|---|------|-------|
| K23 | `K23_test_abbrev_patterns.py` | Abbreviation pattern matching and extraction |
| K24 | `K24_test_noise_filters.py` | Noise filtering constants and validation |
| K25 | `K25_test_inline_definition_detector.py` | Inline abbreviation definition detection |
| K26 | `K26_test_disease_fp_filter.py` | Disease false positive filter scoring |
| K27 | `K27_test_drug_fp_filter.py` | Drug false positive filtering |
| K28 | `K28_test_feasibility_patterns.py` | Clinical trial feasibility extraction patterns |
| K29 | `K29_test_gene_fp_filter.py` | Gene false positive filtering for ambiguous symbols |
| K30 | `K30_test_lexicon_loaders.py` | Lexicon loading mixin methods |
| K31 | `K31_test_c_generators_imports.py` | C_generators module import verification |

### D_validation Tests

| # | File | Tests |
|---|------|-------|
| K32 | `K32_test_prompt_registry.py` | Prompt registry, task enums, bundle generation |
| K33 | `K33_test_llm_engine.py` | LLM engine, Claude client, model tier routing |
| K34 | `K34_test_validation_logger.py` | Validation logging and statistics tracking |
| K35 | `K35_test_quote_verifier.py` | Quote verification and extraction verification |

### E_normalization Tests

| # | File | Tests |
|---|------|-------|
| K36 | `K36_test_term_mapper.py` | Term mapping and normalization |
| K37 | `K37_test_deduplicator.py` | Abbreviation deduplication with quality selection |
| K38 | `K38_test_span_deduplicator.py` | NER span deduplication from multiple sources |
| K39 | `K39_test_entity_deduplicator.py` | Entity deduplication (diseases, drugs, genes) |

### F_evaluation Tests

| # | File | Tests |
|---|------|-------|
| K40 | `K40_test_gold_loader.py` | Gold standard loading and annotation parsing |
| K41 | `K41_test_scorer.py` | Precision/recall/F1 scoring |

### G/H/I/J Tests

| # | File | Tests |
|---|------|-------|
| K43 | `K43_test_config_keys.py` | Configuration key enums and helpers |
| K44 | `K44_test_merge_resolver.py` | Deterministic merge and conflict resolution |
| K45 | `K45_test_component_factory.py` | ComponentFactory initialization |
| K46 | `K46_test_entity_processors.py` | EntityProcessor and entity creation helpers |
| K47 | `K47_test_feasibility_processor.py` | FeasibilityProcessor and NER enrichment |
| K48 | `K48_test_export_handlers.py` | ExportManager and export functionality |
| K49 | `K49_test_visual_export.py` | Visual export functions and serialization |

### Z_utils Tests

| # | File | Tests |
|---|------|-------|
| K03 | `K03_api_client.py` | Shared API client (cache, rate limiter) |
| K15 | `K15_orchestrator_utils.py` | Orchestrator utilities (timing, tee logging) |
| K16 | `K16_path_utils.py` | Path utilities and base path detection |
| K58 | `K58_test_text_helpers.py` | Text processing helpers for abbreviations |
| K59 | `K59_test_text_normalization.py` | Text normalization for PDF extraction cleanup |

### Cross-Layer Tests

| # | File | Tests |
|---|------|-------|
| K42 | `K42_test_d_e_f_imports.py` | D/E/F layer module import verification |
| K50 | `K50_test_g_h_i_j_imports.py` | G/H/I/J layer module import verification |
| K60 | `K60_test_cross_entity_filter.py` | Cross-entity filtering logic |

---

## Z_utils — Shared Utilities (11 files)

Cross-cutting infrastructure: API client, text processing, image handling, usage tracking, lexicon downloads.

### Core Utilities

| # | File | Objective | Key Exports |
|---|------|-----------|-------------|
| Z01 | `Z01_api_client.py` | Shared API client with disk caching and rate limiting | `BaseAPIClient`, `DiskCache`, `RateLimiter`, `retry_with_exponential_backoff` |
| Z02 | `Z02_text_helpers.py` | Text processing helpers for normalization and context | `extract_context_snippet`, `normalize_lf_for_dedup`, `compute_candidate_quality_score` |
| Z03 | `Z03_text_normalization.py` | Text normalization for whitespace and dehyphenation | `clean_whitespace`, `dehyphenate_long_form`, `normalize_long_form` |
| Z04 | `Z04_image_utils.py` | Image utilities for Vision LLM and compression | `get_image_size_bytes`, `is_image_oversized`, `compress_image_to_limit` |
| Z05 | `Z05_path_utils.py` | Path resolution with environment variable support | `get_base_path` |
| Z06 | `Z06_usage_tracker.py` | SQLite-based LLM usage tracking | `UsageTracker` |
| Z07 | `Z07_console_output.py` | Console output with ANSI colors and progress indicators | `Colors`, `C`, `print_step`, `print_substep` |
| Z11 | `Z11_entity_helpers.py` | Shared entity creation helpers from candidates and matches | `create_entity_from_candidate`, `create_entity_from_search` |

### Lexicon Download Scripts

| # | File | Objective | Key Exports |
|---|------|-----------|-------------|
| Z08 | `Z08_download_utils.py` | Shared utilities for downloading lexicons | `download_file`, `get_default_output_dir` |
| Z09 | `Z09_download_gene_lexicon.py` | Download and build gene lexicon from Orphadata/HGNC | `parse_orphadata_genes`, `parse_hgnc_tsv`, `build_gene_lexicon` |
| Z10 | `Z10_download_lexicons.py` | Download Meta-Inventory, MONDO, ChEMBL lexicons | `download_meta_inventory`, `download_mondo`, `download_chembl` |

---

## Root-Level Entry Points (2 files)

| File | Objective | Key Exports |
|------|-----------|-------------|
| `orchestrator.py` | Main pipeline entry point coordinating all extraction stages | _(execution script — no importable exports)_ |
| `orchestrator_utils.py` | Orchestrator timing, warning suppression, tee logging | `StageTimer`, `setup_warning_suppression`, `activate_tee`, `deactivate_tee` |

---

## Summary

| Layer | Files | Purpose |
|-------|------:|---------|
| A_core | 24 | Domain models, interfaces, provenance, exceptions |
| B_parsing | 31 | PDF → DocumentGraph, layout, tables, figures, visual pipeline |
| C_generators | 35 | High-recall candidate generation (lexicons, regex, LLM, VLM) |
| D_validation | 4 | LLM-based verification for precision |
| E_normalization | 18 | Ontology mapping, API enrichment, deduplication |
| F_evaluation | 3 | Gold standard loading and P/R/F1 scoring |
| G_config | 1 | Type-safe config key definitions |
| H_pipeline | 4 | Component factory, abbreviation pipeline, merge resolution |
| I_extraction | 2 | High-level entity and feasibility processors |
| J_export | 4 | JSON serialization for all entity types |
| K_tests | 60 | Comprehensive pytest suite |
| Z_utils | 11 | API client, text helpers, image utils, lexicon downloads |
| Root | 2 | Orchestrator entry point and utilities |
| **Total** | **199** | |

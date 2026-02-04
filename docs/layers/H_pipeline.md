# H_pipeline -- Pipeline Orchestration

## Purpose

Layer H coordinates the assembly and execution of pipeline components. It provides a factory for constructing all components from configuration, a dedicated pipeline for abbreviation extraction with PASO heuristics, a visual extraction integration wrapper, and a deterministic merge/resolve layer for deduplicating outputs from multiple strategies.

## Modules

### H01_component_factory.py

Centralized factory for creating and configuring all pipeline components from YAML configuration.

**Class: `ComponentFactory`**

```python
factory = ComponentFactory(config, run_id, pipeline_version, log_dir, api_key)
```

**Constructor parameters:** `config` (dict from YAML), `run_id`, `pipeline_version`, `log_dir` (Path), `api_key` (optional).

**Initialization behavior:** Reads `paths`, `lexicons`, `api.claude`, and `extraction_pipeline.options` from config. Resolves `base_path` via `get_base_path()` (CORPUS_BASE_PATH env var or auto-detect).

**Factory methods:**

| Method | Returns | Source module |
|--------|---------|--------------|
| `create_parser()` | `PDFToDocGraphParser` | B_parsing.B01 |
| `create_table_extractor()` | `Optional[TableExtractor]` | B_parsing.B03 (None if Docling unavailable) |
| `create_generators()` | `List` of 5 generators | C01-C05 (syntax, glossary, regex, layout, flashtext) |
| `create_claude_client(model)` | `Optional[ClaudeClient]` | D_validation.D02 |
| `create_llm_engine(claude_client, model)` | `Optional[LLMEngine]` | D_validation.D02 |
| `create_vlm_table_extractor(claude_client)` | `Optional[VLMTableExtractor]` | C_generators.C15 |
| `create_validation_logger()` | `ValidationLogger` | D_validation.D03 |
| `create_term_mapper()` | `TermMapper` | E_normalization.E01 |
| `create_disambiguator()` | `Disambiguator` | E_normalization.E02 |
| `create_deduplicator()` | `Deduplicator` | E_normalization.E07 |
| `create_disease_detector()` | `DiseaseDetector` | C_generators.C06 |
| `create_disease_normalizer()` | `DiseaseNormalizer` | E_normalization.E03 |
| `create_drug_detector()` | `DrugDetector` | C_generators.C07 |
| `create_gene_detector()` | `GeneDetector` | C_generators.C16 |
| `create_pharma_detector()` | `PharmaCompanyDetector` | C_generators.C18 |
| `create_author_detector()` | `AuthorDetector` | C_generators.C13 |
| `create_citation_detector()` | `CitationDetector` | C_generators.C14 |
| `create_feasibility_detector()` | `FeasibilityDetector` | C_generators.C08 |
| `create_llm_feasibility_extractor(claude_client)` | `Optional[LLMFeasibilityExtractor]` | C_generators.C11 |
| `create_epi_enricher()` | `Optional[EpiExtractEnricher]` | E_normalization.E08 |
| `create_zeroshot_bioner()` | `Optional[ZeroShotBioNEREnricher]` | E_normalization.E09 |
| `create_biomedical_ner()` | `Optional[BiomedicalNEREnricher]` | E_normalization.E10 |
| `create_patient_journey_enricher()` | `Optional[PatientJourneyEnricher]` | E_normalization.E12 |
| `create_registry_enricher()` | `Optional[RegistryEnricher]` | E_normalization.E13 |
| `create_genetic_enricher()` | `Optional[GeneticEnricher]` | E_normalization.E15 |
| `create_disease_enricher()` | `Optional[DiseaseEnricher]` | E_normalization.E04 (PubTator) |
| `create_drug_enricher()` | `Optional[DrugEnricher]` | E_normalization.E05 (PubTator) |
| `create_gene_enricher()` | `Optional[GeneEnricher]` | E_normalization.E18 (PubTator) |
| `create_doc_metadata_strategy(claude_client, model)` | `DocumentMetadataStrategy` | C_generators.C09 |
| `create_nct_enricher()` | `Optional[NCTEnricher]` | E_normalization.E06 |
| `load_rare_disease_lookup()` | `Dict[str, str]` | SF-to-LF dictionary from rare disease lexicon |

All enricher/detector methods return `None` when the corresponding feature is disabled in config.

### H02_abbreviation_pipeline.py

Dedicated pipeline for abbreviation extraction with multi-stage workflow and PASO heuristics.

**Class: `AbbreviationPipeline`**

```python
pipeline = AbbreviationPipeline(
    run_id, pipeline_version, parser, table_extractor,
    generators, heuristics, term_mapper, disambiguator,
    deduplicator, logger, claude_client, llm_engine,
    vlm_table_extractor, nct_enricher, rare_disease_lookup,
    use_vlm_tables, use_normalization, model,
)
```

**Pipeline stages:**

| Method | Stage | Description |
|--------|-------|-------------|
| `parse_pdf(pdf_path, output_dir)` | 1 | Parse PDF into `DocumentGraph`, extract tables |
| `generate_candidates(doc)` | 2 | Run all generators, deduplicate by normalized (SF, LF) |
| `filter_candidates(candidates, full_text)` | 3 | Corroboration check, lexicon reduction, SF form validation |
| `apply_heuristics(candidates, counters)` | 4 | Auto-approve/reject via PASO rules, returns `(auto_results, llm_candidates)` |
| `validate_with_llm(llm_candidates, batch_delay_ms)` | 5 | Fast-reject pre-screening (Haiku) + batch LLM validation (explicit pairs batch=10, lexicon batch=15) |
| `search_missing_abbreviations(doc_id, full_text, found_sfs, counters)` | 6 | PASO C (hyphenated) + direct search abbreviations |
| `extract_sf_only_with_llm(doc_id, full_text, found_sfs, counters)` | 7 | PASO D: LLM SF-only extraction with lexicon fallback. Uses `call_type="sf_only_extraction"` (Haiku tier). |
| `normalize_results(results, full_text)` | 8 | Normalize, NCT enrich, disambiguate, re-disambiguate, deduplicate |

**PASO heuristics applied in `apply_heuristics()`:**

- Auto-reject: DOI patterns, blacklisted SFs, Figure/Table references, author initials, context mismatch, trial IDs, common words, malformed long forms (partial starters, unclosed brackets)
- Auto-approve: Stats abbreviations with numeric evidence (PASO A), country codes

### H03_visual_integration.py

Integration wrapper for the visual extraction pipeline (tables and figures).

**Class: `VisualPipelineIntegration`**

```python
integration = VisualPipelineIntegration(config)
if integration.enabled:
    result = integration.extract(pdf_path)  # Returns Optional[PipelineResult]
    exported = integration.export(result, output_dir, doc_name, save_images_as_files=True)
```

**Initialization:** Reads `extraction_pipeline.visual_extraction` from config. Checks availability of Docling, PyMuPDF, and Anthropic SDK. Disables itself if PyMuPDF is missing.

**Configuration mapping:** Builds `PipelineConfig` from YAML sections: `visual_detection` (mode, model), `detection` (table mode, escalation, noise filtering), `rendering` (DPI, padding), `triage` (area ratios, repeat threshold), `vlm` (enabled, model, validate_tables), `resolution` (merge multipage, deduplicate).

**Detection modes:** `layout-aware`, `vlm-only`, `hybrid`, `heuristic`.

**Factory function:** `create_visual_integration(config)` returns a `VisualPipelineIntegration` instance.

### H04_merge_resolver.py

Deterministic deduplication and conflict resolution for overlapping extractions from multiple strategies.

**Classes:**

- `MergeConfig` -- Configuration dataclass:
  - `dedupe_key_fields`: tuple of fields forming the dedup key (default: `doc_id`, `field_name`, `value`, `normalized_value`)
  - `strategy_priority`: dict mapping strategy IDs to priority integers (higher wins)
  - `mutually_exclusive_fields`: groups of fields where only one may be present
  - `prefer_table_evidence: bool = True`
  - `prefer_longer_evidence: bool = True`
  - `MergeConfig.default()` -- Factory with preset priorities (disease specialized=10, orphanet=5, general=1; abbreviation syntax=10, glossary=8, lexicon=5)

- `MergeResolver` -- Main resolver class.

**Public API:**

```python
resolver = MergeResolver(MergeConfig.default())
merged = resolver.merge(raw_extractions)

# Or use singleton
from H_pipeline.H04_merge_resolver import get_merge_resolver
merged = get_merge_resolver().merge(raw_extractions)
```

**Merge algorithm (deterministic):**

1. Sort inputs by stable key (doc_id, entity_type, field_name, value, ...)
2. Group by dedupe key fields
3. Resolve each group: highest strategy priority wins, then table evidence bonus, then longer evidence text, then deterministic tie-break
4. Merge supporting evidence from all duplicates into winner
5. Enforce mutual exclusivity constraints
6. Sort output for determinism

**Invariant:** Same inputs always produce same outputs.

## Usage Patterns

```python
# Typical orchestrator setup
factory = ComponentFactory(config, run_id, version, log_dir)
parser = factory.create_parser()
generators = factory.create_generators()
claude_client = factory.create_claude_client()
llm_engine = factory.create_llm_engine(claude_client)

pipeline = AbbreviationPipeline(
    run_id=run_id, pipeline_version=version,
    parser=parser, generators=generators, ...
)

doc = pipeline.parse_pdf(pdf_path, output_dir)
candidates, full_text = pipeline.generate_candidates(doc)
```

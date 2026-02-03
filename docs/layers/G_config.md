# G_config -- Pipeline Configuration

## Purpose

Layer G centralizes all pipeline parameters in a single YAML configuration file and provides type-safe configuration key enums that prevent typos, enable IDE autocomplete, and document defaults.

## Modules

### G01_config_keys.py

Type-safe configuration key enums with default values and descriptions.

**Base class:**

- `ConfigKeyBase(str, Enum)` -- Inherits from `str` for direct use as dictionary keys. Each member has `.default` and `.description` properties.

**Enum classes:**

| Class | Scope | Example keys |
|-------|-------|-------------|
| `ConfigKey` | Top-level keys | `RUN_ID`, `CONTEXT_WINDOW(300)`, `TIMEOUT_SECONDS(30)`, `PATHS`, `API`, `HEURISTICS`, `GENERATORS` |
| `CacheConfig` | Cache section | `ENABLED(True)`, `DIRECTORY("cache")`, `TTL_HOURS(24)`, `TTL_DAYS(30)` |
| `ParserConfig` | PDF parsing | `EXTRACTION_METHOD("unstructured")`, `HI_RES_MODEL_NAME("yolox")`, `INFER_TABLE_STRUCTURE(True)` |
| `GeneratorConfig` | Candidate generation | `MIN_SF_LENGTH(2)`, `MAX_SF_LENGTH(10)`, `CONTEXT_WINDOW_CHARS(400)`, `MAX_CANDIDATES_PER_BLOCK(200)` |
| `EnricherConfig` | Normalization | `FUZZY_CUTOFF(0.90)`, `FILL_LONG_FORM_FOR_ORPHANS(True)`, `ENRICH_MISSING_MESH(True)` |
| `PipelineConfig` | Extraction pipeline | `USE_LLM_VALIDATION(True)`, `USE_VLM_TABLES(False)`, `USE_NORMALIZATION(True)` |
| `LLMConfig` | LLM settings | `MODEL("claude-sonnet-4-20250514")`, `TEMPERATURE(0.0)`, `MAX_TOKENS(4096)` |

**Helper functions:**

```python
from G_config.G01_config_keys import get_config, get_nested_config, ConfigKey, CacheConfig

value = get_config(config, ConfigKey.TIMEOUT_SECONDS)
ttl = get_nested_config(config, ConfigKey.CACHE, CacheConfig.TTL_HOURS)
```

### config.yaml

Complete pipeline configuration (v15.0, ~1041 lines). All parameters read from this file; no hardcoded values in scripts.

**Major sections:**

| Section | Content |
|---------|---------|
| `system` | Name, version (`15.0`), pipeline version (`0.7`) |
| `paths` | Base paths, dictionaries, databases, logs, cache, PDF input, gold data |
| `lexicons` | 20+ lexicon file mappings (abbreviation, disease, drug, UMLS, MONDO, ChEMBL, trial acronyms, PRO scales) |
| `databases` | SQLite databases (disease_ontology.db, orphanet_nlp.db) |
| `defaults` | confidence_threshold (0.75), fuzzy_match_threshold (85), context_window (300) |
| `features` | Feature flags: drug_detection, disease_detection, classification, pubtator_enrichment, ai_validation |
| `generators` | Per-generator config: syntax_pattern, glossary_table, regex_pattern, layout, lexicon (with obvious_noise list) |
| `heuristics` | PASO rules: stats_abbrevs (PASO A), sf_blacklist (PASO B, 70+ entries), common_words, hyphenated_abbrevs (PASO C), LLM SF extractor (PASO D), context_required_sfs |
| `api` | PubTator3 (base_url, rate_limit, cache TTL), Claude (fast/validation models, batch_delay_ms) |
| `nct_enricher` | ClinicalTrials.gov cache settings |
| `validation` | Per-entity validation: abbreviation, drug (stages), disease (stages, enrichment_mode) |
| `normalization` | term_mapper (fuzzy matching), disambiguator (min_context_score, whitelist_unexpanded) |
| `disease_detection` | Lexicon toggles, scispacy enable, FP filter config, confidence thresholds |
| `drug_detection` | Lexicon priorities (alexion, investigational, FDA, RxNorm), compound ID patterns |
| `deduplication` | priority_order, remove_expansion_matches |
| `extraction_pipeline` | Preset selection, extractor toggles, processing options, visual extraction pipeline, page limits |
| `table_extraction` | Docling TableFormer mode, cell matching, validation thresholds, definition table salvage |
| `vision` | Max image size (5MB), compression settings, OCR fallback |
| `logging` | Level (INFO), console (ERROR), file rotation |
| `runtime` | batch_size (10), timeout (300s), max_file_size (100MB), on_error policy |

**Environment variables:**

- `CORPUS_BASE_PATH` -- Base path for all resources (auto-detected if unset)
- `ANTHROPIC_API_KEY` or `CLAUDE_API_KEY` -- API key for Claude validation

**Extraction presets:**

| Preset | Extractors |
|--------|-----------|
| `standard` | Drugs, diseases, genes, abbreviations, feasibility, tables, figures, care pathways, recommendations |
| `all` | Everything including authors, citations, tables, figures, metadata |
| `minimal` | Abbreviations only (no LLM) |
| `drugs_only` | Drug detection only |
| `diseases_only` | Disease detection only |
| `genes_only` | Gene detection only |
| `abbreviations_only` | Abbreviation extraction only |
| `feasibility_only` | Feasibility extraction only |
| `entities_only` | Drugs, diseases, genes, abbreviations |
| `clinical_entities` | Drugs, diseases only |
| `metadata_only` | Authors, citations, document metadata |
| `images_only` | Tables + figures/visuals |
| `tables_only` | Table extraction only (no figures) |

## Usage Patterns

```yaml
# Switch preset
extraction_pipeline:
  preset: "standard"

# Or customize individual extractors
extraction_pipeline:
  preset: null
  extractors:
    drugs: true
    diseases: true
    genes: false
    abbreviations: true

# Adjust API model
api:
  claude:
    validation:
      model: "claude-sonnet-4-20250514"
```

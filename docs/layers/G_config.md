# Layer G: Pipeline Configuration

## Purpose

Centralizes all pipeline parameters in a single YAML file with type-safe config key enums. 2 modules (`G01_config_keys.py`, `config.yaml`).

---

## Modules

### G01_config_keys.py

Type-safe enums inheriting from `ConfigKeyBase` with `.default` and `.description` properties.

| Class | Example keys |
|-------|-------------|
| `ConfigKey` | `RUN_ID`, `CONTEXT_WINDOW(300)`, `TIMEOUT_SECONDS(30)`, `PATHS`, `API` |
| `CacheConfig` | `ENABLED(True)`, `DIRECTORY("cache")`, `TTL_HOURS(24)` |
| `ParserConfig` | `EXTRACTION_METHOD("unstructured")`, `HI_RES_MODEL_NAME("yolox")` |
| `GeneratorConfig` | `MIN_SF_LENGTH(2)`, `MAX_SF_LENGTH(10)`, `CONTEXT_WINDOW_CHARS(400)` |
| `EnricherConfig` | `FUZZY_CUTOFF(0.90)`, `ENRICH_MISSING_MESH(True)` |
| `PipelineConfig` | `USE_LLM_VALIDATION(True)`, `USE_VLM_TABLES(False)` |
| `LLMConfig` | `MODEL("claude-sonnet-4-20250514")`, `TEMPERATURE(0.0)`, `MAX_TOKENS(4096)` |

```python
from G_config.G01_config_keys import get_config, get_nested_config, ConfigKey, CacheConfig
value = get_config(config, ConfigKey.TIMEOUT_SECONDS)
ttl = get_nested_config(config, ConfigKey.CACHE, CacheConfig.TTL_HOURS)
```

### config.yaml

Complete pipeline configuration.

| Section | Content |
|---------|---------|
| `system` | Name, version, pipeline version |
| `paths` | Base paths, dictionaries, databases, logs, cache, PDFs, gold data |
| `lexicons` | 20+ lexicon file mappings |
| `defaults` | confidence_threshold (0.75), fuzzy_match_threshold (85), context_window (300) |
| `features` | Feature flags: drug/disease detection, classification, enrichment, validation |
| `generators` | Per-generator config with obvious_noise list |
| `heuristics` | PASO rules: stats_abbrevs (A), sf_blacklist (B, 70+ entries), hyphenated (C), LLM SF (D) |
| `api` | PubTator3, Claude (models, batch_delay_ms, model_tiers) |
| `validation` | Per-entity validation stages |
| `disease_detection` | Lexicon toggles, scispacy, FP filter, thresholds |
| `drug_detection` | Lexicon priorities, compound ID patterns |
| `extraction_pipeline` | Presets, extractor toggles, visual pipeline, page limits |
| `table_extraction` | Docling TableFormer mode, validation thresholds |

**Environment variables:** `CORPUS_BASE_PATH`, `ANTHROPIC_API_KEY`

---

## Model Tier Routing

Maps `call_type` to model in `api.claude.model_tiers`. Resolved via `D02_llm_engine.resolve_model_tier(call_type)`.

```yaml
api:
  claude:
    model_tiers:
      # Haiku tier ($1/$5 per MTok)
      abbreviation_batch_validation: "claude-haiku-4-5-20251001"
      abbreviation_single_validation: "claude-haiku-4-5-20251001"
      fast_reject: "claude-haiku-4-5-20251001"
      document_classification: "claude-haiku-4-5-20251001"
      description_extraction: "claude-haiku-4-5-20251001"
      image_classification: "claude-haiku-4-5-20251001"
      sf_only_extraction: "claude-haiku-4-5-20251001"
      layout_analysis: "claude-haiku-4-5-20251001"
      vlm_visual_enrichment: "claude-haiku-4-5-20251001"
      ocr_text_fallback: "claude-haiku-4-5-20251001"
      # Sonnet tier ($3/$15 per MTok)
      feasibility_extraction: "claude-sonnet-4-20250514"
      recommendation_extraction: "claude-sonnet-4-20250514"
      recommendation_vlm: "claude-sonnet-4-20250514"
      vlm_table_extraction: "claude-sonnet-4-20250514"
      flowchart_analysis: "claude-sonnet-4-20250514"
      chart_analysis: "claude-sonnet-4-20250514"
      vlm_detection: "claude-sonnet-4-20250514"
```

---

## Extraction Presets

| Preset | Extractors |
|--------|-----------|
| `standard` | Drugs, diseases, genes, abbreviations, feasibility, tables, figures, care pathways, recommendations |
| `all` | Everything including authors, citations, metadata |
| `minimal` | Abbreviations only |
| `entities_only` | Drugs, diseases, genes, abbreviations |
| `clinical_entities` | Drugs, diseases only |
| `metadata_only` | Authors, citations, document metadata |
| `images_only` | Tables + figures |

Single-entity presets: `drugs_only`, `diseases_only`, `genes_only`, `abbreviations_only`, `feasibility_only`, `tables_only`.

## Usage

```yaml
extraction_pipeline:
  preset: "standard"

# Or customize individually
extraction_pipeline:
  preset: null
  extractors:
    drugs: true
    diseases: true
    genes: false
```

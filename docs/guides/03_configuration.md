# Configuration Guide

All parameters in `corpus_metadata/G_config/config.yaml`.

## Extraction Presets

Presets override individual extractor flags.

```yaml
extraction_pipeline:
  preset: "standard"  # Options: standard, all, minimal, drugs_only, diseases_only,
                      # genes_only, abbreviations_only, feasibility_only,
                      # entities_only, clinical_entities, metadata_only,
                      # images_only, tables_only
```

| Preset | Extractors Enabled |
|--------|-------------------|
| `standard` | Drugs, diseases, genes, abbreviations, feasibility, tables, figures, care pathways, recommendations |
| `all` | All entity types + figures, tables, metadata |
| `minimal` | Abbreviations only |
| `entities_only` | Drugs, diseases, genes, abbreviations |
| `clinical_entities` | Drugs, diseases only |
| `metadata_only` | Authors, citations, document metadata |
| `images_only` | Tables + figures/visuals |

Set `preset: null` to use individual extractor flags.

## Entity Toggles

Active when `preset: null`:

```yaml
extraction_pipeline:
  preset: null
  extractors:
    drugs: true
    diseases: true
    genes: true
    abbreviations: true
    feasibility: true
    pharma_companies: true
    authors: true
    citations: true
    document_metadata: true
    tables: true
    figures: true
```

## Processing Options

```yaml
extraction_pipeline:
  options:
    use_llm_validation: true
    use_llm_feasibility: true
    use_vlm_tables: true
    use_normalization: true
    use_epi_enricher: true
    use_zeroshot_bioner: true
    use_biomedical_ner: true
    use_patient_journey: true
    use_registry_extraction: true
    parallel_extraction: true
```

## Page Limits

```yaml
extraction_pipeline:
  page_limits:
    max_pages: null        # null = no limit
    page_range: null       # null = all, or [start, end]
```

## API Configuration

### Claude API

```yaml
api:
  claude:
    fast:
      model: "claude-sonnet-4-5-20250929"
      max_tokens: 1500
      temperature: 0
    validation:
      model: "claude-sonnet-4-20250514"
      max_tokens: 450
      temperature: 0
      top_p: 1.0
    batch_delay_ms: 100
```

### Model Tier Routing

Routes LLM calls by task complexity. Simple tasks use Haiku; complex reasoning uses Sonnet.

```yaml
api:
  claude:
    model_tiers:
      # Haiku ($1/$5 per MTok)
      abbreviation_batch_validation: "claude-haiku-4-5-20251001"
      document_classification: "claude-haiku-4-5-20251001"
      layout_analysis: "claude-haiku-4-5-20251001"
      image_classification: "claude-haiku-4-5-20251001"
      # Sonnet ($3/$15 per MTok)
      feasibility_extraction: "claude-sonnet-4-20250514"
      flowchart_analysis: "claude-sonnet-4-20250514"
      vlm_table_extraction: "claude-sonnet-4-20250514"
```

See [Cost Optimization Guide](05_cost_optimization.md) for the full call_type table.

### PubTator3 API

```yaml
api:
  pubtator:
    enabled: true
    base_url: "https://www.ncbi.nlm.nih.gov/research/pubtator3-api"
    timeout_seconds: 30
    rate_limit_per_second: 3
    cache:
      enabled: true
      directory: "cache/pubtator"
      ttl_hours: 168    # 7 days
```

### NCT Enricher

```yaml
nct_enricher:
  enabled: true
  cache:
    enabled: true
    directory: "cache/clinicaltrials"
    ttl_days: 30
```

## Paths

```yaml
paths:
  base: null               # Auto-detect from CORPUS_BASE_PATH env var
  dictionaries: "ouput_datasources"
  databases: "corpus_db"
  logs: "corpus_log"
  cache: "cache"
  pdf_input: "Pdfs"
  gold_data: "gold_data"
  papers_folder: "gold_data/PAPERS"
  gold_json: "gold_data/papers_gold_v2.json"
```

## PASO Heuristics (Abbreviations)

### PASO A: Stats Whitelist (Auto-Approve)

```yaml
heuristics:
  stats_abbrevs:
    CI: "confidence interval"
    SD: "standard deviation"
    HR: "hazard ratio"
    OR: "odds ratio"
    # ... (full list in config.yaml)
```

### PASO B: Blacklist (Auto-Reject)

Country codes (US, UK, EU), regulatory agencies (FDA, EMA), credentials (MD, PHD), etc. Full list in `config.yaml` under `heuristics.sf_blacklist`.

### PASO C: Hyphenated Abbreviations

Entries with `"trial name"` are auto-enriched from ClinicalTrials.gov:

```yaml
heuristics:
  hyphenated_abbrevs:
    APPEAR-C3G: "trial name"
    CKD-EPI: "Chronic Kidney Disease Epidemiology Collaboration"
```

### PASO D: LLM SF-Only Extraction

```yaml
heuristics:
  enable_llm_sf_extractor: true
  llm_sf_max_chunks: 5
  llm_sf_chunk_size: 3000
  llm_sf_confidence: 0.75
```

## Feature Flags

```yaml
features:
  drug_detection: true
  disease_detection: true
  abbreviation_extraction: true
  classification: true
  section_detection: true
  table_extraction: true
  title_extraction: true
  date_extraction: true
  description_extraction: true
  citation_extraction: true
  person_extraction: true
  reference_extraction: true
  pubtator_enrichment: true
  ai_validation: true
  intelligent_rename: true
  caching: true
```

## Disease Detection

```yaml
disease_detection:
  enabled: true
  enable_general_lexicon: true          # 29K+
  enable_orphanet: true                 # 9.6K rare diseases
  enable_mondo: true                    # 97K MONDO ontology
  enable_rare_disease_acronyms: true    # 1.6K acronyms
  enable_scispacy: true
  context_window: 300
  fp_filter:
    enabled: true
    filter_chromosomes: true
    filter_genes: true
    short_match_threshold: 4
    min_confidence_after_adjustment: 0.35
  validation:
    enabled: false
    batch_size: 10
    confidence_threshold: 0.75
```

## Drug Detection

```yaml
drug_detection:
  enabled: true
  enable_alexion_lexicon: true
  enable_investigational_lexicon: true   # 32K
  enable_fda_lexicon: true               # 50K
  enable_rxnorm_lexicon: true            # 133K
  enable_scispacy: true
  enable_patterns: true                  # Compound IDs (LNP023, ALXN1720)
  context_window: 300
```

## Validation

```yaml
validation:
  abbreviation:
    enabled: true
    model_tier: "validation"
    skip_validation: false
  drug:
    enabled: true
    confidence_threshold: 0.85
    stages: ["false_positive_filter", "knowledge_base", "claude_ai"]
  disease:
    enabled: true
    confidence_threshold: 0.75
    enrichment_mode: "balanced"
    stages: ["pattern_detection", "lexicon_matching", "claude_ai"]
```

## Normalization

```yaml
normalization:
  term_mapper:
    enabled: true
    enable_fuzzy_matching: false
    fuzzy_cutoff: 0.90
  disambiguator:
    enabled: true
    min_context_score: 2
    min_margin: 1
    whitelist_unexpanded: ["CI", "SD", "HR", "OR", "IQR", "BMI", "DNA", "RNA", "IV", "IM", "SC", "PO"]
```

## Deduplication

```yaml
deduplication:
  enabled: true
  priority_order: ["drug", "disease"]
  remove_expansion_matches: true
```

## Output Format

```yaml
output:
  format: "json"
  json_indent: 2
  include:
    statistics: true
    confidence: true
    context: true
    identifiers: true
```

## Table Extraction

```yaml
table_extraction:
  mode: "accurate"          # "accurate" or "fast"
  do_cell_matching: true
  ocr_enabled: true
  max_avg_cell_length: 150
  vlm:
    enabled: false
    min_confidence: 0.3
    min_data_rows: 2
```

## Visual Extraction

```yaml
extraction_pipeline:
  visual_extraction:
    enabled: true
    visual_detection:
      mode: "layout-aware"     # layout-aware, vlm-only, hybrid, heuristic
      model: "claude-sonnet-4-20250514"
    vlm:
      enabled: true
      model: "claude-sonnet-4-20250514"
      validate_tables: true
```

## Logging and Runtime

```yaml
logging:
  level: "INFO"
  console: false
  console_level: "ERROR"
  file: "corpus.log"
  max_size_mb: 10
  backup_count: 5

runtime:
  batch_size: 10
  timeout_seconds: 300
  max_file_size_mb: 100
  on_error: "log_and_continue"
```

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `CORPUS_BASE_PATH` | Base directory (auto-detected from `corpus_metadata/`) |
| `ANTHROPIC_API_KEY` | Claude API key (required for LLM features) |

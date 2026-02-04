# Configuration Guide

All pipeline parameters are centralized in `corpus_metadata/G_config/config.yaml`. No parameters are hardcoded in scripts.

## Extraction Presets

Presets control which extractors run. When a preset is set, it overrides the individual extractor flags.

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
| `all` | All entity types including authors, citations, tables, figures, metadata |
| `minimal` | Abbreviations only |
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

Set `preset: null` to use individual extractor flags instead.

## Entity Toggles

When `preset` is `null`, individual extractors are controlled here:

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

Fine-grained control over pipeline behavior:

```yaml
extraction_pipeline:
  options:
    use_llm_validation: true       # LLM validation for abbreviations
    use_llm_feasibility: true      # LLM-based feasibility extraction
    use_vlm_tables: true           # Vision model for table extraction
    use_normalization: true        # Entity normalization/enrichment (RxNorm, MONDO)
    use_epi_enricher: true         # EpiExtract4GARD-v2 for epidemiology NER
    use_zeroshot_bioner: true      # ZeroShotBioNER for ADE, dosage, frequency, route
    use_biomedical_ner: true       # d4data/biomedical-ner-all for 84 entity types
    use_patient_journey: true      # Patient journey extraction
    use_registry_extraction: true  # Registry extraction
    parallel_extraction: true      # Run independent extractors in parallel
```

## Page Limits

```yaml
extraction_pipeline:
  page_limits:
    max_pages: null              # null = no limit, or integer to cap pages processed
    page_range: null             # null = all pages, or [start, end] for specific range
```

## Deduplication

```yaml
deduplication:
  enabled: true
  priority_order: ["drug", "disease"]  # Entity type priority for resolving duplicates
  remove_expansion_matches: true  # Remove candidates where SF matches an LF of another
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

- `fast`: Used for basic tasks (cheaper, faster).
- `validation`: Used for final validation and enrichment (best quality).

### Model Tier Routing (LLM Cost Optimization)

Routes LLM calls to different models based on task complexity. Simple tasks use cheaper Haiku; complex reasoning stays on Sonnet.

```yaml
api:
  claude:
    model_tiers:
      # Haiku tier ($1/$5 per MTok) — simple tasks
      abbreviation_batch_validation: "claude-haiku-4-5-20250901"
      document_classification: "claude-haiku-4-5-20250901"
      layout_analysis: "claude-haiku-4-5-20250901"
      image_classification: "claude-haiku-4-5-20250901"
      # Sonnet tier ($3/$15 per MTok) — complex reasoning
      feasibility_extraction: "claude-sonnet-4-20250514"
      flowchart_analysis: "claude-sonnet-4-20250514"
      vlm_table_extraction: "claude-sonnet-4-20250514"
```

To change a task's model, update its value in this mapping. To add a new LLM call site, add a new entry here. See [Cost Optimization Guide](05_cost_optimization.md) for the full call_type table and implementation details.

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
    enrichment:
      enrich_missing_mesh: true
      add_aliases: true
```

### NCT Enricher (ClinicalTrials.gov)

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

When `base` is `null`, the pipeline auto-detects from the `CORPUS_BASE_PATH` environment variable or the location of `corpus_metadata/`.

## PASO Heuristics (Abbreviation Extraction)

PASO heuristics provide deterministic rules that run before LLM validation for abbreviations.

### PASO A: Stats Whitelist (Auto-Approve)

Statistical abbreviations with well-known definitions are auto-approved when found with numeric evidence:

```yaml
heuristics:
  stats_abbrevs:
    CI: "confidence interval"
    SD: "standard deviation"
    SE: "standard error"
    OR: "odds ratio"
    HR: "hazard ratio"
    IQR: "interquartile range"
    AUC: "area under the curve"
    ROC: "receiver operating characteristic"
```

### PASO B: Blacklist (Auto-Reject)

Non-domain terms that should never be extracted as abbreviations:

```yaml
heuristics:
  sf_blacklist:
    - "US"       # Country codes
    - "UK"
    - "EU"
    - "FDA"      # Regulatory agencies
    - "EMA"
    - "MD"       # Credentials
    - "PHD"
    # ... (full list in config.yaml)
```

### PASO C: Hyphenated Abbreviations

Pre-defined hyphenated terms with known expansions. Entries with `"trial name"` as the value are auto-enriched at runtime from ClinicalTrials.gov:

```yaml
heuristics:
  hyphenated_abbrevs:
    APPEAR-C3G: "trial name"    # Auto-enriched from ClinicalTrials.gov
    CKD-EPI: "Chronic Kidney Disease Epidemiology Collaboration"
    IL-6: "interleukin-6"
```

### PASO D: LLM SF-Only Extraction

LLM-based extractor for finding abbreviations that other generators missed:

```yaml
heuristics:
  enable_llm_sf_extractor: true
  llm_sf_max_chunks: 5
  llm_sf_chunk_size: 3000
  llm_sf_confidence: 0.75
```

## Feature Flags

Global switches for pipeline capabilities:

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
  citation_extraction: true
  person_extraction: true
  pubtator_enrichment: true
  ai_validation: true
  intelligent_rename: true
  caching: true
```

## Disease Detection Settings

```yaml
disease_detection:
  enabled: true
  enable_general_lexicon: true    # 29K+ diseases
  enable_orphanet: true           # 9.6K rare diseases
  enable_scispacy: true           # scispacy NER with UMLS linking
  context_window: 300
  fp_filter:
    enabled: true
    filter_chromosomes: true
    filter_genes: true
    short_match_threshold: 4
    min_confidence_after_adjustment: 0.35
  validation:
    enabled: false                # Set to true for LLM validation
    batch_size: 10
    confidence_threshold: 0.75
```

## Drug Detection Settings

```yaml
drug_detection:
  enabled: true
  enable_alexion_lexicon: true           # Specialized pipeline drugs
  enable_investigational_lexicon: true   # ClinicalTrials.gov drugs (32K)
  enable_fda_lexicon: true               # FDA approved drugs (50K)
  enable_rxnorm_lexicon: true            # RxNorm general terms (133K)
  enable_scispacy: true                  # scispacy NER with UMLS
  enable_patterns: true                  # Compound ID patterns (LNP023, ALXN1720)
  context_window: 300
```

## Validation Settings

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

## Normalization Settings

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
    whitelist_unexpanded:
      - "CI"
      - "SD"
      - "HR"
      - "BMI"
      - "DNA"
      - "IV"
```

## Table Extraction Settings

```yaml
table_extraction:
  mode: "accurate"          # "accurate" or "fast" (TableFormer mode)
  do_cell_matching: true
  ocr_enabled: true
  max_avg_cell_length: 150
  vlm:
    enabled: false
    min_confidence: 0.3
    min_data_rows: 2
```

## Visual Extraction Pipeline

```yaml
extraction_pipeline:
  visual_extraction:
    enabled: true
    visual_detection:
      mode: "layout-aware"     # layout-aware, vlm-only, hybrid, heuristic
      model: "claude-sonnet-4-20250514"
    rendering:
      default_dpi: 300
    vlm:
      enabled: true
      model: "claude-sonnet-4-20250514"
      validate_tables: true
```

## Vision LLM Settings

```yaml
vision:
  max_image_size_bytes: 5242880  # 5MB
  compression:
    enabled: true
    quality: 85
    max_dimension: 2048
  fallback_to_ocr: true
  delay_between_calls_ms: 100
```

## Logging

```yaml
logging:
  level: "INFO"
  console: false
  console_level: "ERROR"
  file: "corpus.log"
  max_size_mb: 10
  backup_count: 5
```

## Runtime

```yaml
runtime:
  batch_size: 10
  timeout_seconds: 300
  max_file_size_mb: 100
  on_error: "log_and_continue"
```

## Global Defaults

```yaml
defaults:
  confidence_threshold: 0.75
  fuzzy_match_threshold: 85
  context_window: 300
  min_term_length: 2
```

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `CORPUS_BASE_PATH` | Base directory for all resources | Auto-detected from `corpus_metadata/` location |
| `ANTHROPIC_API_KEY` | Claude API key | Required for LLM features |
| `CLAUDE_API_KEY` | Mentioned in config comments as alternative | Not used in pipeline code; use `ANTHROPIC_API_KEY` |

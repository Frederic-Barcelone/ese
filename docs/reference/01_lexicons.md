# Lexicon Reference

ESE loads approximately 617,000 terms from multiple biomedical lexicons at startup. These lexicons power the high-recall candidate generation layer (C_generators) using FlashText keyword processors for O(n) matching against document text.

## Lexicon Inventory

| Source | Approx Terms | Purpose | Lexicon File(s) |
|--------|-------------|---------|-----------------|
| Meta-Inventory | 65K+ (104K entries, 170K senses) | Clinical abbreviations | `2025_meta_inventory_abbreviations.json` |
| Clinical Research Abbreviations | varies | Clinical research-specific abbreviations | `clinical_research_abbreviations.json` |
| General Abbreviations | varies | General abbreviation vocabulary | `2025_08_abbreviation_general.json` |
| MONDO | 97K | Unified disease ontology | `2025_mondo_diseases.json` |
| RxNorm | 132K | Drug vocabulary | `2025_08_lexicon_drug.json` |
| ChEMBL | 23K | Approved drugs | `2025_chembl_drugs.json` |
| Orphanet | 9.5K | Rare diseases with ORPHA codes | `2025_08_orphanet_diseases.json` |
| HGNC/Orphadata | ~40K | Gene symbols and synonyms | `2025_08_orphadata_genes.json` |
| Trial Acronyms | 125K | ClinicalTrials.gov trial acronyms | `trial_acronyms_lexicon.json` |
| General Disease | 29K | General disease terms | `2025_08_lexicon_disease.json` |
| Specialized Disease | ~3K | PAH, ANCA, IgAN specific | `disease_lexicon_pah.json`, `disease_lexicon_anca.json`, `disease_lexicon_igan.json` |
| Rare Disease Acronyms | varies | Disease abbreviations with ORPHA codes | `2025_08_rare_disease_acronyms.json` |
| FDA Approved | 50K | FDA-approved drugs | `2025_08_fda_approved_drugs.json` |
| Investigational | 32K | ClinicalTrials.gov drugs | `2025_08_investigational_drugs.json` |
| Alexion Pipeline | varies | Alexion-specific drugs | `2025_08_alexion_drugs.json` |
| Medical Terms | varies | General medical terminology | `2025_08_lexicon_medical_terms.json` |
| UMLS Biological | varies | UMLS biological abbreviations | `2025_08_umls_biological_abbreviations_v5.tsv` |
| UMLS Clinical | varies | UMLS clinical abbreviations | `2025_08_umls_clinical_abbreviations_v5.tsv` |
| PRO Scales | varies | Patient-Reported Outcome instruments | `pro_scales_lexicon.json` |
| Pharma Companies | varies | Pharmaceutical company names | (loaded via pharma loader) |

## Loading Mechanism

All lexicon loading is centralized in `C_generators/C22_lexicon_loaders.py` via the `LexiconLoaderMixin` class. This mixin is used by the FlashText-based generator (`C04_strategy_flashtext.py`) and entity-specific generators.

### FlashText KeywordProcessor

Lexicons are loaded into FlashText `KeywordProcessor` instances for fast O(n) matching against document text. FlashText handles:

- Case-insensitive matching (configurable per lexicon)
- Word boundary detection (no partial-word matches)
- Simultaneous matching against 600K+ terms in a single pass

### Data Structures

Each loaded term is tracked in several dictionaries:

- `entity_kp` -- FlashText KeywordProcessor holding all terms
- `entity_canonical` -- Maps matched text to canonical/preferred name
- `entity_source` -- Maps matched text to source lexicon filename
- `entity_ids` -- Maps matched text to external identifiers (ORPHA, MONDO, ICD-10, etc.)

For abbreviation lexicons, terms are loaded as `LexiconEntry` objects (a dataclass defined in `C21_noise_filters.py`; the actual lexicon loading logic is handled by `LexiconLoaderMixin` in `C22_lexicon_loaders.py`) with fields:

- `sf` -- Short form (the abbreviation)
- `lf` -- Long form (the expansion)
- `pattern` -- Compiled regex pattern for matching
- `source` -- Source lexicon filename
- `lexicon_ids` -- External identifiers (optional)
- `preserve_case` -- Whether to preserve case during matching (used for trial acronyms, PRO scales)

## Lexicon File Formats

### Abbreviation Lexicons (JSON)

```json
{
  "TNF": {
    "canonical_expansion": "tumor necrosis factor",
    "regex": "\\bTNF\\b",
    "case_insensitive": false,
    "expansions": ["tumor necrosis factor", "tumour necrosis factor"]
  }
}
```

### Disease Lexicons (JSON array)

```json
[
  {
    "label": "Pulmonary arterial hypertension",
    "sources": [
      {"source": "Orphanet", "id": "ORPHA:182090"},
      {"source": "ICD-10", "id": "I27.0"}
    ],
    "synonyms": ["PAH"]
  }
]
```

### Orphanet Lexicon (JSON array)

```json
[
  {
    "name": "IgA nephropathy",
    "orphacode": "97555",
    "synonyms": ["IgAN", "Berger disease"]
  }
]
```

### Specialized Disease Lexicons (ANCA, IgAN, PAH)

These lexicons have a richer structure with multiple sections:

```json
{
  "diseases": {
    "disease_key": {
      "preferred_label": "...",
      "abbreviation": "...",
      "synonyms": ["..."],
      "identifiers": {"ORPHA": "...", "ICD10": "...", "SNOMED_CT": "..."}
    }
  },
  "abbreviation_expansions": {
    "SF": {"preferred": "long form"}
  },
  "composite_terms": {
    "TERM": {"expansion": "full expansion"}
  }
}
```

The IgAN lexicon additionally includes a `renal_terms` section; the PAH lexicon includes `hemodynamic_terms`.

### ChEMBL Drug Lexicon

```json
{
  "drugs": [
    {
      "label": "iptacopan",
      "chembl_id": "CHEMBL4594299",
      "max_phase": 3,
      "synonyms": ["LNP023"]
    }
  ]
}
```

### UMLS Lexicons (TSV)

Tab-separated with columns: `Abbreviation`, `Expansion`, `TopSource`

### Trial Acronyms Lexicon

```json
{
  "APPEAR-C3G": {
    "canonical_expansion": "A Phase 3 Study of Iptacopan in C3G",
    "regex": "\\bAPPEAR-C3G\\b",
    "case_insensitive": false,
    "nct_id": "NCT04817618"
  }
}
```

## Lexicon File Paths

All lexicon filenames are configured in [`G_config/config.yaml`](../layers/G_config.md) under the `lexicons:` section. Files are stored in the `ouput_datasources/` directory (relative to the project base path). The base path is resolved from:

1. `CORPUS_BASE_PATH` environment variable
2. Auto-detection relative to the `corpus_metadata/` directory

## Version Convention

Lexicon files follow a date-based naming convention: `2025_08_*` indicates August 2025 vintage. Newer lexicons use `2025_*` without a month. This allows tracking which version of ontology data was used for extraction.

## Loading Summary

At startup, `C22_lexicon_loaders.py` prints a categorized summary of loaded terms:

```
Lexicons loaded: 14 files, 617,234 terms
  Abbreviation (289,432 terms)
  Drug (205,100 terms)
  Disease (97,502 terms)
  Other (25,200 terms)
```

Categories are: Abbreviation, Drug, Disease, and Other (trial acronyms, PRO scales, pharma companies).

## Usage Tracking

`Z_utils/Z06_usage_tracker.py` tracks which lexicons contributed to extractions during each pipeline run. Statistics are stored in a SQLite database for analysis of lexicon effectiveness.

## Downloading Lexicons

To download or update lexicons:

```bash
python corpus_metadata/Z_utils/Z10_download_lexicons.py
```

This script fetches the latest versions of public lexicons (MONDO, ChEMBL, Orphanet, etc.) and writes them to the configured lexicon directory.

## Adding a New Lexicon

1. Create the JSON file in the `ouput_datasources/` directory following one of the formats above
2. Add the filename to `G_config/config.yaml` under the `lexicons:` section
3. Add a `_load_*` method to `C_generators/C22_lexicon_loaders.py` in the `LexiconLoaderMixin` class
4. Call the loader from the relevant generator strategy (e.g., `C06_strategy_disease.py` for disease lexicons, `C07_strategy_drug.py` for drug lexicons)
5. Add the lexicon name to the `category_map` dict in `_print_lexicon_summary()` for correct categorization in startup logs
6. Register the lexicon in the appropriate entity processor if needed

## Noise Filtering

Loaded terms pass through noise filters defined in `C_generators/C21_noise_filters.py`:

- `BAD_LONG_FORMS` -- Known incorrect expansions to exclude
- `WRONG_EXPANSION_BLACKLIST` -- Known bad SF-LF pairs
- Minimum term length checks (typically 2-3 characters)
- The `obvious_noise` list in `config.yaml` filters single letters, common English words, and measurement units from FlashText matching

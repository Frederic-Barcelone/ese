# Lexicon Reference

~617,000 terms loaded at startup from biomedical lexicons. Powers candidate generation (C_generators) via FlashText keyword processors for O(n) matching.

## Lexicon Inventory

| Source | Terms | Purpose | File |
|--------|-------|---------|------|
| Meta-Inventory | 104K+ | Clinical abbreviations (170K senses) | `2025_meta_inventory_abbreviations.json` |
| MONDO | 97K | Unified disease ontology | `2025_mondo_diseases.json` |
| RxNorm | 133K | Drug vocabulary | `2025_08_lexicon_drug.json` |
| ChEMBL | 23K | Approved drugs | `2025_chembl_drugs.json` |
| Orphanet | 9.5K | Rare diseases (ORPHA codes) | `2025_08_orphanet_diseases.json` |
| HGNC/Orphadata | ~4,100+ | Gene symbols and aliases | `2025_08_orphadata_genes.json` |
| Trial Acronyms | 125K | ClinicalTrials.gov | `trial_acronyms_lexicon.json` |
| General Disease | 29K | General disease terms | `2025_08_lexicon_disease.json` |
| FDA Approved | 50K | FDA-approved drugs | `2025_08_fda_approved_drugs.json` |
| Investigational | 32K | ClinicalTrials.gov drugs | `2025_08_investigational_drugs.json` |
| Specialized Disease | ~3K | PAH, ANCA, IgAN, C3G specific | `disease_lexicon_pah.json`, etc. |
| Rare Disease Acronyms | ~1.6K | Disease abbreviations (ORPHA codes) | `2025_08_rare_disease_acronyms.json` |
| UMLS | varies | Clinical + biological abbreviations | `2025_08_umls_*_abbreviations_v5.tsv` |
| PRO Scales | varies | Patient-Reported Outcomes | `pro_scales_lexicon.json` |

Also loaded: general abbreviations, clinical research abbreviations, Alexion pipeline drugs, medical terms, pharma companies.

## Loading Mechanism

Centralized in `C22_lexicon_loaders.py` via `LexiconLoaderMixin`, used by FlashText generator (C04) and entity-specific generators.

**FlashText KeywordProcessor:**
- Case-insensitive matching (configurable)
- Word boundary detection (no partial matches)
- 600K+ terms matched in a single O(n) pass

**Data structures per term:**
- `entity_kp` -- FlashText KeywordProcessor
- `entity_canonical` -- Matched text to canonical name
- `entity_source` -- Matched text to source lexicon
- `entity_ids` -- Matched text to external identifiers

## Lexicon File Formats

**Abbreviation (JSON):**
```json
{"TNF": {"canonical_expansion": "tumor necrosis factor", "regex": "\\bTNF\\b", "case_insensitive": false}}
```

**Disease (JSON array):**
```json
[{"label": "Pulmonary arterial hypertension", "sources": [{"source": "Orphanet", "id": "ORPHA:182090"}], "synonyms": ["PAH"]}]
```

**ChEMBL Drug:**
```json
{"drugs": [{"label": "iptacopan", "chembl_id": "CHEMBL4594299", "max_phase": 3, "synonyms": ["LNP023"]}]}
```

**UMLS (TSV):** Tab-separated: `Abbreviation`, `Expansion`, `TopSource`

## File Paths

Filenames configured in `config.yaml` under `lexicons:`. Files in `ouput_datasources/` relative to project base path.

Version convention: `2025_08_*` = August 2025 vintage. `2025_*` = no specific month.

## Adding a New Lexicon

1. Create JSON file in `ouput_datasources/`
2. Add filename to `config.yaml` under `lexicons:`
3. Add `_load_*` method in `C22_lexicon_loaders.py`
4. Call from the relevant generator strategy
5. Add to `category_map` in `_print_lexicon_summary()`

## Noise Filtering

Loaded terms pass through `C21_noise_filters.py`: `BAD_LONG_FORMS`, `WRONG_EXPANSION_BLACKLIST`, minimum length checks, and the `obvious_noise` list in config.yaml.

# Layer E: Normalization and Enrichment

## Purpose

`E_normalization/` standardizes validated entities by mapping them to canonical ontology codes, resolving ambiguities, deduplicating entries, and enriching them with data from external APIs and NER models. After this layer, every entity carries standard identifiers (MONDO, RxNorm, HGNC, MeSH, etc.) and has been merged with duplicate mentions.

This layer contains 18 modules organized into core normalization, API enrichment, NER enrichment, and specialized enrichment groups.

All enrichers inherit from `BaseEnricher[InputT, OutputT]` (defined in `A_core/A02_interfaces.py`). API-based enrichers use disk caching and rate limiting. NER models are loaded lazily to avoid startup overhead when not needed.

See also: [Data Flow](../architecture/02_data_flow.md) | [Domain Models](../architecture/03_domain_models.md)

---

## Key Modules

### Core Normalization

| Module | Description |
|--------|-------------|
| `E01_term_mapper.py` | Abbreviation long-form canonicalization. Normalizes whitespace, case, punctuation, and Unicode to produce canonical forms for deduplication. |
| `E02_disambiguator.py` | Resolves `SHORT_FORM_ONLY` abbreviations using document context. Employs bag-of-words voting across the document to select the most likely long form. |
| `E03_disease_normalizer.py` | Extracts primary codes from disease identifiers. Maps raw identifier lists to canonical ICD-10, SNOMED, MONDO, and ORPHA codes on `ExtractedDisease`. |
| `E07_deduplicator.py` | Merges duplicate abbreviations by `short_form`. Uses quality ranking (DEFINITION_PAIR > GLOSSARY_ENTRY > SHORT_FORM_ONLY) to select the best representative. |
| `E11_span_deduplicator.py` | Merges overlapping NER spans from multiple sources (FlashText, scispacy, biomedical-ner-all). Resolves conflicts when multiple strategies extract overlapping text ranges. |
| `E17_entity_deduplicator.py` | General entity deduplication for all entity types (diseases, drugs, genes). Merges by canonical identifier, combining mention counts and page references. |

### API Enrichment

| Module | Description |
|--------|-------------|
| `E04_pubtator_enricher.py` | PubTator3 API enrichment for diseases. Adds MeSH identifiers, normalized names, and aliases. Uses `PubTator3Client` with 7-day disk cache and 3 queries/sec rate limit (per NCBI guidelines). |
| `E05_drug_enricher.py` | PubTator3 API enrichment for drugs. Reuses `PubTator3Client` for drug-specific MeSH and alias lookup. |
| `E06_nct_enricher.py` | ClinicalTrials.gov API enrichment for NCT identifiers. Fetches trial title, conditions, interventions, sponsor, and status. 30-day disk cache. |
| `E18_gene_enricher.py` | PubTator3 API enrichment for genes. Adds gene-specific identifiers and aliases. |

### NER Enrichment

| Module | Description |
|--------|-------------|
| `E08_epi_extract_enricher.py` | EpiExtract4GARD-v2 model for rare disease epidemiology extraction. Detects LOC (location), EPI (epidemiology), and STAT (statistic) entities for prevalence and incidence data. |
| `E09_zeroshot_bioner.py` | ZeroShotBioNER for flexible entity extraction without pre-training. Extracts ADE (adverse drug events), dosage, frequency, route, and other configurable entity types. |
| `E10_biomedical_ner_all.py` | d4data/biomedical-ner-all model supporting 84 biomedical entity types. Provides broad NER coverage as a supplementary source. |

### Specialized Enrichment

| Module | Description |
|--------|-------------|
| `E12_patient_journey_enricher.py` | Patient journey phase extraction: diagnostic delay, treatment lines, disease progression milestones. |
| `E13_registry_enricher.py` | Clinical trial registry data extraction and cross-referencing. |
| `E14_citation_validator.py` | Citation validation via external APIs (PubMed, CrossRef). Verifies PMID and DOI existence and retrieves metadata. |
| `E15_genetic_enricher.py` | Gene-disease association enrichment from Orphadata. Maps genes to associated rare diseases with association types (disease-causing, susceptibility, modifier). |
| `E16_drug_combination_parser.py` | Combination drug therapy parsing. Decomposes "Drug A + Drug B" patterns into individual drug entities with combination metadata. |

---

## Public Interfaces

All enrichers and normalizers implement interfaces from `A_core/A02_interfaces.py`. See [A_core Layer](A_core.md#public-interfaces) for full signatures.

- **`BaseEnricher[InputT, OutputT]`** -- Generic interface for all enrichers. Provides `enrich()` and `enrich_batch()`. Default `enrich_batch()` calls `enrich()` per entity; override for optimized batch processing.
- **`BaseNormalizer`** -- Interface for post-verification standardization via `normalize()`.

### PubTator3Client

```python
from E_normalization.E04_pubtator_enricher import DiseaseEnricher

enricher = DiseaseEnricher(config={"enabled": True})
enriched_disease = enricher.enrich(extracted_disease)
# enriched_disease.mesh_id, enriched_disease.mesh_aliases populated
```

### NCT Enricher

```python
from E_normalization.E06_nct_enricher import NCTEnricher

enricher = NCTEnricher(config={"enabled": True})
trial_info = enricher.enrich("NCT04578834")
# trial_info.title, trial_info.conditions, trial_info.interventions
```

---

## Usage Patterns

### Per-Entity-Type Normalization Flow

**Abbreviations:**
```
Validated abbreviations
  --> E01_term_mapper (canonical form)
  --> E02_disambiguator (resolve SHORT_FORM_ONLY)
  --> E07_deduplicator (merge by short_form)
  --> E11_span_deduplicator (consolidate overlapping mentions)
```

**Diseases:**
```
Disease candidates
  --> E03_disease_normalizer (extract primary codes)
  --> E04_pubtator_enricher (MeSH codes, aliases)
  --> E17_entity_deduplicator (merge by canonical ID)
```

**Drugs:**
```
Drug candidates
  --> E05_drug_enricher (RxCUI, MeSH, DrugBank)
  --> E16_drug_combination_parser (parse combinations)
  --> E17_entity_deduplicator (merge by canonical drug name)
```

**Genes:**
```
Gene candidates
  --> E15_genetic_enricher (Orphadata gene-disease links)
  --> E18_gene_enricher (PubTator3 gene data)
  --> E17_entity_deduplicator (merge by HGNC symbol)
```

### Caching Strategy

All API-based enrichers use disk caching to avoid redundant network calls:

| Enricher | Cache TTL | Rate Limit |
|----------|-----------|------------|
| PubTator3 (diseases, drugs, genes) | 7 days | 3 req/sec |
| ClinicalTrials.gov (NCT) | 30 days | Per API guidelines |
| Citation validation | 7 days | Per API guidelines |

Caching is implemented in `Z_utils/Z01_api_client.py` (`BaseAPIClient`), which all API enrichers extend.

### Deduplication Strategy

Deduplication happens at multiple levels:

1. **Span deduplication** (`E11`): Merges overlapping NER spans from different sources (e.g., FlashText found "complement C3" while scispacy found "C3"). Keeps the most specific match.
2. **Entity deduplication** (`E07`, `E17`): Merges entities with the same canonical identifier. Combines mention counts, page references, and evidence spans. Uses quality ranking to select the best representative.

### NER Model Loading

NER models (`E08`, `E09`, `E10`) use lazy loading to avoid startup overhead:

```python
# Model is loaded on first call to enrich(), not at import time
enricher = EpiExtractEnricher(config={"enabled": True})
# No model loaded yet
result = enricher.enrich(disease_entity)
# Model loaded here on first invocation
```

### Graceful API Failure

All API enrichers handle failures without crashing the pipeline:

- Network errors: Logged and skipped; entity retains pre-enrichment state.
- Rate limit errors: Automatic retry with exponential backoff.
- Invalid responses: Logged and skipped.
- The entity is always returned (enriched or not), never dropped due to API failure.

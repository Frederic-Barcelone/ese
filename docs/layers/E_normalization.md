# Layer E: Normalization and Enrichment

## Purpose

`E_normalization/` standardizes validated entities: maps to ontology codes, resolves ambiguities, deduplicates, and enriches via external APIs and NER models. After this layer, every entity carries standard identifiers (MONDO, RxNorm, HGNC, MeSH, etc.) and duplicates are merged.

18 modules. All enrichers inherit from `BaseEnricher[InputT, OutputT]`. API enrichers use disk caching and rate limiting. NER models lazy-load.

See also: [Data Flow](../architecture/02_data_flow.md) | [Domain Models](../architecture/03_domain_models.md)

---

## Key Modules

### Core Normalization

| Module | Description |
|--------|-------------|
| `E01_term_mapper.py` | Abbreviation long-form canonicalization (whitespace, case, punctuation, Unicode). |
| `E02_disambiguator.py` | Resolves `SHORT_FORM_ONLY` abbreviations using bag-of-words document context voting. |
| `E03_disease_normalizer.py` | Maps identifier lists to canonical ICD-10, SNOMED, MONDO, ORPHA codes. |
| `E07_deduplicator.py` | Merges duplicate abbreviations by `short_form`. Quality ranking: DEFINITION_PAIR > GLOSSARY_ENTRY > SHORT_FORM_ONLY. |
| `E11_span_deduplicator.py` | Merges overlapping NER spans from multiple sources. |
| `E17_entity_deduplicator.py` | General entity dedup (diseases, drugs, genes) by canonical ID, merging mentions and pages. |

### API Enrichment

| Module | Description |
|--------|-------------|
| `E04_pubtator_enricher.py` | `DiseaseEnricher`, `PubTator3Client`. PubTator3 for diseases: MeSH IDs, normalized names, aliases. 7-day cache, 3 req/sec. |
| `E05_drug_enricher.py` | `DrugEnricher`. PubTator3 for drugs: MeSH and alias lookup. |
| `E06_nct_enricher.py` | ClinicalTrials.gov: trial title, conditions, interventions, status. 30-day cache. |
| `E18_gene_enricher.py` | `GeneEnricher`. PubTator3 for genes: identifiers and aliases. |

### NER Enrichment

| Module | Description |
|--------|-------------|
| `E08_epi_extract_enricher.py` | EpiExtract4GARD-v2: rare disease epidemiology (LOC, EPI, STAT entities). |
| `E09_zeroshot_bioner.py` | ZeroShotBioNER: ADE, dosage, frequency, route extraction. |
| `E10_biomedical_ner_all.py` | d4data/biomedical-ner-all: 84 biomedical entity types. |

### Specialized Enrichment

| Module | Description |
|--------|-------------|
| `E12_patient_journey_enricher.py` | Diagnostic delay, treatment lines, progression milestones. |
| `E13_registry_enricher.py` | Clinical trial registry cross-referencing. |
| `E14_citation_validator.py` | PMID/DOI validation via PubMed and CrossRef. |
| `E15_genetic_enricher.py` | Orphadata gene-disease associations (disease-causing, susceptibility, modifier). |
| `E16_drug_combination_parser.py` | Decomposes "Drug A + Drug B" into individual entities. |

---

## Per-Entity Normalization Flow

**Abbreviations:** E01 (canonical form) -> E02 (disambiguate) -> E07 (dedup by SF) -> E11 (span dedup)

**Diseases:** E03 (primary codes) -> E04 (PubTator3 MeSH) -> E17 (dedup by canonical ID)

**Drugs:** E05 (RxCUI, MeSH, DrugBank) -> E16 (parse combinations) -> E17 (dedup)

**Genes:** E15 (Orphadata links) -> E18 (PubTator3) -> E17 (dedup by HGNC symbol)

---

## Caching and Rate Limits

| Enricher | Cache TTL | Rate Limit |
|----------|-----------|------------|
| PubTator3 (diseases, drugs, genes) | 7 days | 3 req/sec |
| ClinicalTrials.gov (NCT) | 30 days | Per API guidelines |
| Citation validation | 7 days | Per API guidelines |

Implemented in `Z_utils/Z01_api_client.py` (`DiskCache`, `RateLimiter`, `BaseAPIClient`).

## Graceful API Failure

All API enrichers handle failures without crashing. Network errors and invalid responses are logged and skipped. Entities are always returned (enriched or not), never dropped due to API failure.

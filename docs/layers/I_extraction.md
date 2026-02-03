# I_extraction -- Entity Processing

## Purpose

Layer I coordinates the extraction, validation, enrichment, and deduplication of domain-specific entities (diseases, genes, drugs, pharma companies, authors, citations) and clinical trial feasibility information. It sits between the generators (C) and exporters (J), consuming detection results and producing validated, enriched entity objects.

## Modules

### I01_entity_processors.py

Unified entity extraction orchestrator that handles six entity types through a consistent detect-validate-enrich-deduplicate workflow.

**Class: `EntityProcessor`**

```python
processor = EntityProcessor(
    run_id=run_id,
    pipeline_version="1.0.0",
    disease_detector=disease_detector,
    disease_normalizer=disease_normalizer,
    drug_detector=drug_detector,
    gene_detector=gene_detector,
    pharma_detector=pharma_detector,
    author_detector=author_detector,
    citation_detector=citation_detector,
)
# Optional enrichers (set after init)
processor.disease_enricher = disease_enricher
processor.drug_enricher = drug_enricher
processor.gene_enricher = gene_enricher
```

**Processing methods:**

| Method | Returns | Workflow |
|--------|---------|---------|
| `process_diseases(doc, pdf_path)` | `List[ExtractedDisease]` | Detect (C06) -> auto-validate -> normalize (E03) -> PubTator enrich (E04) -> deduplicate (E17) |
| `process_genes(doc, pdf_path)` | `List[ExtractedGene]` | Detect (C16) -> auto-validate -> PubTator enrich (E18) -> deduplicate (E17) |
| `process_drugs(doc, pdf_path)` | `List[ExtractedDrug]` | Detect (C07) -> auto-validate by source -> PubTator enrich (E05) -> deduplicate (E17) |
| `process_pharma(doc, pdf_path)` | `List[ExtractedPharma]` | Detect (C18) -> validate candidates |
| `process_authors(doc, pdf_path, full_text)` | `List[ExtractedAuthor]` | Detect (C13) -> validate candidates |
| `process_citations(doc, pdf_path, full_text)` | `List[ExtractedCitation]` | Detect (C14) -> validate candidates |

**Validation strategy:** Specialized lexicon matches (PAH, ANCA, IgAN for diseases; Alexion, investigational, compound ID for drugs) are auto-validated with higher confidence. General lexicon matches receive lower confidence scores.

**Entity creation helpers:**

- `create_entity_from_candidate(candidate, status, confidence, reason, flags, raw_response, long_form_override)` -- Delegates to `Z_utils.Z11_entity_helpers`.
- `create_entity_from_search(doc_id, full_text, match, long_form, field_type, confidence, flags, rule_version, lexicon_source)` -- Creates entity from regex match.

**Internal entity conversion methods:**

- `_create_disease_entity(candidate, status_validated, confidence, flags)` -- Extracts ICD-10, ICD-11, SNOMED, MONDO, ORPHA, UMLS, MeSH codes from candidate identifiers.
- `_create_gene_entity(candidate, ...)` -- Extracts HGNC, Entrez, Ensembl, OMIM, UniProt identifiers.
- `_candidate_to_extracted_drug(candidate, ...)` -- Converts `DrugCandidate` with drug-specific fields (drug_class, mechanism, development_phase, sponsor, conditions).

**Deduplication:** All entity types use `EntityDeduplicator` (E17) which groups by canonical identifier and merges mention counts/page lists.

### I02_feasibility_processor.py

Multi-stage feasibility information extraction combining LLM-based and NER-based approaches.

**Class: `FeasibilityProcessor`**

```python
processor = FeasibilityProcessor(
    run_id=run_id,
    feasibility_detector=detector,          # C08 pattern-based
    llm_feasibility_extractor=llm_extractor, # C11 LLM-based
    epi_enricher=epi_enricher,              # E08
    zeroshot_bioner=zeroshot,               # E09
    biomedical_ner=biomed_ner,              # E10
    patient_journey_enricher=pj_enricher,   # E12
    registry_enricher=reg_enricher,         # E13
    genetic_enricher=gen_enricher,          # E15
)
candidates = processor.process(doc, pdf_path, full_text)
```

**Processing pipeline (`process()`):**

1. **Base extraction:** LLM-based (C11, preferred) or pattern-based (C08, fallback). Returns `FeasibilityCandidate` objects.
2. **EpiExtract4GARD enrichment:** Rare disease epidemiology NER. Extracts locations, epi types, statistics. Adds `NERCandidate` with `epidemiology_data`.
3. **ZeroShotBioNER enrichment:** Extracts adverse events, dosages, frequencies, routes, durations. Category labels: `adverse_event`, `drug_dosage`, `drug_frequency`, `drug_route`, `treatment_duration`.
4. **BiomedicalNER enrichment:** 84 clinical entity types from d4data model. Maps to categories: `symptom`, `diagnostic_procedure`, `therapeutic_procedure`, `lab_value`, `outcome`, `demographics_*`.
5. **PatientJourneyNER enrichment:** Diagnostic delays, treatment lines, care pathway steps, surveillance frequencies, pain points, recruitment touchpoints.
6. **RegistryNER enrichment:** Registry names, sizes, geographic coverage, data types, access policies, eligibility criteria. Reports linked known registries.
7. **GeneticNER enrichment:** Gene symbols, HGVS variants, rsIDs, HPO terms, ORDO disease codes. Reports gene-variant associations.
8. **Span deduplication:** Removes overlapping NER spans, keeping highest confidence. Uses `E11_span_deduplicator.deduplicate_feasibility_candidates()`.

**Helper functions:**

- `_make_ner_candidate(category, text, confidence, source, evidence_text, **kwargs)` -- Creates `NERCandidate` with common defaults.
- `_add_ner_entities(candidates, entities, category, source, text_attr, evidence_attr)` -- Batch adds NER entities to candidates list.

**Return type:** `List[Union[FeasibilityCandidate, NERCandidate]]` -- Mixed types: structured feasibility data and NER-extracted entities.

## Usage Patterns

```python
# Entity processing
processor = EntityProcessor(run_id=run_id, pipeline_version="0.7",
                           disease_detector=factory.create_disease_detector())
processor.disease_enricher = factory.create_disease_enricher()
diseases = processor.process_diseases(doc, pdf_path)
drugs = processor.process_drugs(doc, pdf_path)

# Feasibility processing
feas_processor = FeasibilityProcessor(
    run_id=run_id,
    feasibility_detector=factory.create_feasibility_detector(),
    llm_feasibility_extractor=factory.create_llm_feasibility_extractor(claude_client),
    epi_enricher=factory.create_epi_enricher(),
)
feasibility = feas_processor.process(doc, pdf_path, full_text)
```

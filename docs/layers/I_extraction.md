# Layer I: Entity Processing

## Purpose

Coordinates extraction, validation, enrichment, and deduplication of domain entities (diseases, genes, drugs, pharma, authors, citations) and feasibility data. Sits between generators (C) and exporters (J). 2 modules.

---

## Modules

### I01_entity_processors.py

`EntityProcessor`: unified entity orchestrator with detect-validate-enrich-deduplicate workflow.

```python
processor = EntityProcessor(
    run_id=run_id, pipeline_version="0.8",
    disease_detector=disease_detector,
    drug_detector=drug_detector,
    gene_detector=gene_detector, ...
)
processor.disease_enricher = disease_enricher
```

| Method | Workflow |
|--------|---------|
| `process_diseases(doc, pdf_path)` | Detect (C06) -> validate -> normalize (E03) -> PubTator (E04) -> dedup (E17) |
| `process_genes(doc, pdf_path)` | Detect (C16) -> validate -> PubTator (E18) -> dedup (E17) |
| `process_drugs(doc, pdf_path)` | Detect (C07) -> validate by source -> PubTator (E05) -> dedup (E17) |
| `process_pharma(doc, pdf_path)` | Detect (C18) -> validate |
| `process_authors(doc, pdf_path, full_text)` | Detect (C13) -> validate |
| `process_citations(doc, pdf_path, full_text)` | Detect (C14) -> validate |

### I02_feasibility_processor.py

`FeasibilityProcessor`: multi-stage feasibility extraction combining LLM and NER.

```python
processor = FeasibilityProcessor(
    run_id=run_id,
    feasibility_detector=detector,          # C08
    llm_feasibility_extractor=llm_extractor, # C11
    epi_enricher=epi_enricher,              # E08
    zeroshot_bioner=zeroshot,               # E09
    biomedical_ner=biomed_ner, ...          # E10
)
candidates = processor.process(doc, pdf_path, full_text)
```

**Processing stages:**
1. LLM-based (C11, preferred) or pattern-based (C08, fallback)
2. EpiExtract4GARD: epidemiology NER (locations, statistics)
3. ZeroShotBioNER: adverse events, dosages, frequencies, routes
4. BiomedicalNER: 84 clinical entity types
5. PatientJourneyNER: diagnostic delays, treatment lines, care steps
6. RegistryNER: registry names, sizes, geographic coverage
7. GeneticNER: gene symbols, HGVS variants, HPO terms
8. Span deduplication (E11)

Returns `List[Union[FeasibilityCandidate, NERCandidate]]`.

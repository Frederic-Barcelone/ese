# Adding a New Entity Type

Adding a new entity type touches multiple pipeline layers:

```
A_core/ (models) -> C_generators/ (detection) -> E_normalization/ (enrichment)
    -> H_pipeline/ (factory) -> I_extraction/ (processing) -> J_export/ (output)
```

## Step 1: Domain Model (A_core/)

Create `A_core/AXX_entity_models.py` with Pydantic models:

1. **Identifier model** -- codes from standard ontologies
2. **Provenance model** -- extends `BaseProvenanceMetadata`
3. **Candidate model** -- pre-validation entity with context
4. **Extracted model** -- validated entity with codes and evidence
5. **Export models** -- simplified structures for JSON output

Key patterns:
- `ConfigDict(extra="forbid")` catches field name typos
- `model_validator(mode="after")` for non-empty field checks
- Provenance and confidence on all entities
- No optional fields without defaults
- Reference: `A05_disease_models.py`, `A06_drug_models.py`, `A19_gene_models.py`

## Step 2: Generator (C_generators/)

Create `C_generators/CXX_strategy_entity.py`. Two interfaces in `A_core/A02_interfaces.py`:

- **`BaseCandidateGenerator`** -- Implement `generator_type` and `extract(doc_structure) -> List[Candidate]`
- **`BaseExtractor`** -- Implement `strategy_id`, `entity_type` and `extract(doc_graph, ctx, config) -> List[RawExtraction]`

Design principles:
- Optimize for **high recall** -- FPs acceptable at this stage
- Use **FlashText** for lexicon matching (not regex for large vocabularies)
- Iterate via `doc_graph.iter_linear_blocks()` or `doc_graph.iter_tables()`

Optionally add a FP filter in `CXX_entity_fp_filter.py`:
- Confidence adjustments preferred over hard filtering
- Hard filter only for catastrophic FPs
- See `C24_disease_fp_filter.py`, `C25_drug_fp_filter.py`, `C34_gene_fp_filter.py`

## Step 3: Enricher (E_normalization/, if needed)

Implement `BaseEnricher[InputT, OutputT]` from `A_core/A02_interfaces.py`:
- `enricher_name` property
- `enrich(entity: InputT) -> OutputT`
- Optionally override `enrich_batch()` for batch optimization

Common patterns:
- PubTator3 (`E04_pubtator_enricher.py`): MeSH codes, aliases
- NCT (`E06_nct_enricher.py`): Trial metadata
- Deduplication by canonical ID (`E17_entity_deduplicator.py`)
- Handle API failures gracefully -- return entity unmodified on error

## Step 4: Register in H01_component_factory.py

```python
def create_entity_detector(self) -> "EntityDetector":
    from C_generators.CXX_strategy_entity import EntityDetector
    return EntityDetector(config={"run_id": self.run_id, "lexicon_base_path": str(self.dict_path)})
```

Add TYPE_CHECKING imports at the top.

## Step 5: Processing Method (I01_entity_processors.py)

Add to `EntityProcessor`: detector + optional enricher as constructor params, then a processing method:

```python
def process_entities(self, doc, pdf_path):
    candidates = self.detector.extract(doc)           # 1. Generate (high recall)
    results = [self._create_entity(c) for c in candidates]  # 2. Convert
    if self.normalizer:
        results = self.normalizer.normalize_batch(results)   # 3. Normalize
    if self.enricher:
        results = self.enricher.enrich_batch(results)        # 4. Enrich
    results = EntityDeduplicator().deduplicate(results)      # 5. Deduplicate
    return results
```

## Step 6: Export Handler (J_export/)

Add to `J01a_entity_exporters.py` (entities) or `J01b_metadata_exporters.py` (metadata). Follow existing patterns using export models from Step 1.

## Step 7: Wire Up (orchestrator.py)

1. Create detector/enricher via `ComponentFactory`
2. Pass to `EntityProcessor`
3. Add extraction call in processing loop
4. Add export call
5. Add to relevant extraction presets in `config.yaml`
6. Add toggle: `extraction_pipeline.extractors.your_entity: true`

## Step 8: Tests (K_tests/)

Cover: model validation, generator detection, FP filter, enricher (mocked API), deduplication, export schema.

## Reference Files

| Step | Example Files |
|------|--------------|
| Domain models | `A05_disease_models.py`, `A06_drug_models.py`, `A19_gene_models.py` |
| Generators | `C06_strategy_disease.py`, `C07_strategy_drug.py`, `C16_strategy_gene.py` |
| FP filters | `C24_disease_fp_filter.py`, `C25_drug_fp_filter.py`, `C34_gene_fp_filter.py` |
| Enrichers | `E04_pubtator_enricher.py`, `E05_drug_enricher.py`, `E18_gene_enricher.py` |
| Deduplication | `E17_entity_deduplicator.py` |
| Factory | `H01_component_factory.py` |
| Processors | `I01_entity_processors.py` |
| Exporters | `J01a_entity_exporters.py`, `J01b_metadata_exporters.py` |

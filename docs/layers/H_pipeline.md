# Layer H: Pipeline Orchestration

## Purpose

Coordinates assembly and execution of pipeline components: factory for creating components from config, abbreviation pipeline with PASO heuristics, visual extraction integration, and deterministic merge/resolve for deduplication. 4 modules.

---

## Modules

### H01_component_factory.py

`ComponentFactory` creates all pipeline components from YAML config.

```python
factory = ComponentFactory(config, run_id, pipeline_version, log_dir, api_key)
```

Resolves `base_path` via `get_base_path()` (`CORPUS_BASE_PATH` env var or auto-detect).

| Method | Returns |
|--------|---------|
| `create_parser()` | `PDFToDocGraphParser` |
| `create_generators()` | 5 generators (C01-C05) |
| `create_claude_client(model)` | `Optional[ClaudeClient]` |
| `create_llm_engine(claude_client, model)` | `Optional[LLMEngine]` |
| `create_disease_detector()` | `DiseaseDetector` |
| `create_drug_detector()` | `DrugDetector` |
| `create_gene_detector()` | `GeneDetector` |
| `create_disease_enricher()` | `Optional[DiseaseEnricher]` (PubTator) |
| `create_drug_enricher()` | `Optional[DrugEnricher]` (PubTator) |
| `create_gene_enricher()` | `Optional[GeneEnricher]` (PubTator) |
| `create_feasibility_detector()` | `FeasibilityDetector` |
| `create_llm_feasibility_extractor(...)` | `Optional[LLMFeasibilityExtractor]` |

All enricher/detector methods return `None` when the feature is disabled.

### H02_abbreviation_pipeline.py

`AbbreviationPipeline`: multi-stage abbreviation extraction with PASO heuristics.

| Stage | Method | Description |
|-------|--------|-------------|
| 1 | `parse_pdf()` | PDF to DocumentGraph, extract tables |
| 2 | `generate_candidates()` | Run generators, deduplicate by (SF, LF) |
| 3 | `filter_candidates()` | Corroboration, lexicon reduction, SF validation |
| 4 | `apply_heuristics()` | PASO auto-approve/reject, returns `(auto_results, llm_candidates)` |
| 5 | `validate_with_llm()` | Fast-reject (Haiku) + batch validation |
| 6 | `search_missing_abbreviations()` | PASO C + direct search |
| 7 | `extract_sf_only_with_llm()` | PASO D: SF-only extraction (Haiku) |
| 8 | `normalize_results()` | Normalize, NCT enrich, disambiguate, deduplicate |

### H03_visual_integration.py

`VisualPipelineIntegration`: visual extraction pipeline wrapper (tables and figures).

```python
integration = VisualPipelineIntegration(config)
if integration.enabled:
    result = integration.extract(pdf_path)
    exported = integration.export(result, output_dir, doc_name)
```

Detection modes: `layout-aware`, `vlm-only`, `hybrid`, `heuristic`.

### H04_merge_resolver.py

`MergeResolver`: deterministic deduplication for overlapping extractions from multiple strategies.

```python
resolver = MergeResolver(MergeConfig.default())
merged = resolver.merge(raw_extractions)
```

Algorithm: sort by stable key -> group by dedup key -> resolve by strategy priority -> merge evidence -> enforce mutual exclusivity.

---

## Usage

```python
factory = ComponentFactory(config, run_id, version, log_dir)
pipeline = AbbreviationPipeline(
    run_id=run_id, parser=factory.create_parser(),
    generators=factory.create_generators(), ...
)
doc = pipeline.parse_pdf(pdf_path, output_dir)
candidates, full_text = pipeline.generate_candidates(doc)
```

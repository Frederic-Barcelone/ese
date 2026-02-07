# ESE Documentation

**ESE (Entity & Structure Extraction)** -- Production-grade pipeline for extracting structured metadata from clinical trial and rare disease PDFs. Pipeline v0.8.

---

## Performance & Benchmarks

| Document | Description |
|----------|-------------|
| [Performance & Benchmarks](performance_benchmarks.md) | 1,482 tests, gold standard results (CADEC F1=93.2%, NLP4RARE F1=74.6%), throughput, cost |
| [Held-Out Evaluation](evaluation/held_out_evaluation_report.md) | Generalization: CADEC 82.6%, NLP4RARE diseases 76.4%, abbreviations 61.6% |
| [Quality Metrics Dossier](quality_metrics_dossier.md) | 4 gold standards, evaluation results, test coverage by layer |
| [Future Work](future_work.md) | Accuracy targets, cost optimization, new entity types |

## Deep Dives

| Document | Description |
|----------|-------------|
| [Entity Detection](entity_detection_deep_dive.md) | Detection strategies per entity type |
| [Visual Extraction](visual_extraction_deep_dive.md) | Tables & figures: DocLayout-YOLO, Docling TableFormer, VLM analysis |
| [LLM Usage Strategy](llm_usage_deep_dive.md) | 17 call sites, 2-tier routing (Haiku/Sonnet), cost tracking |
| [Extracted Entities Reference](extracted_entities_reference.md) | All 14+ entity types with fields, enums, JSON output format |
| [Claude Code Setup](claude_code_setup.md) | CLAUDE.md manifest, plugins, development workflows |

## Architecture

| Document | Description |
|----------|-------------|
| [Pipeline Overview](architecture/01_overview.md) | Layer philosophy, directory structure, technology stack |
| [Data Flow](architecture/02_data_flow.md) | Entity lifecycle, PASO heuristics, extraction flows |
| [Domain Models](architecture/03_domain_models.md) | Pydantic models: Candidate, ExtractedEntity, provenance |

## Layers

| Document | Description |
|----------|-------------|
| [A_core](layers/A_core.md) | Domain models, interfaces, provenance, exceptions |
| [B_parsing](layers/B_parsing.md) | PDF to DocumentGraph, table/figure extraction, layout detection |
| [C_generators](layers/C_generators.md) | Candidate generation (syntax, lexicon, regex, LLM) |
| [D_validation](layers/D_validation.md) | LLM verification, prompt registry, quote verifier |
| [E_normalization](layers/E_normalization.md) | Term mapping, PubTator/NCT enrichment, deduplication |
| [F_evaluation](layers/F_evaluation.md) | Gold standard loading, precision/recall/F1 scoring |
| [G_config](layers/G_config.md) | config.yaml reference |
| [H_pipeline](layers/H_pipeline.md) | Abbreviation pipeline, component factory |
| [I_extraction](layers/I_extraction.md) | Entity and feasibility processors |
| [J_export](layers/J_export.md) | JSON export handlers |
| [Z_utils](layers/Z_utils.md) | API client, text helpers, image utilities |

## Guides

| Document | Description |
|----------|-------------|
| [Getting Started](guides/01_getting_started.md) | Environment setup, dependencies, first run |
| [Adding an Entity Type](guides/02_adding_entity_type.md) | End-to-end: model, generator, validator, exporter |
| [Configuration](guides/03_configuration.md) | config.yaml reference, extraction presets, API settings |
| [Evaluation](guides/04_evaluation.md) | Gold standard format, scoring methodology |
| [Cost Optimization](guides/05_cost_optimization.md) | Model tier routing, prompt caching, usage tracking |
| [Gene Evaluation](guides/06_gene_evaluation.md) | NLM-Gene + RareDisGene benchmarks |
| [Drug Evaluation](guides/07_drug_evaluation.md) | CADEC benchmark (F1=93.2%) |

## Reference

| Document | Description |
|----------|-------------|
| [Lexicons](reference/01_lexicons.md) | Lexicon sources, term counts, file formats |
| [External APIs](reference/02_external_apis.md) | Claude API, PubTator3, ClinicalTrials.gov |
| [Output Format](reference/03_output_format.md) | JSON output schema per entity type |

## Reports

| Document | Description |
|----------|-------------|
| [Performance Analysis](performance_analysis.md) | Bottlenecks, throughput, cost breakdown |
| [Error Handling Analysis](error_handling_analysis.md) | Exception hierarchy, resilience patterns |
| [Testing Strategy Analysis](testing_strategy_analysis.md) | Test suite structure, coverage gaps |

## Plans

| Document | Description |
|----------|-------------|
| [Visual Extraction Redesign](plans/2026-01-31-visual-extraction-redesign.md) | Visual extraction pipeline redesign |
| [Layout-Aware Visual Extraction](plans/2026-02-02-layout-aware-visual-extraction.md) | Layout-aware visual detection |
| [A_core Refactoring](plans/2026-02-02-a-core-refactoring.md) | Core domain models refactoring |
| [B_parsing Refactoring](plans/2026-02-02-b-parsing-refactoring.md) | PDF parsing layer refactoring |
| [Z_utils Refactor Design](plans/2026-02-02-z-utils-refactor-design.md) | Utilities layer refactoring |

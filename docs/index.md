# ESE Documentation

**ESE (Entity & Structure Extraction)** is a production-grade 6-layer pipeline for extracting structured metadata from clinical trial and medical PDF documents, with a focus on rare disease research.

---

## Performance & Benchmarks

| Document | Description |
|----------|-------------|
| [Performance & Benchmarks](performance_benchmarks.md) | Test suite health (1,474 tests), all gold standard results (CADEC F1=93.2%, BC2GM P=90.3%, NLP4RARE F1=75.7%), throughput, cost profile, SOTA comparison |

## Future Work

| Document | Description |
|----------|-------------|
| [Future Work & Roadmap](future_work.md) | Short/medium/long-term improvements: accuracy targets, cost optimization, scalability, testing infrastructure, new entity types, research directions |

---

## Architecture

Foundational design, data flow, and type system.

| Document | Description |
|----------|-------------|
| [Pipeline Overview](architecture/01_overview.md) | Layer philosophy, directory structure, technology stack, orchestrator stages |
| [Data Flow](architecture/02_data_flow.md) | Entity lifecycle, PASO heuristics, extraction flows per entity type, presets |
| [Domain Models](architecture/03_domain_models.md) | Pydantic models from A_core: Candidate, ExtractedEntity, enums, provenance |

## Layers

Per-layer documentation for each directory in `corpus_metadata/`.

| Document | Description |
|----------|-------------|
| [A_core](layers/A_core.md) | Domain models, interfaces, provenance utilities, exceptions |
| [B_parsing](layers/B_parsing.md) | PDF to DocumentGraph, table/figure extraction, layout detection |
| [C_generators](layers/C_generators.md) | Candidate generation strategies (syntax, lexicon, regex, LLM) |
| [D_validation](layers/D_validation.md) | LLM-based verification, prompt registry, quote verifier |
| [E_normalization](layers/E_normalization.md) | Term mapping, PubTator/NCT enrichment, deduplication |
| [F_evaluation](layers/F_evaluation.md) | Gold standard loading, precision/recall/F1 scoring |
| [G_config](layers/G_config.md) | Pipeline configuration (config.yaml, config keys) |
| [H_pipeline](layers/H_pipeline.md) | Pipeline orchestration, abbreviation pipeline, component factory |
| [I_extraction](layers/I_extraction.md) | Entity and feasibility processors |
| [J_export](layers/J_export.md) | JSON export handlers for all entity types |
| [Z_utils](layers/Z_utils.md) | API client, text helpers, image utilities, download scripts |

## Guides

Step-by-step guides for common tasks.

| Document | Description |
|----------|-------------|
| [Getting Started](guides/01_getting_started.md) | Environment setup, dependencies, first pipeline run |
| [Adding an Entity Type](guides/02_adding_entity_type.md) | End-to-end walkthrough: model, generator, validator, exporter |
| [Configuration](guides/03_configuration.md) | config.yaml reference, extraction presets, API settings |
| [Evaluation](guides/04_evaluation.md) | Gold standard format, scoring methodology, interpreting results |
| [Cost Optimization](guides/05_cost_optimization.md) | Model tier routing, prompt caching, usage tracking, adding new LLM call sites |
| [Gene Evaluation](guides/06_gene_evaluation.md) | BC2GM gene benchmark: setup, results (P=90.3%), FP/FN analysis, methodology |
| [Drug Evaluation](guides/07_drug_evaluation.md) | CADEC drug benchmark: setup, results (F1=93.2%), improvement trajectory, error patterns |

## Reference

Detailed reference material for external integrations and output formats.

| Document | Description |
|----------|-------------|
| [Lexicons](reference/01_lexicons.md) | Loaded lexicon sources, term counts, file formats |
| [External APIs](reference/02_external_apis.md) | Claude API, PubTator3, ClinicalTrials.gov, Unstructured.io |
| [Output Format](reference/03_output_format.md) | JSON output schema per entity type, directory structure |

## Reports

Analysis reports and quality assessments.

| Document | Description |
|----------|-------------|
| [Quality Metrics Dossier](quality_metrics_dossier.md) | Full dossier: test suite (1,474 tests), 4 gold standards, evaluation results, coverage by layer |
| [Performance Analysis](performance_analysis.md) | Processing bottlenecks, throughput measurements, cost breakdown |
| [Error Handling Analysis](error_handling_analysis.md) | Exception hierarchy, resilience patterns, identified gaps |
| [Testing Strategy Analysis](testing_strategy_analysis.md) | Test suite structure, coverage gaps, SOTA comparison, recommendations |

## Plans

Design documents and architectural plans for pipeline improvements.

| Document | Description |
|----------|-------------|
| [Visual Extraction Redesign](plans/2026-01-31-visual-extraction-redesign.md) | Redesign of the visual extraction pipeline |
| [Layout-Aware Visual Extraction](plans/2026-02-02-layout-aware-visual-extraction.md) | Layout-aware approach for visual detection |
| [A_core Refactoring](plans/2026-02-02-a-core-refactoring.md) | Refactoring plan for core domain models |
| [B_parsing Refactoring](plans/2026-02-02-b-parsing-refactoring.md) | Refactoring plan for PDF parsing layer |
| [Z_utils Refactor Design](plans/2026-02-02-z-utils-refactor-design.md) | Refactoring plan for utilities layer |

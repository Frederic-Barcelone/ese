# Pipeline Overview

ESE processes clinical PDF documents through a layered extraction pipeline. Each layer has a distinct responsibility and quality goal, enabling the system to balance high recall at generation time with high precision at validation time.

## Pipeline Diagram

```
                         +------------------+
                         |    Source PDF     |
                         +--------+---------+
                                  |
                                  v
                    +-------------+-------------+
                    |     B_parsing (Layer 2)    |
                    |   PDF -> DocumentGraph     |
                    |  pages, blocks, tables,    |
                    |  images, layout detection   |
                    +-------------+-------------+
                                  |
                                  v
                    +-------------+-------------+
                    |   C_generators (Layer 3)   |
                    |   Candidate Generation     |
                    |  syntax, regex, FlashText,  |
                    |  scispacy NER, LLM, VLM    |
                    +-------------+-------------+
                                  |
                                  v
                    +-------------+-------------+
                    |   D_validation (Layer 4)   |
                    |   LLM Verification         |
                    |  Claude API, PASO heuristics|
                    +-------------+-------------+
                                  |
                                  v
                    +-------------+-------------+
                    |  E_normalization (Layer 5)  |
                    |   Standardization          |
                    |  PubTator3, NCT, ontology   |
                    |  mapping, deduplication     |
                    +-------------+-------------+
                                  |
                                  v
                    +-------------+-------------+
                    |  I_extraction (Layer 9)     |
                    |   Entity Processing        |
                    |  entity + feasibility       |
                    |  processors                 |
                    +-------------+-------------+
                                  |
                                  v
                    +-------------+-------------+
                    |    J_export (Layer 10)      |
                    |   JSON Export              |
                    |  per-entity type output     |
                    +-------------+-------------+
```

## Layer Philosophy

| Layer | Goal | Strategy |
|-------|------|----------|
| **B_parsing** | Faithful document representation | PDF parsing, layout detection, table/figure extraction |
| **C_generators** | **High Recall** | Exhaustive candidate extraction; noise is acceptable |
| **D_validation** | **High Precision** | Claude LLM filters false positives aggressively |
| **E_normalization** | **Standardization** | Map to ontologies (MONDO, RxNorm, HGNC), deduplicate |
| **I_extraction** | Entity assembly | Orchestrate per-entity-type processing pipelines |
| **J_export** | Structured output | JSON files with provenance, evidence, and audit trail |

## Directory Structure

All pipeline code resides under `corpus_metadata/`.

| Directory | Files | Purpose |
|-----------|-------|---------|
| `A_core/` | 24 | Domain models (Pydantic), interfaces, provenance, exceptions, enums |
| `B_parsing/` | 31 | PDF to DocumentGraph, table extraction, figure detection, layout analysis |
| `C_generators/` | 35 | Candidate generation strategies (syntax, regex, FlashText lexicons, LLM, VLM) |
| `D_validation/` | 4 | LLM verification engine, prompt registry, quote verifier, validation logger |
| `E_normalization/` | 18 | Term mapping, disambiguation, PubTator/NCT enrichment, deduplication |
| `F_evaluation/` | 3 | Gold standard loading, precision/recall scorer, evaluation runner |
| `G_config/` | 1 | `G01_config_keys.py` (+ `config.yaml` configuration file) |
| `H_pipeline/` | 4 | Component factory, abbreviation pipeline, visual integration, merge resolver |
| `I_extraction/` | 2 | Entity processors (`I01`) and feasibility processor (`I02`) |
| `J_export/` | 4 | Export handlers: entity exporters, metadata exporters, visual export |
| `Z_utils/` | 11 | API client, text helpers, image utilities, path utilities, download scripts |
| `K_tests/` | varies | Pytest test suite |
| `orchestrator.py` | 1 | Main entry point |

## Technology Stack

| Technology | Version | Role |
|------------|---------|------|
| Python | 3.12+ | Runtime |
| Pydantic | v2 | Domain models, validation, serialization |
| Claude API (Anthropic) | claude-sonnet-4 | LLM validation, Vision analysis, feasibility extraction |
| Unstructured.io / Docling | latest | PDF parsing backends |
| PyMuPDF (fitz) | latest | Native PDF figure extraction, rendering |
| scispacy | latest | Biomedical NER with UMLS linking |
| FlashText | latest | Fast multi-pattern lexicon matching (600K+ terms) |
| Rich | latest | Console output, progress bars, tables |
| YAML | stdlib | Pipeline configuration |

## Orchestrator

The `orchestrator.py` entry point processes each PDF through 16 stages:

1. **PDF Parsing** -- PDF to DocumentGraph (via Unstructured.io or Docling)
2. **Candidate Generation** -- Abbreviation extraction (syntax, regex, lexicon, layout)
3. **LLM Validation** -- PASO heuristic filtering + Claude API abbreviation validation
4. **Normalization** -- Abbreviation normalization and deduplication
5. **Disease Detection** -- FlashText + scispacy NER, FP filtering, PubTator3 enrichment
6. **Gene Detection** -- HGNC lexicon + pattern matching, PubTator3 enrichment
7. **Drug Detection** -- RxNorm/ChEMBL lexicon, FP filtering, enrichment
8. **Pharma Detection** -- Pharmaceutical company identification
9. **Author Detection** -- Author name and affiliation extraction
10. **Citation Detection** -- PMID, DOI, NCT identifier extraction and API validation
11. **Feasibility Extraction** -- Pattern-based + LLM-based + NER enrichment pipeline
12. **Care Pathway Extraction** -- Clinical treatment algorithm extraction
13. **Recommendation Extraction** -- Guideline recommendation extraction (LLM + VLM)
14. **Visual Extraction** -- Table and figure detection with VLM triage + Claude Vision
15. **Document Metadata** -- Document type classification, title, date extraction
16. **Export & Summary** -- JSON export for all entity types with metrics

Each stage is independently configurable through `G_config/config.yaml` and can be enabled or disabled via extraction presets.

## Related Documentation

- [Data Flow](02_data_flow.md) -- detailed entity lifecycle and per-type flows
- [Domain Models](03_domain_models.md) -- Pydantic model reference
- [Configuration](../guides/03_configuration.md) -- config.yaml reference

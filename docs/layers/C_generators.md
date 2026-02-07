# Layer C: Candidate Generation

## Purpose

`C_generators/` is the high-recall extraction layer. Extracts every plausible candidate from the `DocumentGraph`, accepting false positives. Precision is handled downstream by `D_validation/` and FP filters within this layer.

35 modules with multiple strategies per entity type: syntax, regex, lexicon (FlashText 600K+ terms), NER (scispacy), LLM (Claude), VLM (Claude Vision). All generators implement `BaseCandidateGenerator` or `BaseExtractor` from `A_core/A02_interfaces.py`.

See also: [Data Flow](../architecture/02_data_flow.md) | [Pipeline Overview](../architecture/01_overview.md)

---

## Key Modules

### Abbreviation Strategies

| Module | Description |
|--------|-------------|
| `C01_strategy_abbrev.py` | Schwartz-Hearst algorithm for parenthetical SF/LF pairs. |
| `C20_abbrev_patterns.py` | Regex patterns for abbreviation-like structures. |
| `C23_inline_definition_detector.py` | Inline definitions: "TNF, defined as..." or "TNF (i.e., ...)". |

### Pattern and Layout Strategies

| Module | Description |
|--------|-------------|
| `C00_strategy_identifiers.py` | Identifier regex: PMID, DOI, NCT, OMIM, ORPHA. |
| `C02_strategy_regex.py` | Rigid regex for trial IDs, dosages, dates. |
| `C03_strategy_layout.py` | Spatial extraction using bounding box coordinates. |

### Lexicon Strategies

| Module | Description |
|--------|-------------|
| `C04_strategy_flashtext.py` | FlashText matching (600K+ terms) + scispacy NER. Lexicons: Meta-Inventory 65K, MONDO 97K, RxNorm 132K, Orphanet 9.5K, ChEMBL 23K, trial acronyms 125K. |
| `C05_strategy_glossary.py` | Glossary table detection and extraction. |
| `C22_lexicon_loaders.py` | Lexicon loading for all FlashText dictionaries. |

### Entity-Specific Strategies

| Module | Description |
|--------|-------------|
| `C06_strategy_disease.py` | `DiseaseDetector`. Multi-layer: rare disease lexicons + general lexicons + scispacy NER. |
| `C07_strategy_drug.py` | `DrugDetector`. Multi-layer: FDA/RxNorm/ChEMBL lexicons + compound ID patterns + scispacy CHEMICAL. |
| `C16_strategy_gene.py` | `GeneDetector`. HGNC symbol lexicon + gene pattern matching + scispacy GENE. |
| `C09_strategy_document_metadata.py` | Classification, date extraction. Haiku tier. |
| `C13_strategy_author.py` | Author extraction from headers and investigator lists. |
| `C14_strategy_citation.py` | PMID, DOI, NCT, URL regex + reference section parsing. |
| `C18_strategy_pharma.py` | Pharmaceutical company name detection. |

### Feasibility Strategies

| Module | Description |
|--------|-------------|
| `C08_strategy_feasibility.py` | `FeasibilityDetector`. Pattern-based multi-category feasibility extraction. |
| `C11_llm_feasibility.py` | `LLMFeasibilityExtractor`. LLM-based feasibility via Claude. Sonnet tier. |
| `C27_feasibility_patterns.py` | Regex pattern library. |
| `C29_feasibility_prompts.py` | Structured prompts for LLM extraction. |
| `C30_feasibility_response_parser.py` | LLM JSON response parser. |

### Vision Strategies

| Module | Description |
|--------|-------------|
| `C10_vision_image_analysis.py` | Claude Vision: flowcharts, CONSORT, study design. Multiple call_types (Sonnet for analysis, Haiku for classification). |
| `C15_vlm_table_extractor.py` | VLM table structure extraction. |
| `C17_flowchart_graph_extractor.py` | Flowchart node-edge extraction. Sonnet tier. |
| `C19_vlm_visual_enrichment.py` | VLM title/description generation. Haiku tier. |

### Recommendation Strategies

| Module | Description |
|--------|-------------|
| `C12_guideline_recommendation_extractor.py` | Guideline recommendation extraction. |
| `C31_recommendation_patterns.py` | Regex for recommendation detection. |
| `C32_recommendation_llm.py` | LLM-based recommendation extraction. Sonnet tier. |
| `C33_recommendation_vlm.py` | VLM-based recommendation extraction. Sonnet tier. |

### False Positive Filters

| Module | Description |
|--------|-------------|
| `C21_noise_filters.py` | General noise filtering. |
| `C24_disease_fp_filter.py` | Disease FP removal (generic terms, anatomy, procedures). |
| `C25_drug_fp_filter.py` | Drug FP removal. |
| `C26_drug_fp_constants.py` | Drug FP blocklist. |
| `C28_feasibility_fp_filter.py` | Feasibility FP filtering. |
| `C34_gene_fp_filter.py` | Gene FP removal (words colliding with gene symbols: IT, ALL, WAS). |

---

## Public Interfaces

- **`BaseCandidateGenerator`** -- Abbreviation strategies (C01, C04, C05). Returns `List[Candidate]`.
- **`BaseExtractor`** -- Entity strategies (C06, C07, C16, etc.). Returns `List[RawExtraction]`.

### Multi-Strategy Pipelines

**Disease** (C06): FlashText (MONDO 97K + Orphanet 9.5K) + scispacy NER + C24 FP filter.

**Drug** (C07): FlashText (RxNorm 132K + ChEMBL 23K + FDA) + compound ID regex + scispacy CHEMICAL + C25 FP filter.

**Gene** (C16): FlashText (HGNC symbols) + pattern matching + scispacy GENE + C34 FP filter.

### Adding a New Generator

1. Create module in `C_generators/` implementing `BaseCandidateGenerator` or `BaseExtractor`.
2. Define candidate model in `A_core/` if needed.
3. Optionally create a FP filter.
4. Register in `H_pipeline/H01_component_factory.py`.
5. Wire into `orchestrator.py`.

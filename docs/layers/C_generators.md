# Layer C: Candidate Generation

## Purpose

`C_generators/` is the high-recall extraction layer. Its goal is to extract every plausible candidate entity from the `DocumentGraph`, accepting false positives as an expected cost. Precision is handled downstream by `D_validation/` and false-positive filters within this layer.

This layer contains 35 modules implementing multiple extraction strategies per entity type: syntax-based, regex-based, lexicon-based (FlashText with 600K+ terms), NER-based (scispacy), LLM-based (Claude), and VLM-based (Claude Vision).

All generators implement the `BaseCandidateGenerator` interface from `A_core/A02_interfaces.py` and output typed candidate lists with `GeneratorType` provenance tracking.

See also: [Data Flow](../architecture/02_data_flow.md) | [Pipeline Overview](../architecture/01_overview.md)

---

## Key Modules

### Abbreviation Strategies

| Module | Description |
|--------|-------------|
| `C01_strategy_abbrev.py` | Schwartz-Hearst algorithm for explicit SF/LF parenthetical pairs (e.g., "Tumor Necrosis Factor (TNF)"). Primary high-precision abbreviation source. |
| `C20_abbrev_patterns.py` | Regex patterns for abbreviation-like structures. |
| `C23_inline_definition_detector.py` | Inline definition detection for patterns like "TNF, defined as..." or "TNF (i.e., Tumor Necrosis Factor)". |

### Pattern-Based Strategies

| Module | Description |
|--------|-------------|
| `C00_strategy_identifiers.py` | Structured identifier extraction via regex: PMID, DOI, NCT, OMIM, ORPHA numbers. |
| `C02_strategy_regex.py` | Rigid regex patterns for trial IDs, dosage expressions, date formats. |
| `C03_strategy_layout.py` | Spatial extraction using `DocumentGraph` bounding box coordinates for structured page regions. |

### Lexicon-Based Strategies

| Module | Description |
|--------|-------------|
| `C04_strategy_flashtext.py` | FlashText multi-pattern matching (600K+ terms) combined with scispacy biomedical NER. Lexicons include Meta-Inventory (65K), MONDO (97K), RxNorm (132K), Orphanet (9.5K), ChEMBL (23K), and trial acronyms (125K). |
| `C05_strategy_glossary.py` | Glossary table detection and extraction (abbreviation definition tables). |
| `C22_lexicon_loaders.py` | Lexicon loading utilities for all FlashText dictionaries. |

### Entity-Specific Strategies

| Module | Description |
|--------|-------------|
| `C06_strategy_disease.py` | Multi-layer disease detection: specialized rare disease lexicons, general disease lexicons, scispacy NER. Outputs `DiseaseCandidate` with identifiers. |
| `C07_strategy_drug.py` | Multi-layer drug detection: Alexion pipeline drugs, investigational compounds (compound ID patterns like LNP023), FDA approved, RxNorm lexicon, scispacy CHEMICAL entities. Outputs `DrugCandidate`. |
| `C16_strategy_gene.py` | Gene detection: HGNC symbol lexicon, scispacy GENE entities, gene symbol pattern matching. Outputs `GeneCandidate`. |
| `C09_strategy_document_metadata.py` | Document classification, date extraction, metadata assembly. Uses `call_type="document_classification"` and `call_type="description_extraction"` (both Haiku tier). |
| `C18_strategy_pharma.py` | Pharmaceutical company name detection. |
| `C13_strategy_author.py` | Author extraction from paper headers, author blocks, and investigator lists. |
| `C14_strategy_citation.py` | Citation extraction: PMID, DOI, NCT, URL regex patterns plus reference section parsing. |

### Feasibility Strategies

| Module | Description |
|--------|-------------|
| `C08_strategy_feasibility.py` | Multi-category pattern-based feasibility extraction (eligibility, screening flow, study design, operational burden, epidemiology, study footprint). |
| `C11_llm_feasibility.py` | LLM-based feasibility extraction via Claude API for complex or ambiguous feasibility data. Uses `call_type="feasibility_extraction"` (Sonnet tier). |
| `C27_feasibility_patterns.py` | Regex pattern library for feasibility categories. |
| `C29_feasibility_prompts.py` | Structured prompts for LLM-based feasibility extraction. |
| `C30_feasibility_response_parser.py` | JSON response parser for LLM feasibility output. |

### Vision Strategies

| Module | Description |
|--------|-------------|
| `C10_vision_image_analysis.py` | Claude Vision analysis of flowcharts, CONSORT diagrams, and study design figures. Per-caller `call_type` routing: `flowchart_analysis` (Sonnet), `chart_analysis` (Sonnet), `vlm_table_extraction` (Sonnet), `image_classification` (Haiku), `ocr_text_fallback` (Haiku). |
| `C15_vlm_table_extractor.py` | VLM-based table structure extraction and correction. |
| `C17_flowchart_graph_extractor.py` | Flowchart/algorithm structure extraction from figures (node-edge graph analysis). Uses `record_api_usage()` with `call_type="flowchart_analysis"` (Sonnet tier). |
| `C19_vlm_visual_enrichment.py` | VLM enrichment: title generation, description, and semantic classification for visual elements. Uses `resolve_model_tier("vlm_visual_enrichment")` (Haiku tier) with `record_api_usage()`. |

### Recommendation Strategies

| Module | Description |
|--------|-------------|
| `C12_guideline_recommendation_extractor.py` | Clinical guideline recommendation extraction. |
| `C31_recommendation_patterns.py` | Regex patterns for recommendation detection. |
| `C32_recommendation_llm.py` | LLM-based recommendation extraction via Claude. Uses `record_api_usage()` with `call_type="recommendation_extraction"` (Sonnet tier). |
| `C33_recommendation_vlm.py` | VLM-based recommendation extraction from figures. Uses `record_api_usage()` with `call_type="recommendation_vlm"` (Sonnet tier). |

### False Positive Filters

| Module | Description |
|--------|-------------|
| `C21_noise_filters.py` | General noise filtering (common non-entity patterns). |
| `C24_disease_fp_filter.py` | Disease false positive removal (generic medical terms, anatomical terms, procedure names). |
| `C25_drug_fp_filter.py` | Drug false positive removal. |
| `C26_drug_fp_constants.py` | Blocklist of known drug false positive strings. |
| `C28_feasibility_fp_filter.py` | Feasibility false positive filtering. |
| `C34_gene_fp_filter.py` | Gene false positive removal (common English words that collide with gene symbols like IT, ALL, WAS). |

---

## Public Interfaces

All generators implement one of two interfaces defined in `A_core/A02_interfaces.py`. See [A_core Layer](A_core.md#public-interfaces) for full interface signatures.

- **`BaseCandidateGenerator`** -- Used by abbreviation strategies (C01, C04, C05). Returns `List[Candidate]`.
- **`BaseExtractor`** -- Used by entity-specific strategies (C06, C07, C16, etc.). Returns `List[RawExtraction]` with `ExecutionContext`.

### GeneratorType Enum

Tracks which strategy produced each candidate:

| Value | Strategy | Module |
|-------|----------|--------|
| `gen:syntax_pattern` | Schwartz-Hearst abbreviations | C01 |
| `gen:glossary_table` | Glossary tables | C05 |
| `gen:rigid_pattern` | DOI, trial IDs, doses | C02 |
| `gen:table_layout` | Spatial extraction | C03 |
| `gen:lexicon_match` | FlashText dictionary matching | C04 |
| `gen:inline_definition` | Inline SF=LF definitions | C23 |

---

## Usage Patterns

### Standard Entity Extraction

```python
from C_generators.C06_strategy_disease import DiseaseDetector

detector = DiseaseDetector(config=config)
candidates = detector.extract(doc_graph)
# Returns List[DiseaseCandidate] with high recall
```

### Multi-Strategy Pipeline

Each entity type combines multiple strategies for maximum recall:

**Disease detection** (C06):
1. FlashText lexicon matching against MONDO (97K terms) and Orphanet (9.5K terms)
2. scispacy biomedical NER (DISEASE entity type)
3. False positive filtering via `C24_disease_fp_filter`

**Drug detection** (C07):
1. FlashText against RxNorm (132K), ChEMBL (23K), FDA approved drugs
2. Compound ID regex patterns (e.g., LNP023, ALXN1720)
3. scispacy NER (CHEMICAL entity type)
4. False positive filtering via `C25_drug_fp_filter`

**Gene detection** (C16):
1. FlashText against HGNC symbol lexicon
2. Gene symbol pattern matching
3. scispacy NER (GENE entity type)
4. False positive filtering via `C34_gene_fp_filter`

### False Positive Filter Pattern

Filters run after candidate generation but before validation. They use blocklists and heuristic rules to remove obvious noise without consuming LLM API calls.

```
Candidates (high recall, noisy)
  |
  v
FP Filter (blocklist + heuristic rules)
  |
  v
Filtered candidates --> D_validation or E_normalization
```

### Adding a New Generator

1. Create a new module in `C_generators/` (e.g., `C35_strategy_new_entity.py`).
2. Implement `BaseCandidateGenerator` or `BaseExtractor` interface.
3. Define the candidate model in `A_core/` if needed.
4. Optionally create a false positive filter (e.g., `C36_new_entity_fp_filter.py`).
5. Register in `H_pipeline/H01_component_factory.py`.
6. Wire into `orchestrator.py`.

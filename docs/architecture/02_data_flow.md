# Data Flow

How data moves through the ESE pipeline, from raw PDF bytes to structured JSON output.

## Entity Lifecycle

Every extracted entity follows this progression (stages, not class names):

```
PDF bytes
  |
  v
DocumentGraph          [B_parsing]
  pages, blocks (TextBlock, Table, ImageBlock)
  |
  v
Candidate              [C_generators]
  noisy pre-verification object, high recall
  |
  v
Validated              [D_validation]
  LLM-verified: VALIDATED / REJECTED / AMBIGUOUS
  |
  v
Enriched               [E_normalization]
  ontology codes, deduplicated, disambiguated
  |
  v
Processed              [I_extraction]
  evidence, mention counts, page references
  |
  v
JSON Export            [J_export]
  structured output with provenance
```

---

## Abbreviation Flow

### Generation

```
DocumentGraph
  |
  +---> C01_strategy_abbrev (Schwartz-Hearst syntax)
  +---> C02_strategy_regex (rigid regex)
  +---> C03_strategy_layout (spatial table extraction)
  +---> C04_strategy_flashtext (lexicon: Meta-Inventory, trial acronyms)
  +---> C05_strategy_glossary (glossary table detection)
  +---> C23_inline_definition_detector (SF=LF inline patterns)
  |
  v
Candidate pool (high recall, noisy)
```

### PASO Heuristic Filtering

```
Candidate pool
  |
  v
PASO A: Auto-approve known statistical terms (CI, HR, SD, OR, etc.)
  |
  v
PASO B: Auto-reject country codes in geographic context (US, UK, EU)
  |
  v
PASO C: Auto-enrich hyphenated abbreviations via ClinicalTrials.gov
  |
  v
PASO D: LLM SF-only extraction for candidates without definitions
  |
  v
Remaining candidates --> D_validation (LLM verification)
```

### Post-Validation

```
Validated abbreviations
  |
  v
E_normalization
  +---> E01_term_mapper (canonical form)
  +---> E02_disambiguator (resolve ambiguous short forms)
  +---> E07_deduplicator (merge duplicates)
  +---> E11_span_deduplicator (consolidate overlapping mentions)
  |
  v
J_export --> abbreviations_<doc>_<timestamp>.json
```

---

## Disease Flow

```
DocumentGraph
  |
  v
C06_strategy_disease
  +---> FlashText lexicon (MONDO 97K, Orphanet 9.5K)
  +---> scispacy NER
  |
  v
C24_disease_fp_filter (removes generic terms, anatomical terms)
  |
  v
E_normalization
  +---> E03_disease_normalizer (canonical codes)
  +---> E04_pubtator_enricher (MeSH codes, aliases)
  +---> E17_entity_deduplicator (merge by canonical ID)
  |
  v
J_export --> diseases_<doc>_<timestamp>.json
```

---

## Drug Flow

```
DocumentGraph
  |
  v
C07_strategy_drug
  +---> FlashText lexicon (RxNorm 132K, ChEMBL 23K, FDA approved)
  +---> Compound ID regex (e.g., LNP023, ALXN1720)
  +---> scispacy NER (CHEMICAL)
  |
  v
C25_drug_fp_filter / C26_drug_fp_constants
  |
  v
E_normalization
  +---> E05_drug_enricher (RxCUI, MeSH, DrugBank)
  +---> E16_drug_combination_parser
  +---> E17_entity_deduplicator
  |
  v
J_export --> drugs_<doc>_<timestamp>.json
```

---

## Gene Flow

```
DocumentGraph
  |
  v
C16_strategy_gene
  +---> FlashText (HGNC symbols, Orphadata gene-disease)
  +---> Gene symbol pattern matching
  +---> scispacy NER (GENE)
  |
  v
C34_gene_fp_filter (common words colliding with gene symbols)
  |
  v
E_normalization
  +---> E15_genetic_enricher (Orphadata disease associations)
  +---> E18_gene_enricher (PubTator3)
  +---> E17_entity_deduplicator (merge by HGNC symbol)
  |
  v
J_export --> genes_<doc>_<timestamp>.json
```

---

## Feasibility Flow

```
DocumentGraph
  |
  v
C08_strategy_feasibility (pattern-based)
  +---> C27_feasibility_patterns (regex)
  +---> C28_feasibility_fp_filter
  +---> B08_eligibility_parser
  |
  v
C11_llm_feasibility (LLM-based, Sonnet tier)
  +---> C29_feasibility_prompts
  +---> C30_feasibility_response_parser
  |
  v
I02_feasibility_processor --> J_export --> feasibility_<doc>_<timestamp>.json
```

---

## Visual Extraction Flow

```
PDF
  |
  v
B_parsing
  +---> B09/B24 (PyMuPDF native figures)
  +---> B22 (YOLO layout detection)
  +---> B28 (Docling table extraction)
  |
  v
B16_triage: SKIP | CHEAP_PATH | VLM_REQUIRED
  |
  v
VLM_REQUIRED --> C10/C15/C19 (Claude Vision)
  |
  v
H03_visual_integration --> J_export
  +---> figures_<doc>_<timestamp>.json
  +---> tables_<doc>_<timestamp>.json
```

---

## Other Entity Flows

### Pharma Companies
`C18_strategy_pharma` --> `E17_entity_deduplicator` --> `J_export`

### Authors and Citations
```
DocumentGraph
  +---> C13_strategy_author --> ExtractedAuthor --> J_export
  +---> C14_strategy_citation --> E14_citation_validator --> J_export
```

### Care Pathways
`C17_flowchart_graph_extractor` (VLM) --> `J_export`

### Recommendations
`C12_guideline_recommendation_extractor` (regex + LLM + VLM) --> `J_export`

### Document Metadata
`C09_strategy_document_metadata` (classification, title, date) --> `J_export`

---

## Extraction Presets

The `config.yaml` `extraction_pipeline.preset` controls which entity types are extracted:

| Preset | Entities Extracted |
|--------|--------------------|
| `standard` | Drugs, diseases, genes, abbreviations, feasibility, tables, figures, care pathways, recommendations |
| `all` | All entity types including authors, citations, metadata |
| `minimal` | Abbreviations only |
| `entities_only` | Drugs, diseases, genes, abbreviations |
| `clinical_entities` | Drugs, diseases only |
| `metadata_only` | Authors, citations, document metadata |
| `images_only` | Tables, figures |

Single-entity presets: `drugs_only`, `diseases_only`, `genes_only`, `abbreviations_only`, `feasibility_only`, `tables_only`.

See [Configuration Guide](../guides/03_configuration.md) for details.

---

## Related Documentation

- [Pipeline Overview](01_overview.md) -- layer philosophy and directory structure
- [Domain Models](03_domain_models.md) -- Pydantic model reference
- [Configuration](../guides/03_configuration.md) -- config.yaml reference

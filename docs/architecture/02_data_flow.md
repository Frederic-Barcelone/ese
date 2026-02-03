# Data Flow

This document describes how data moves through the ESE pipeline, from raw PDF bytes to structured JSON output.

## Entity Lifecycle

Every extracted entity follows the same progression through pipeline layers:

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
ValidatedCandidate     [D_validation]
  LLM-verified with VALIDATED / REJECTED / AMBIGUOUS status
  |
  v
EnrichedEntity         [E_normalization]
  mapped to ontology codes, deduplicated, disambiguated
  |
  v
ProcessedEntity        [I_extraction]
  assembled with evidence, mention counts, page references
  |
  v
JSON Export            [J_export]
  structured output with provenance and audit trail
```

Each entity type (abbreviation, disease, drug, gene, etc.) follows this lifecycle with type-specific models and strategies.

---

## Abbreviation Flow

Abbreviations have the most complex flow due to the PASO heuristic system.

### Generation

```
DocumentGraph
  |
  +---> C01_strategy_abbrev (Schwartz-Hearst syntax patterns)
  +---> C02_strategy_regex (rigid regex patterns)
  +---> C03_strategy_layout (spatial table extraction)
  +---> C04_strategy_flashtext (lexicon matching: Meta-Inventory, trial acronyms)
  +---> C05_strategy_glossary (glossary table detection)
  +---> C23_inline_definition_detector (SF=LF inline patterns)
  |
  v
Candidate pool (high recall, noisy)
```

### PASO Heuristic Filtering

Before LLM validation, candidates pass through deterministic PASO heuristic rules:

```
Candidate pool
  |
  v
PASO A: Statistical Abbreviation Auto-Approve
  Known statistical terms (CI, HR, SD, OR, RR, IQR, BMI, etc.)
  are auto-approved without LLM verification.
  |
  v
PASO B: Country Code Blacklist
  Two-letter codes matching ISO country codes (US, UK, EU, etc.)
  are auto-rejected when they appear in geographic context.
  |
  v
PASO C: Hyphenated Abbreviations
  Abbreviations with hyphens (anti-C5, IL-6) are auto-enriched
  with definitions from ClinicalTrials.gov NCT metadata.
  |
  v
PASO D: LLM SF-Only Extraction
  For SHORT_FORM_ONLY candidates without definitions in the document,
  the LLM attempts to infer the long form from context.
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
  +---> E01_term_mapper (canonical form normalization)
  +---> E02_disambiguator (resolve ambiguous short forms)
  +---> E07_deduplicator (merge duplicate entries)
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
  +---> FlashText lexicon matching (MONDO: 97K terms, Orphanet: 9.5K terms)
  +---> scispacy NER (biomedical entity recognition)
  |
  v
DiseaseCandidate pool
  |
  v
C24_disease_fp_filter
  False positive filtering (removes generic medical terms,
  anatomical terms, procedure names)
  |
  v
E_normalization
  +---> E03_disease_normalizer (canonical disease names)
  +---> E04_pubtator_enricher (MeSH codes, aliases from PubTator3)
  +---> E17_entity_deduplicator (merge by canonical ID)
  |
  v
ExtractedDisease (with ICD-10, SNOMED, MONDO, ORPHA codes)
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
  +---> FlashText lexicon matching (RxNorm: 132K, ChEMBL: 23K, FDA approved)
  +---> Compound ID regex patterns (e.g., LNP023, ALXN1720)
  +---> scispacy NER (CHEMICAL entity type)
  |
  v
DrugCandidate pool
  |
  v
C25_drug_fp_filter
  False positive filtering (removes common non-drug terms)
  C26_drug_fp_constants (blocklist of false positive strings)
  |
  v
E_normalization
  +---> E05_drug_enricher (RxCUI, MeSH, DrugBank mapping)
  +---> E04_pubtator_enricher (MeSH aliases)
  +---> E16_drug_combination_parser (parse combination therapies)
  +---> E17_entity_deduplicator (merge by canonical drug name)
  |
  v
ExtractedDrug (with RxCUI, MeSH, DrugBank, development phase)
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
  +---> FlashText lexicon matching (HGNC symbols, Orphadata gene-disease)
  +---> Gene symbol pattern matching
  +---> scispacy NER (GENE entity type)
  |
  v
GeneCandidate pool
  |
  v
C34_gene_fp_filter
  False positive filtering (removes common English words
  that collide with gene symbols)
  |
  v
E_normalization
  +---> E15_genetic_enricher (HGNC, Entrez, Ensembl mapping)
  +---> E18_gene_enricher (Orphadata disease associations)
  +---> E17_entity_deduplicator (merge by HGNC symbol)
  |
  v
ExtractedGene (with HGNC, Entrez, Ensembl IDs, disease linkages)
  |
  v
J_export --> genes_<doc>_<timestamp>.json
```

---

## Feasibility Flow

Feasibility extraction uses both pattern-based and LLM-based strategies.

```
DocumentGraph
  |
  v
C08_strategy_feasibility (pattern-based)
  +---> C27_feasibility_patterns (regex for eligibility, screening, endpoints)
  +---> C28_feasibility_fp_filter (remove false positives)
  +---> B08_eligibility_parser (structured criteria parsing)
  |
  v
C11_llm_feasibility (LLM-based)
  +---> C29_feasibility_prompts (structured prompts for Claude)
  +---> C30_feasibility_response_parser (parse LLM JSON responses)
  |
  v
FeasibilityCandidate pool
  (eligibility criteria, screening flow, study design,
   epidemiology, operational burden, study footprint)
  |
  v
I02_feasibility_processor
  Assembles candidates into structured feasibility output
  |
  v
J_export --> feasibility_<doc>_<timestamp>.json
```

---

## Visual Extraction Flow

Figures and tables follow a multi-stage pipeline with VLM triage.

```
PDF
  |
  v
B_parsing
  +---> B09_pdf_native_figures / B24_native_figure_extraction (PyMuPDF raster/vector)
  +---> B22_doclayout_detector (YOLO-based layout detection)
  +---> B28_docling_backend (Docling table extraction with TableFormer)
  |
  v
VisualCandidate pool
  |
  v
B16_triage
  Routing decision: SKIP | CHEAP_PATH | VLM_REQUIRED
  Based on area ratio, grid structure, caption presence
  |
  v
VLM_REQUIRED candidates
  +---> C10_vision_image_analysis (Claude Vision classification)
  +---> C15_vlm_table_extractor (VLM table structure correction)
  +---> C19_vlm_visual_enrichment (title, description generation)
  |
  v
ExtractedVisual (with caption, reference, table structure)
  |
  v
H03_visual_integration (document-level resolution)
  +---> B15_caption_extractor (caption assignment)
  +---> B11_extraction_resolver (multi-page continuation)
  |
  v
J_export
  +---> figures_<doc>_<timestamp>.json
  +---> tables_<doc>_<timestamp>.json
```

---

## Pharma Company Flow

```
DocumentGraph
  |
  v
C18_strategy_pharma
  +---> FlashText lexicon matching (pharma company names)
  +---> Pattern matching for company name suffixes
  |
  v
PharmaCandidate pool
  |
  v
E17_entity_deduplicator (merge duplicates)
  |
  v
ExtractedPharma (company name, headquarters, therapeutic areas)
  |
  v
J_export --> pharma_<doc>_<timestamp>.json
```

---

## Author and Citation Flow

```
DocumentGraph
  |
  +---> C13_strategy_author
  |       Header patterns, affiliation blocks, investigator lists
  |       |
  |       v
  |     AuthorCandidate --> ExtractedAuthor
  |       |
  |       v
  |     J_export --> authors_<doc>_<timestamp>.json
  |
  +---> C14_strategy_citation
          Regex for PMID, DOI, NCT, URL
          Reference section parsing
          |
          v
        CitationCandidate --> E14_citation_validator (API validation)
          |
          v
        ExtractedCitation
          |
          v
        J_export --> citations_<doc>_<timestamp>.json
```

---

## Care Pathway Flow

```
DocumentGraph
  |
  v
C17_flowchart_graph_extractor
  VLM-based clinical flowchart analysis
  Extracts treatment algorithms, decision trees
  |
  v
CarePathway (nodes, edges, taper schedules)
  |
  v
J_export --> care_pathways_<doc>_<timestamp>.json
```

---

## Recommendation Flow

```
DocumentGraph
  |
  v
C12_guideline_recommendation_extractor
  +---> C31_recommendation_patterns (regex for guidelines, evidence levels)
  +---> C32_recommendation_llm (Claude-based extraction)
  +---> C33_recommendation_vlm (Vision-based extraction from tables/figures)
  |
  v
RecommendationSet
  (guideline name, organization, evidence levels,
   recommendation strength, target condition)
  |
  v
J_export --> recommendations_<doc>_<timestamp>.json
```

---

## Document Metadata Flow

```
DocumentGraph
  |
  v
C09_strategy_document_metadata
  +---> Document type classification (article, protocol, guidelines, etc.)
  +---> Title extraction (header analysis, PDF metadata)
  +---> Date extraction (publication date, PDF creation date)
  +---> DOI extraction (regex patterns)
  |
  v
DocumentMetadata (type, title, date, DOI, confidence)
  |
  v
J_export --> metadata_<doc>_<timestamp>.json
```

---

## Extraction Presets

The `config.yaml` `extraction_pipeline.preset` field controls which entity types are extracted:

| Preset | Entities Extracted |
|--------|--------------------|
| `standard` | Drugs, diseases, genes, abbreviations, feasibility, tables, figures, care pathways, recommendations |
| `all` | All entity types including visuals, authors, citations, document metadata |
| `minimal` | Abbreviations only (no LLM) |
| `drugs_only` | Drugs |
| `diseases_only` | Diseases |
| `genes_only` | Genes |
| `abbreviations_only` | Abbreviations |
| `feasibility_only` | Feasibility data |
| `entities_only` | Drugs, diseases, genes, abbreviations |
| `clinical_entities` | Drugs, diseases only |
| `metadata_only` | Authors, citations, document metadata |
| `images_only` | Tables, figures/visuals |
| `tables_only` | Table extraction only (no figures) |

Each preset toggles individual extractors in the `extractors` configuration section. See [Configuration Guide](../guides/03_configuration.md) for details.

---

## Related Documentation

- [Pipeline Overview](01_overview.md) -- layer philosophy and directory structure
- [Domain Models](03_domain_models.md) -- detailed Pydantic model reference
- [Configuration](../guides/03_configuration.md) -- config.yaml reference

# Output Format Reference

JSON output files per entity type with timestamps. Export logic in `J01_export_handlers.py`, `J01a_entity_exporters.py`, `J01b_metadata_exporters.py`, `J02_visual_export.py`.

## Directory Structure

Processing `document.pdf` creates:

```
document/
  abbreviations_document_YYYYMMDD_HHMMSS.json
  diseases_document_YYYYMMDD_HHMMSS.json
  drugs_document_YYYYMMDD_HHMMSS.json
  genes_document_YYYYMMDD_HHMMSS.json
  feasibility_document_YYYYMMDD_HHMMSS.json
  tables_document_YYYYMMDD_HHMMSS.json
  figures_document_YYYYMMDD_HHMMSS.json
  metadata_document_YYYYMMDD_HHMMSS.json
  document_extracted_text_YYYYMMDD_HHMMSS.txt
  document_figure_page3_1.png
```

Naming: `{entity_type}_{doc_stem}_{YYYYMMDD}_{HHMMSS}.json`. Timestamp is UTC.

## Common Envelope

Every export file includes:

```json
{
  "run_id": "RUN_20250203_143045_abc123",
  "timestamp": "2025-02-03T14:30:45.123456",
  "document": "document.pdf",
  "document_path": "/absolute/path/to/document.pdf",
  "pipeline_version": "0.8"
}
```

## Entity Types

**Abbreviations** -- `short_form`, `long_form`, `field_type` (DEFINITION_PAIR, GLOSSARY_ENTRY, SHORT_FORM_ONLY), `generator_type`, `validation_status`, `confidence`, `evidence_spans`, `provenance`. Statistics: `total_candidates`, `validated`, `rejected`, `ambiguous`.

**Diseases** -- `matched_text`, `preferred_label`, `is_rare_disease`, `category`, `codes` (icd10, icd11, snomed, mondo, orpha, umls, mesh), `all_identifiers`, `mention_count`, `pages_mentioned`, `mesh_aliases`, `pubtator_normalized_name`.

**Drugs** -- `preferred_name`, `brand_name`, `compound_id`, `is_investigational`, `drug_class`, `mechanism`, `development_phase`, `sponsor`, `conditions`, `nct_id`, `codes` (rxcui, mesh, ndc, drugbank, unii), `mention_count`, `pages_mentioned`.

**Genes** -- `hgnc_symbol`, `full_name`, `is_alias`, `locus_type`, `chromosome`, `codes` (hgnc_id, entrez, ensembl, uniprot, omim), `associated_diseases`, `mention_count`, `pages_mentioned`.

**Authors** -- `full_name`, `role`, `affiliation`, `email`, `orcid`, `confidence`.

**Citations** -- `pmid`, `pmcid`, `doi`, `nct`, `url`, `citation_text`, `citation_number`. Includes API validation results.

**Pharma Companies** -- `matched_text`, `canonical_name`, `full_name`, `headquarters`, `parent_company`, `subsidiaries`.

**Feasibility** -- Trial identifiers, eligibility criteria with parsed lab values, CONSORT screening flow, study design, epidemiology, patient journey.

**Care Pathways** -- Treatment algorithm graphs: nodes (type, text, phase, drugs), edges (source, target, condition), pathway-level fields.

**Recommendations** -- Guideline recommendations with dosing, evidence levels, strength grades, population targeting.

**Document Metadata** -- File info, PDF metadata, classification (type code/name/group), LLM-generated title/descriptions, primary date.

**Visuals** -- Tables: structured data (headers, rows, cells). Figures: VLM classification and description. Both include page, bounding boxes, caption, optional base64 image data. Images saved as `{doc_name}_{type}_page{N}_{idx}.png`.

## JSON Formatting

- 2-space indentation, UTF-8 with `ensure_ascii=False`
- Pydantic `model_dump_json()` for entity exports
- Set `export_combined: true` in config for a single combined JSON

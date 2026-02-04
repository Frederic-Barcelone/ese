# J_export -- Export

## Purpose

Layer J serializes extraction results to timestamped JSON files, one per entity type per document. It handles all entity types, visual elements (figures and tables with image files), feasibility data, document metadata, care pathways, and guideline recommendations.

## Modules

### J01_export_handlers.py

Main export orchestrator managing all export operations through the `ExportManager` class.

**Class: `ExportManager`**

```python
manager = ExportManager(
    run_id=run_id,
    pipeline_version="0.8",
    output_dir=None,          # Optional override (default: {pdf_dir}/{pdf_stem}/)
    gold_json=None,           # Path to gold standard for evaluation
    claude_client=claude_client,  # Optional, for Vision LLM analysis
)
```

**Output directory:** Creates `{pdf_stem}/` folder in the same directory as the PDF. Override with `output_dir` parameter.

**Export methods:**

| Method | Output file | Content |
|--------|------------|---------|
| `export_results(pdf_path, results, candidates, counters, ...)` | `abbreviations_{stem}_{ts}.json` | Abbreviations + inline disease/drug/pharma summaries |
| `export_disease_results(pdf_path, results)` | `diseases_{stem}_{ts}.json` | Delegates to J01a |
| `export_gene_results(pdf_path, results)` | `genes_{stem}_{ts}.json` | Delegates to J01a |
| `export_drug_results(pdf_path, results)` | `drugs_{stem}_{ts}.json` | Delegates to J01a |
| `export_pharma_results(pdf_path, results)` | `pharma_{stem}_{ts}.json` | Delegates to J01a |
| `export_author_results(pdf_path, results)` | `authors_{stem}_{ts}.json` | Delegates to J01a |
| `export_citation_results(pdf_path, results)` | `citations_{stem}_{ts}.json` | Delegates to J01a (includes API validation) |
| `export_feasibility_results(pdf_path, results, doc)` | `feasibility_{stem}_{ts}.json` | Trial IDs, eligibility, epidemiology, patient journey, endpoints, sites, study design, operational burden, screening flow |
| `export_images(pdf_path, doc)` | `figures_{stem}_{ts}.json` + PNG files | Vision LLM analysis (flowchart, chart, table, unknown), OCR extraction |
| `export_tables(pdf_path, doc)` | `tables_{stem}_{ts}.json` + PNG files | Table images with re-rendering, PyMuPDF bbox detection |
| `export_extracted_text(pdf_path, doc)` | `{stem}_{ts}.txt` | Full extracted text with page markers |
| `export_document_metadata(pdf_path, metadata)` | `metadata_{stem}_{ts}.json` | Delegates to J01b |
| `export_care_pathways(pdf_path, pathways)` | `care_pathways_{stem}_{ts}.json` | Delegates to J01b |
| `export_recommendations(pdf_path, recs)` | `recommendations_{stem}_{ts}.json` | Delegates to J01b |

**Image rendering:** `render_figure_with_padding(pdf_path, page_num, bbox, dpi, padding, ...)` re-renders figures from PDF using PyMuPDF with configurable padding for captions/legends. Handles coordinate scaling, two-column layout detection, and table-specific PyMuPDF bbox detection.

**Vision analysis:** When `claude_client` is provided, images are analyzed by type: flowcharts (patient flow extraction), charts (data point extraction, taper schedules), tables (structure extraction), and unknown images (auto-classification then type-specific analysis).

### J01a_entity_exporters.py

Per-entity-type export functions, each creating Pydantic export documents serialized with `model_dump_json()`.

**Functions:**

| Function | Export model | Key fields |
|----------|-------------|-----------|
| `export_disease_results(out_dir, pdf_path, results, run_id, version)` | `DiseaseExportDocument` / `DiseaseExportEntry` | matched_text, preferred_label, codes (icd10, icd11, snomed, mondo, orpha, umls, mesh), is_rare_disease, mention_count, mesh_aliases, pubtator_normalized_name |
| `export_gene_results(...)` | `GeneExportDocument` / `GeneExportEntry` | hgnc_symbol, full_name, codes (hgnc_id, entrez, ensembl, uniprot, omim), associated_diseases, locus_type, chromosome |
| `export_drug_results(...)` | `DrugExportDocument` / `DrugExportEntry` | preferred_name, brand_name, compound_id, codes (rxcui, mesh, ndc, drugbank, unii), drug_class, mechanism, development_phase, is_investigational, sponsor |
| `export_pharma_results(...)` | `PharmaExportDocument` / `PharmaExportEntry` | canonical_name, full_name, headquarters, parent_company, subsidiaries |
| `export_author_results(...)` | `AuthorExportDocument` / `AuthorExportEntry` | full_name, role, affiliation, email, orcid |
| `export_citation_results(...)` | `CitationExportDocument` / `CitationExportEntry` | pmid, pmcid, doi, nct, url, citation_text, citation_number. Includes API validation via `CitationValidator` (DOI, NCT, PMID resolution) |

All functions filter to `ValidationStatus.VALIDATED` entities only. Each export includes `run_id`, `timestamp`, `document`, `pipeline_version`, and aggregate counts.

### J01b_metadata_exporters.py

Document-level metadata export functions.

**Functions:**

| Function | Output | Content |
|----------|--------|---------|
| `export_document_metadata(...)` | `metadata_{stem}_{ts}.json` | File info (size, extension), PDF metadata (title, author, page count, creation date), classification (type code/name/group, confidence), description (title, short/long description), primary date with source |
| `export_care_pathways(...)` | `care_pathways_{stem}_{ts}.json` | Treatment algorithm graphs: nodes (id, type, text, phase, drugs, dose, duration), edges (source_id, target_id, condition), pathway-level fields (entry criteria, primary/alternative drugs, decision points, relapse handling) |
| `export_recommendations(...)` | `recommendations_{stem}_{ts}.json` | Guideline recommendations: recommendation sets with individual recommendations (type, population, condition, severity, action, preferred/alternatives, dosing info, evidence level, strength, references) |

### J02_visual_export.py

Export handlers for the visual extraction pipeline, supporting both tables and figures.

**Functions:**

| Function | Purpose |
|----------|---------|
| `visual_to_dict(visual, include_image, image_file)` | Convert `ExtractedVisual` to serializable dict with locations, title, description, caption, table data, relationships, provenance, triage info |
| `pipeline_result_to_dict(result)` | Convert full `PipelineResult` to dict with metadata, statistics, visuals |
| `export_visuals_to_json(result, output_path, include_images, pretty_print)` | Export all visuals to single JSON file |
| `export_tables_only(result, output_path, output_dir, doc_name, save_images)` | Export tables with optional image files |
| `export_figures_only(result, output_path, output_dir, doc_name, save_images)` | Export figures with optional image files |
| `export_images_separately(result, output_dir, doc_name, format)` | Save all visual images as individual files, returns `Dict[visual_id, Path]` |

**Image file naming:** `{doc_name}_{type}_page{N}_{idx}.png` where type is `table` or `figure`.

## Output Structure

Processing `document.pdf` creates:

```
document/
  abbreviations_document_20260203_143022.json
  diseases_document_20260203_143022.json
  drugs_document_20260203_143022.json
  genes_document_20260203_143022.json
  pharma_document_20260203_143022.json
  authors_document_20260203_143022.json
  citations_document_20260203_143022.json
  feasibility_document_20260203_143022.json
  figures_document_20260203_143022.json
  tables_document_20260203_143022.json
  metadata_document_20260203_143022.json
  document_20260203_143022.txt
  document_flowchart_page3_1.png
  document_chart_page5_1.png
  document_table_data_page2_1.png
```

## Usage Patterns

```python
manager = ExportManager(run_id, pipeline_version, claude_client=claude_client)
out_dir = manager.get_output_dir(pdf_path)

manager.export_results(pdf_path, results, candidates, counters,
                       disease_results=diseases, drug_results=drugs)
manager.export_disease_results(pdf_path, diseases)
manager.export_images(pdf_path, doc)
manager.export_feasibility_results(pdf_path, feasibility, doc=doc)
```

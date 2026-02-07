# Layer J: Export

## Purpose

Serializes extraction results to timestamped JSON files, one per entity type per document. 4 modules.

---

## Modules

### J01_export_handlers.py

`ExportManager`: main export orchestrator.

```python
manager = ExportManager(run_id=run_id, pipeline_version="0.8",
                        output_dir=None, claude_client=claude_client)
```

Output directory: `{pdf_stem}/` alongside the PDF (override with `output_dir`).

| Method | Output file |
|--------|------------|
| `export_results(...)` | `abbreviations_{stem}_{ts}.json` |
| `export_disease_results(...)` | `diseases_{stem}_{ts}.json` |
| `export_gene_results(...)` | `genes_{stem}_{ts}.json` |
| `export_drug_results(...)` | `drugs_{stem}_{ts}.json` |
| `export_pharma_results(...)` | `pharma_{stem}_{ts}.json` |
| `export_author_results(...)` | `authors_{stem}_{ts}.json` |
| `export_citation_results(...)` | `citations_{stem}_{ts}.json` |
| `export_feasibility_results(...)` | `feasibility_{stem}_{ts}.json` |
| `export_images(...)` | `figures_{stem}_{ts}.json` + PNGs |
| `export_tables(...)` | `tables_{stem}_{ts}.json` + PNGs |
| `export_extracted_text(...)` | `{stem}_{ts}.txt` |
| `export_document_metadata(...)` | `metadata_{stem}_{ts}.json` |
| `export_care_pathways(...)` | `care_pathways_{stem}_{ts}.json` |
| `export_recommendations(...)` | `recommendations_{stem}_{ts}.json` |

### J01a_entity_exporters.py

Per-entity export functions using Pydantic `model_dump_json()`. Filters to `VALIDATED` entities only. Each includes `run_id`, `timestamp`, `document`, `pipeline_version`.

### J01b_metadata_exporters.py

Document metadata, care pathway, and recommendation exports.

### J02_visual_export.py

Visual pipeline export: `export_tables_only()`, `export_figures_only()`, `export_images_separately()`.

Image naming: `{doc_name}_{type}_page{N}_{idx}.png`.

---

## Output Structure

```
document/
  abbreviations_document_20260203_143022.json
  diseases_document_20260203_143022.json
  drugs_document_20260203_143022.json
  genes_document_20260203_143022.json
  feasibility_document_20260203_143022.json
  figures_document_20260203_143022.json
  tables_document_20260203_143022.json
  metadata_document_20260203_143022.json
  document_20260203_143022.txt
  document_flowchart_page3_1.png
```

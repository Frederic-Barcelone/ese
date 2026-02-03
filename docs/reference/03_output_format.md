# Output Format Reference

This document describes the JSON output files produced by the ESE pipeline. Each entity type is exported to a separate timestamped JSON file. Export logic is implemented in `J_export/J01_export_handlers.py`, `J_export/J01a_entity_exporters.py`, `J_export/J01b_metadata_exporters.py`, and `J_export/J02_visual_export.py`.

## Output Directory Structure

Processing `document.pdf` creates a directory named after the PDF stem containing one file per entity type:

```
document/
  abbreviations_document_YYYYMMDD_HHMMSS.json
  diseases_document_YYYYMMDD_HHMMSS.json
  drugs_document_YYYYMMDD_HHMMSS.json
  genes_document_YYYYMMDD_HHMMSS.json
  pharma_document_YYYYMMDD_HHMMSS.json
  authors_document_YYYYMMDD_HHMMSS.json
  citations_document_YYYYMMDD_HHMMSS.json
  feasibility_document_YYYYMMDD_HHMMSS.json
  care_pathways_document_YYYYMMDD_HHMMSS.json
  recommendations_document_YYYYMMDD_HHMMSS.json
  metadata_document_YYYYMMDD_HHMMSS.json
  visuals_document_YYYYMMDD_HHMMSS.json
  document_extracted_text_YYYYMMDD_HHMMSS.txt
  flowchart_page3_1.png          (extracted figures)
  table_page5_2.png              (rendered tables)
```

### File Naming Convention

- Pattern: `{entity_type}_{doc_stem}_{YYYYMMDD}_{HHMMSS}.json`
- `doc_stem`: PDF filename without the `.pdf` extension
- Timestamp: UTC time of the extraction run
- One file per entity type per document per run

## Common JSON Envelope

Every entity export file wraps results in a document-level envelope. The exact fields vary by entity type, but all include:

```json
{
  "run_id": "RUN_20250203_143045_abc123",
  "timestamp": "2025-02-03T14:30:45.123456",
  "document": "document.pdf",
  "document_path": "/absolute/path/to/document.pdf",
  "pipeline_version": "0.7"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | string | Unique identifier for the pipeline run |
| `timestamp` | string | ISO 8601 timestamp of the extraction |
| `document` | string | Source PDF filename |
| `document_path` | string | Absolute path to the source PDF |
| `pipeline_version` | string | Pipeline version from `config.yaml` (`system.pipeline_version`) |

---

## Entity Type: Abbreviations

**File:** `abbreviations_{doc_stem}_{timestamp}.json`
**Export function:** `ExportManager` in `J01_export_handlers.py`
**Model:** `ExtractedEntity` from `A_core/A01_domain_models.py`

```json
{
  "run_id": "RUN_20250203_143045_abc123",
  "timestamp": "2025-02-03T14:30:45.123456",
  "document": "document.pdf",
  "pipeline_version": "0.7",
  "statistics": {
    "total_candidates": 45,
    "validated": 32,
    "rejected": 10,
    "ambiguous": 3
  },
  "entities": [
    {
      "short_form": "TNF",
      "long_form": "tumor necrosis factor",
      "field_type": "DEFINITION_PAIR",
      "generator_type": "gen:syntax_pattern",
      "validation_status": "VALIDATED",
      "confidence": 0.95,
      "evidence_spans": [
        {
          "text": "tumor necrosis factor (TNF)",
          "location": {"page": 3, "section": "Introduction"},
          "scope_ref": "a1b2c3d4",
          "start_char_offset": 0,
          "end_char_offset": 27
        }
      ],
      "provenance": {
        "run_id": "RUN_20250203_143045_abc123",
        "doc_fingerprint": "sha256:abc123...",
        "generator_name": "gen:syntax_pattern",
        "strategy_version": "1.0",
        "lexicon_source": null,
        "prompt_bundle_hash": "def456...",
        "context_hash": "ghi789..."
      }
    }
  ]
}
```

**Key fields per entity:**

| Field | Type | Description |
|-------|------|-------------|
| `short_form` | string | The abbreviation (e.g., "TNF") |
| `long_form` | string or null | The expansion (null for SHORT_FORM_ONLY) |
| `field_type` | enum | `DEFINITION_PAIR`, `GLOSSARY_ENTRY`, or `SHORT_FORM_ONLY` |
| `generator_type` | enum | `gen:syntax_pattern`, `gen:lexicon_match`, `gen:glossary_table`, `gen:layout_heuristic` |
| `validation_status` | enum | `VALIDATED`, `REJECTED`, or `AMBIGUOUS` |
| `confidence` | float | 0.0 to 1.0 |
| `evidence_spans` | array | Context text with page/location metadata |
| `provenance` | object | Full audit trail |

---

## Entity Type: Diseases

**File:** `diseases_{doc_stem}_{timestamp}.json`
**Export function:** `export_disease_results()` in `J01a_entity_exporters.py`
**Model:** `DiseaseExportDocument` / `DiseaseExportEntry` from `A_core/A05_disease_models.py`

```json
{
  "run_id": "...",
  "timestamp": "...",
  "document": "document.pdf",
  "pipeline_version": "0.7",
  "total_candidates": 120,
  "total_validated": 85,
  "total_rejected": 30,
  "total_ambiguous": 5,
  "diseases": [
    {
      "matched_text": "IgA nephropathy",
      "preferred_label": "IgA nephropathy",
      "abbreviation": "IgAN",
      "confidence": 0.92,
      "is_rare_disease": true,
      "category": "nephrology",
      "codes": {
        "icd10": "N02.2",
        "icd11": null,
        "snomed": "236407003",
        "mondo": "MONDO:0005308",
        "orpha": "ORPHA:97555",
        "umls": null,
        "mesh": "D005922"
      },
      "all_identifiers": [
        {"system": "Orphanet", "code": "ORPHA:97555", "display": "IgA nephropathy"},
        {"system": "ICD-10", "code": "N02.2", "display": null},
        {"system": "SNOMED-CT", "code": "236407003", "display": null},
        {"system": "MeSH", "code": "D005922", "display": "IgA nephropathy"}
      ],
      "context": "...diagnosed with IgA nephropathy (IgAN)...",
      "page": 3,
      "mention_count": 12,
      "pages_mentioned": [1, 3, 5, 7, 8, 10, 12, 14, 15, 18, 20, 22],
      "lexicon_source": "2025_08_orphanet_diseases.json",
      "validation_flags": ["pubtator_enriched"],
      "mesh_aliases": ["Berger disease", "glomerulonephritis, IGA"],
      "pubtator_normalized_name": "Immunoglobulin A Nephropathy",
      "enrichment_source": "pubtator3"
    }
  ]
}
```

**Disease-specific fields:**

| Field | Type | Description |
|-------|------|-------------|
| `matched_text` | string | Exact text matched in document |
| `preferred_label` | string | Canonical disease name from lexicon |
| `abbreviation` | string or null | Disease abbreviation (e.g., "IgAN") |
| `is_rare_disease` | boolean | Flagged from Orphanet data |
| `category` | string or null | Clinical category (e.g., "nephrology") |
| `codes` | object | Flat map of coding systems to primary codes |
| `all_identifiers` | array | Full list of identifiers with system, code, display |
| `mention_count` | integer | Number of mentions in document |
| `pages_mentioned` | array of int | Page numbers where disease appears |
| `mesh_aliases` | array | Aliases from PubTator3 enrichment |
| `pubtator_normalized_name` | string or null | PubTator3 normalized name |
| `enrichment_source` | string or null | Source of enrichment data |

---

## Entity Type: Drugs

**File:** `drugs_{doc_stem}_{timestamp}.json`
**Export function:** `export_drug_results()` in `J01a_entity_exporters.py`
**Model:** `DrugExportDocument` / `DrugExportEntry` from `A_core/A06_drug_models.py`

```json
{
  "run_id": "...",
  "timestamp": "...",
  "document": "document.pdf",
  "pipeline_version": "0.7",
  "total_candidates": 50,
  "total_validated": 30,
  "total_rejected": 18,
  "total_investigational": 5,
  "drugs": [
    {
      "matched_text": "iptacopan",
      "preferred_name": "iptacopan",
      "brand_name": "FABHALTA",
      "compound_id": "LNP023",
      "confidence": 0.88,
      "is_investigational": false,
      "drug_class": "complement factor B inhibitor",
      "mechanism": "Selective inhibitor of complement factor B",
      "development_phase": "Approved",
      "sponsor": "Novartis",
      "conditions": ["C3 glomerulopathy", "paroxysmal nocturnal hemoglobinuria"],
      "nct_id": "NCT04817618",
      "dosage_form": "CAPSULE",
      "route": "ORAL",
      "marketing_status": "Prescription",
      "codes": {
        "rxcui": "2670341",
        "mesh": "D000123",
        "ndc": null,
        "drugbank": "DB16649",
        "unii": "ABC123DEF"
      },
      "all_identifiers": [
        {"system": "RxCUI", "code": "2670341", "display": "iptacopan"},
        {"system": "ChEMBL", "code": "CHEMBL4594299", "display": "iptacopan"}
      ],
      "context": "...treatment with iptacopan 200mg twice daily...",
      "page": 5,
      "mention_count": 28,
      "pages_mentioned": [1, 2, 3, 5, 7, 8, 10],
      "lexicon_source": "2025_08_fda_approved_drugs.json",
      "validation_flags": ["pubtator_enriched"],
      "mesh_aliases": [],
      "pubtator_normalized_name": "iptacopan",
      "enrichment_source": "pubtator3"
    }
  ]
}
```

**Drug-specific fields:**

| Field | Type | Description |
|-------|------|-------------|
| `preferred_name` | string | Generic/canonical drug name |
| `brand_name` | string or null | Brand name (e.g., "FABHALTA") |
| `compound_id` | string or null | Development code (e.g., "LNP023") |
| `is_investigational` | boolean | Whether the drug is investigational |
| `drug_class` | string or null | Pharmacological class |
| `mechanism` | string or null | Mechanism of action |
| `development_phase` | string or null | Phase (Preclinical, Phase 1/2/3, Approved, Withdrawn) |
| `sponsor` | string or null | Sponsoring pharmaceutical company |
| `conditions` | array | Target indications |
| `nct_id` | string or null | Associated ClinicalTrials.gov NCT ID |
| `dosage_form` | string or null | FDA dosage form |
| `route` | string or null | Administration route |
| `marketing_status` | string or null | FDA marketing status |
| `codes` | object | Flat map: `rxcui`, `mesh`, `ndc`, `drugbank`, `unii` |

---

## Entity Type: Genes

**File:** `genes_{doc_stem}_{timestamp}.json`
**Export function:** `export_gene_results()` in `J01a_entity_exporters.py`
**Model:** `GeneExportDocument` / `GeneExportEntry` from `A_core/A19_gene_models.py`

```json
{
  "run_id": "...",
  "timestamp": "...",
  "document": "document.pdf",
  "pipeline_version": "0.7",
  "total_candidates": 25,
  "total_validated": 18,
  "total_rejected": 7,
  "genes": [
    {
      "matched_text": "CFB",
      "hgnc_symbol": "CFB",
      "full_name": "complement factor B",
      "confidence": 0.90,
      "is_alias": false,
      "locus_type": "gene with protein product",
      "chromosome": "6p21.33",
      "codes": {
        "hgnc_id": "HGNC:1037",
        "entrez": "629",
        "ensembl": "ENSG00000243649",
        "uniprot": "P00751",
        "omim": "138470"
      },
      "all_identifiers": [
        {"system": "HGNC", "code": "HGNC:1037", "display": "CFB"},
        {"system": "ENTREZ", "code": "629", "display": "complement factor B"},
        {"system": "ENSEMBL", "code": "ENSG00000243649", "display": null}
      ],
      "associated_diseases": [
        {
          "orphacode": "329918",
          "disease_name": "C3 glomerulopathy",
          "association_type": "Disease-causing germline mutation(s) in"
        }
      ],
      "context": "...mutations in CFB have been associated with...",
      "page": 4,
      "mention_count": 8,
      "pages_mentioned": [2, 4, 6, 10, 12],
      "lexicon_source": "gene_synonyms.json",
      "validation_flags": []
    }
  ]
}
```

**Gene-specific fields:**

| Field | Type | Description |
|-------|------|-------------|
| `hgnc_symbol` | string | Official HGNC gene symbol |
| `full_name` | string or null | Full gene name |
| `is_alias` | boolean | Whether match was via alias/synonym |
| `locus_type` | string or null | Gene locus type (e.g., "gene with protein product") |
| `chromosome` | string or null | Chromosomal location |
| `codes` | object | Flat map: `hgnc_id`, `entrez`, `ensembl`, `uniprot`, `omim` |
| `associated_diseases` | array | Orphadata gene-disease linkages |

Each disease linkage contains:

| Field | Type | Description |
|-------|------|-------------|
| `orphacode` | string | Orphanet code for the disease |
| `disease_name` | string | Disease name |
| `association_type` | string | Type (e.g., "Disease-causing germline mutation(s) in") |

---

## Entity Type: Authors

**File:** `authors_{doc_stem}_{timestamp}.json`
**Export function:** `export_author_results()` in `J01a_entity_exporters.py`
**Model:** `AuthorExportDocument` / `AuthorExportEntry` from `A_core/A10_author_models.py`

```json
{
  "run_id": "...",
  "timestamp": "...",
  "document": "document.pdf",
  "pipeline_version": "0.7",
  "total_detected": 12,
  "unique_authors": 10,
  "authors": [
    {
      "full_name": "Jane Smith, MD, PhD",
      "role": "principal_investigator",
      "affiliation": "Harvard Medical School, Boston, MA",
      "email": "jsmith@hms.harvard.edu",
      "orcid": "0000-0001-2345-6789",
      "confidence": 0.92,
      "context": "Principal Investigator: Jane Smith, MD, PhD",
      "page": 1
    }
  ]
}
```

**Author roles:** `author`, `principal_investigator`, `co_investigator`, `corresponding_author`, `steering_committee`, `study_chair`, `data_safety_board`, `unknown`

---

## Entity Type: Citations

**File:** `citations_{doc_stem}_{timestamp}.json`
**Export function:** `export_citation_results()` in `J01a_entity_exporters.py`
**Model:** `CitationExportDocument` / `CitationExportEntry` from `A_core/A11_citation_models.py`

```json
{
  "run_id": "...",
  "timestamp": "...",
  "document": "document.pdf",
  "pipeline_version": "0.7",
  "total_detected": 45,
  "unique_identifiers": 38,
  "validation_summary": {
    "total_validated": 38,
    "valid_count": 35,
    "invalid_count": 2,
    "error_count": 1
  },
  "citations": [
    {
      "pmid": "12345678",
      "pmcid": "PMC9876543",
      "doi": "10.1056/NEJMoa2024816",
      "nct": null,
      "url": null,
      "citation_text": "Smith J et al. N Engl J Med. 2024;390:1-10.",
      "citation_number": 1,
      "page": 25,
      "confidence": 0.95,
      "validation": {
        "is_valid": true,
        "resolved_url": "https://doi.org/10.1056/NEJMoa2024816",
        "title": "Iptacopan in C3 Glomerulopathy",
        "status": null,
        "error": null
      }
    }
  ]
}
```

Citations include API validation results (DOI resolution, PMID lookup, NCT verification) via `E_normalization/E14_citation_validator.py`.

---

## Entity Type: Pharma Companies

**File:** `pharma_{doc_stem}_{timestamp}.json`
**Export function:** `export_pharma_results()` in `J01a_entity_exporters.py`
**Model:** `PharmaExportDocument` / `PharmaExportEntry` from `A_core/A09_pharma_models.py`

```json
{
  "run_id": "...",
  "timestamp": "...",
  "document": "document.pdf",
  "pipeline_version": "0.7",
  "total_detected": 5,
  "unique_companies": 3,
  "companies": [
    {
      "matched_text": "Novartis",
      "canonical_name": "Novartis AG",
      "full_name": "Novartis International AG",
      "headquarters": "Basel, Switzerland",
      "parent_company": null,
      "subsidiaries": ["Alexion Pharmaceuticals"],
      "confidence": 0.95,
      "context": "Novartis sponsored the APPEAR-C3G trial...",
      "page": 1,
      "lexicon_source": "pharma_companies.json"
    }
  ]
}
```

---

## Entity Type: Feasibility

**File:** `feasibility_{doc_stem}_{timestamp}.json`
**Export function:** `ExportManager.export_feasibility_results()` in `J01_export_handlers.py`
**Model:** `FeasibilityCandidate` from `A_core/A07_feasibility_models.py`

Contains structured clinical trial feasibility data including:

- Trial identifiers (NCT, EudraCT, CTIS)
- Eligibility criteria (inclusion/exclusion) with parsed lab values
- CONSORT screening flow metrics
- Study design (phase, blinding, randomization, arms)
- Epidemiology data (prevalence, incidence, demographics)
- Patient journey phases
- Operational burden (visits, procedures)
- Geographic footprint

---

## Entity Type: Care Pathways

**File:** `care_pathways_{doc_stem}_{timestamp}.json`
**Export function:** `export_care_pathways()` in `J01b_metadata_exporters.py`
**Model:** `CarePathway` from `A_core/A17_care_pathway_models.py`

```json
{
  "run_id": "...",
  "timestamp": "...",
  "document": "document.pdf",
  "pipeline_version": "0.7",
  "total_pathways": 1,
  "pathways": [
    {
      "pathway_id": "...",
      "title": "Induction Therapy for ANCA-Associated Vasculitis",
      "condition": "ANCA-associated vasculitis",
      "phases": ["induction", "maintenance"],
      "nodes": [
        {
          "id": "n1",
          "type": "treatment",
          "text": "Rituximab 375mg/m2 weekly x4",
          "phase": "induction",
          "drugs": ["rituximab"],
          "dose": "375mg/m2",
          "duration": "4 weeks"
        }
      ],
      "edges": [
        {
          "source_id": "n1",
          "target_id": "n2",
          "condition": "Remission achieved",
          "condition_type": "response"
        }
      ],
      "primary_drugs": ["rituximab"],
      "alternative_drugs": ["cyclophosphamide"],
      "source_figure": "Figure 1",
      "source_page": 3,
      "extraction_confidence": 0.85
    }
  ]
}
```

---

## Entity Type: Recommendations

**File:** `recommendations_{doc_stem}_{timestamp}.json`
**Export function:** `export_recommendations()` in `J01b_metadata_exporters.py`
**Model:** `RecommendationSet` from `A_core/A18_recommendation_models.py`

Contains guideline recommendations with dosing information, evidence levels, strength grades, population targeting, and references.

---

## Entity Type: Document Metadata

**File:** `metadata_{doc_stem}_{timestamp}.json`
**Export function:** `export_document_metadata()` in `J01b_metadata_exporters.py`
**Model:** `DocumentMetadataExport` from `A_core/A08_document_metadata_models.py`

```json
{
  "doc_id": "document",
  "doc_filename": "document.pdf",
  "file_size_bytes": 2456789,
  "file_size_human": "2.3 MB",
  "file_extension": ".pdf",
  "pdf_title": "A Phase 3 Study of Iptacopan",
  "pdf_author": "Smith J",
  "pdf_page_count": 28,
  "pdf_creation_date": "2024-06-15T00:00:00",
  "document_type_code": "CLIN_ARTICLE",
  "document_type_name": "Clinical Trial Article",
  "document_type_group": "clinical",
  "classification_confidence": 0.92,
  "title": "Iptacopan in C3 Glomerulopathy",
  "short_description": "Phase 3 trial results...",
  "long_description": "This article presents...",
  "document_date": "2024-06-15",
  "document_date_source": "pdf_metadata"
}
```

---

## Entity Type: Visuals (Tables and Figures)

**File:** `visuals_{doc_stem}_{timestamp}.json`
**Export function:** `J_export/J02_visual_export.py`
**Model:** `ExtractedVisual` from `A_core/A13_visual_models.py`

Contains extracted tables (with structured data) and figures (with VLM analysis), including:

- Visual type (TABLE, FIGURE, OTHER)
- Page range and bounding box coordinates
- Caption text and provenance
- Table structure (headers, rows, cells) for tables
- VLM classification and description for figures
- Base64-encoded image data (optional)

Extracted figure images are also saved as separate PNG files (e.g., `flowchart_page3_1.png`).

---

## JSON Formatting

All JSON output uses:

- 2-space indentation (`json_indent: 2` in `config.yaml`)
- UTF-8 encoding with `ensure_ascii=False` for proper Unicode rendering
- Pydantic `model_dump_json()` for entity exports (ensures schema compliance)
- Standard `json.dump()` for non-Pydantic exports (care pathways, recommendations)

## Output Configuration

Relevant settings in [`G_config/config.yaml`](../layers/G_config.md):

```yaml
output:
  format: "json"
  json_indent: 2
  include:
    statistics: true
    confidence: true
    context: true
    identifiers: true

extraction_pipeline:
  output:
    export_json: true
    export_combined: false
```

Set `export_combined: true` to additionally produce a single combined JSON file containing all entity types.

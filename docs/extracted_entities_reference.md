# Extracted Entities Reference

> **Pipeline version**: v0.8

---

## 1. Overview

Every entity follows: `Candidate -> Extracted Entity -> Export Entry (JSON)`.

**Shared fields**: UUID `id`, `doc_id`, schema version `"1.0.0"`, evidence spans, validation status + confidence score, mention count, provenance metadata.

---

## 2. Abbreviations

**Models**: `A_core/A01_domain_models.py`

### Candidate

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Auto-generated |
| `doc_id` | str | Source document |
| `field_type` | FieldType | `DEFINITION_PAIR`, `GLOSSARY_ENTRY`, or `SHORT_FORM_ONLY` |
| `generator_type` | GeneratorType | `SYNTAX` (C01), `LEXICON_MATCH` (C04), `GLOSSARY` (C05), `REGEX`, `NER`, `LLM` |
| `short_form` | str | Abbreviation (e.g., "PAH") |
| `long_form` | Optional[str] | Definition (e.g., "pulmonary arterial hypertension") |
| `context_text` | str | Surrounding text sent to verifier |
| `context_location` | Coordinate | Page number and bounding box |
| `initial_confidence` | float | Pre-validation confidence (0.0-1.0) |
| `provenance` | ProvenanceMetadata | Full audit trail |

### ExtractedEntity (Post-Validation)

Adds to Candidate:

| Field | Type | Description |
|-------|------|-------------|
| `candidate_id` | UUID | Link to source candidate |
| `schema_version` | str | "1.0.0" |
| `normalized_value` | Optional[str/dict] | Canonical form |
| `standard_id` | Optional[str] | External identifier |
| `primary_evidence` | EvidenceSpan | Best evidence |
| `supporting_evidence` | List[EvidenceSpan] | Additional evidence |
| `mention_count` | int | Occurrences in document |
| `pages_mentioned` | List[int] | Pages where found |
| `status` | ValidationStatus | APPROVED / REJECTED / PENDING |
| `confidence_score` | float | Final confidence (0.0-1.0) |
| `rejection_reason` | Optional[str] | Why rejected |
| `validation_flags` | List[str] | Validation notes |
| `category` | Optional[AbbreviationCategory] | `ABBREV`, `STATISTICAL`, `DISEASE`, `DRUG`, `GENE`, `STUDY`, `ORGANIZATION` |
| `raw_llm_response` | Optional[dict/str] | Raw LLM response |

---

## 3. Diseases

**Models**: `A_core/A05_disease_models.py`

### DiseaseCandidate

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Auto-generated |
| `doc_id` | str | Source document |
| `matched_text` | str | Exact text matched |
| `preferred_label` | str | Canonical name from lexicon |
| `abbreviation` | Optional[str] | Disease abbreviation |
| `synonyms` | List[str] | Known synonyms |
| `field_type` | DiseaseFieldType | `EXACT_MATCH`, `PATTERN_MATCH`, `NER_DETECTION`, `ABBREVIATION_EXPAND` |
| `generator_type` | DiseaseGeneratorType | `gen:disease_lexicon_specialized`, `_general` (29K), `_orphanet` (9.5K), `_scispacy_ner` |
| `identifiers` | List[DiseaseIdentifier] | Medical codes (system: ICD-10/11, SNOMED-CT, MONDO, ORPHA, MeSH, UMLS) |
| `context_text` | str | Surrounding text |
| `context_location` | Coordinate | Page and bbox |
| `is_rare_disease` | bool | Orphanet rare disease flag |
| `prevalence` | Optional[str] | e.g., "<1/1000000" |
| `parent_disease` | Optional[str] | Parent category |
| `disease_category` | Optional[str] | Clinical specialty |
| `initial_confidence` | float | Pre-validation confidence |
| `confidence_boost` | float | Context keyword boost (0.0-0.5) |
| `provenance` | DiseaseProvenanceMetadata | Audit trail with medical codes |

### ExtractedDisease (Post-Validation)

All DiseaseCandidate fields plus:

| Field | Type | Description |
|-------|------|-------------|
| `candidate_id` | UUID | Link to source candidate |
| `schema_version` | str | "1.0.0" |
| `icd10_code`, `icd11_code`, `snomed_code`, `mondo_id`, `orpha_code`, `umls_cui`, `mesh_id` | Optional[str] | Ontology codes |
| `primary_evidence` / `supporting_evidence` | EvidenceSpan / List | Evidence spans |
| `mention_count` / `pages_mentioned` | int / List[int] | Occurrence tracking |
| `status` / `confidence_score` | ValidationStatus / float | Validation result |
| `mesh_aliases` | List[str] | PubTator MeSH aliases |
| `pubtator_normalized_name` | Optional[str] | PubTator canonical name |
| `enrichment_source` | Optional[str] | Which API enriched this |

---

## 4. Drugs

**Models**: `A_core/A06_drug_models.py`

### DrugCandidate

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Auto-generated |
| `doc_id` | str | Source document |
| `matched_text` | str | Exact text matched |
| `preferred_name` | str | Generic/canonical name |
| `brand_name` | Optional[str] | Brand name (e.g., FABHALTA) |
| `compound_id` | Optional[str] | Development code (e.g., LNP023) |
| `field_type` | DrugFieldType | Detection method |
| `generator_type` | DrugGeneratorType | `gen:drug_lexicon_alexion`, `_investigational`, `_fda` (23K), `_rxnorm` (132K), `_pattern_compound`, `_scispacy_ner` |
| `identifiers` | List[DrugIdentifier] | Drug codes (system: RxCUI, NDC, MeSH, DrugBank, ChEMBL, NCT, UNII) |
| `context_text` / `context_location` | str / Coordinate | Context |
| `drug_class` | Optional[str] | e.g., "Factor B inhibitor" |
| `mechanism` | Optional[str] | Mechanism of action |
| `development_phase` | Optional[str] | Preclinical, Phase 1/2/3, Approved, Withdrawn, Unknown |
| `is_investigational` | bool | Whether investigational |
| `sponsor` | Optional[str] | Pharmaceutical company |
| `conditions` | List[str] | Target indications |
| `nct_id` | Optional[str] | ClinicalTrials.gov NCT ID |
| `dosage_form` / `route` / `marketing_status` | Optional[str] | Formulation details |
| `initial_confidence` | float | Pre-validation confidence |
| `provenance` | DrugProvenanceMetadata | Audit trail |

### ExtractedDrug (Post-Validation)

All DrugCandidate fields plus:

| Field | Type | Description |
|-------|------|-------------|
| `candidate_id` | UUID | Link to source candidate |
| `schema_version` | str | "1.0.0" |
| `rxcui`, `mesh_id`, `ndc_code`, `drugbank_id`, `unii` | Optional[str] | Standard codes |
| `primary_evidence` / `supporting_evidence` | EvidenceSpan / List | Evidence |
| `mention_count` / `pages_mentioned` | int / List[int] | Occurrence tracking |
| `status` / `confidence_score` | ValidationStatus / float | Validation result |
| `mesh_aliases` | List[str] | PubTator MeSH aliases |
| `pubtator_normalized_name` | Optional[str] | PubTator canonical name |

---

## 5. Genes

**Models**: `A_core/A19_gene_models.py`

### GeneCandidate

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Auto-generated |
| `doc_id` | str | Source document |
| `matched_text` | str | Exact text matched |
| `hgnc_symbol` | str | Official HGNC symbol (canonical) |
| `full_name` | Optional[str] | Full gene name |
| `is_alias` | bool | Whether matched via alias |
| `alias_of` | Optional[str] | Canonical symbol if alias |
| `field_type` | GeneFieldType | Detection method |
| `generator_type` | GeneGeneratorType | Source generator |
| `identifiers` | List[GeneIdentifier] | Codes (system: HGNC, ENTREZ, ENSEMBL, OMIM, UNIPROT, ORPHACODE) |
| `context_text` / `context_location` | str / Coordinate | Context |
| `locus_type` | Optional[str] | "protein-coding", "ncRNA", "pseudogene" |
| `chromosome` | Optional[str] | Chromosomal location |
| `associated_diseases` | List[GeneDiseaseLinkage] | Orphadata associations (orphacode, disease_name, association_type/status) |
| `initial_confidence` | float | Pre-validation confidence |
| `provenance` | GeneProvenanceMetadata | Audit trail |

### ExtractedGene (Post-Validation)

All GeneCandidate fields plus:

| Field | Type | Description |
|-------|------|-------------|
| `candidate_id` | UUID | Link to source candidate |
| `schema_version` | str | "1.0.0" |
| `hgnc_id`, `entrez_id`, `ensembl_id`, `omim_id`, `uniprot_id` | Optional[str] | Standard identifiers |
| `primary_evidence` / `supporting_evidence` | EvidenceSpan / List | Evidence |
| `mention_count` / `pages_mentioned` | int / List[int] | Occurrence tracking |
| `status` / `confidence_score` | ValidationStatus / float | Validation result |
| `pubtator_normalized_name` | Optional[str] | PubTator canonical name |
| `pubtator_aliases` | List[str] | PubTator known aliases |
| `enrichment_source` | Optional[EnrichmentSource] | Which API enriched this |

---

## 6. Authors

**Models**: `A_core/A10_author_models.py`

### AuthorCandidate

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Auto-generated |
| `doc_id` | str | Source document |
| `full_name` | str | Full name |
| `first_name` / `last_name` | Optional[str] | Name parts |
| `role` | AuthorRoleType | `author`, `principal_investigator`, `co_investigator`, `corresponding_author`, `steering_committee`, `study_chair`, `data_safety_board`, `unknown` |
| `affiliation` | Optional[str] | Institution |
| `orcid` | Optional[str] | ORCID (0000-0000-0000-000X) |
| `email` | Optional[str] | Contact email |
| `generator_type` | AuthorGeneratorType | `gen:author_header`, `_affiliation`, `_contribution`, `_investigator`, `_regex` |
| `context_text` / `context_location` | str / Coordinate | Context |
| `provenance` | AuthorProvenanceMetadata | Audit trail |

---

## 7. Citations

**Models**: `A_core/A11_citation_models.py`

### CitationCandidate

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Auto-generated |
| `doc_id` | str | Source document |
| `pmid`, `pmcid`, `doi`, `nct_id`, `url` | Optional[str] | Identifiers |
| `citation_text` | Optional[str] | Full citation text |
| `citation_number` | Optional[int] | Reference number |
| `identifier_types` | List[CitationIdentifierType] | `pmid`, `pmcid`, `doi`, `nct`, `url` |
| `generator_type` | CitationGeneratorType | `gen:citation_regex`, `_reference`, `_inline` |
| `context_text` / `context_location` | str / Coordinate | Context |
| `provenance` | CitationProvenanceMetadata | Audit trail |

### CitationValidation

| Field | Type | Description |
|-------|------|-------------|
| `identifier_type` / `identifier_value` | str | What was validated |
| `is_valid` | bool | Whether identifier resolves |
| `resolved_url` | Optional[str] | Resolved URL |
| `title` | Optional[str] | Article/study title |
| `error_message` | Optional[str] | Why validation failed |

---

## 8. Pharma Companies

**Models**: `A_core/A09_pharma_models.py`

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Auto-generated |
| `doc_id` | str | Source document |
| `matched_text` | str | Exact text matched |
| `canonical_name` | str | Official company name |
| `headquarters` | Optional[str] | Location |
| `parent_company` | Optional[str] | Parent/holding company |
| `context_text` / `context_location` | str / Coordinate | Context |
| `provenance` | PharmaProvenanceMetadata | Audit trail |

---

## 9. Document Metadata

**Models**: `A_core/A08_document_metadata_models.py`

### FileMetadata

`filename`, `file_size_bytes`, `file_hash` (SHA-256), `created_at`, `modified_at`

### PDFMetadata

`title`, `author`, `creator`, `producer`, `creation_date`, `modification_date`, `page_count`, `pdf_version`

### DocumentClassification

`document_type` (LLM-classified, e.g., "clinical_trial_protocol"), `confidence`, `sub_type`, `therapeutic_area`

### DateExtractionResult

`primary_date`, `date_source`, `all_dates` (list of dates with sources), `fallback_chain` (priority order tried)

---

## 10. Feasibility

**Models**: `A_core/A07_feasibility_models.py`, `A_core/A21_clinical_criteria.py`, `A_core/A22_logical_expressions.py`

Most complex entity type with nested sub-structures. 19 field types covering eligibility, epidemiology, study design, screening, operational burden.

### Key Sub-Models

- **EligibilityCriterion**: criterion_type (inclusion/exclusion), text, category, parsed_value, lab_criterion (test_name, min/max, unit), severity_grade, logical_expression (AND/OR grouping)
- **ScreeningFlow**: screened, screen_failures, randomized, screen_failure_rate, failure_reasons
- **OperationalBurden**: total_visits, visit_frequency, procedures_per_visit, blood_draw_volume_ml, visit_schedule
- **StudyDesign**: phase, blinding, randomization_ratio, treatment_arms, primary/secondary_endpoints, study_duration
- **EpidemiologyData**: prevalence, incidence, geographic_distribution, demographics
- **PatientJourneyPhaseType**: screening, run_in, randomization, treatment, follow_up, extension

---

## 11. Recommendations

**Models**: `A_core/A18_recommendation_models.py`

### GuidelineRecommendation

| Field | Type | Description |
|-------|------|-------------|
| `recommendation_id` | str | Unique identifier |
| `recommendation_type` | RecommendationType | `treatment`, `dosing`, `duration`, `monitoring`, `contraindication`, `alternative`, `preference`, `other` |
| `population` | str | Target patient population |
| `condition` / `severity` | Optional[str] | Clinical context |
| `action` | str | Recommended action |
| `action_description` | Optional[str] | Detailed description |
| `preferred` | Optional[str] | Preferred option |
| `alternatives` | List[str] | Alternative treatments |
| `dosing` | List[DrugDosingInfo] | drug_name, dose_range, starting/maintenance/max_dose, route, frequency |
| `taper_target` / `duration` / `stop_window` | Optional[str] | Timing details |
| `evidence_level` | EvidenceLevel | `high`, `moderate`, `low`, `very_low`, `expert_opinion`, `unknown` |
| `strength` | RecommendationStrength | `strong`, `conditional`, `weak`, `unknown` |
| `references` | List[str] | PMID, DOI references |
| `source` / `source_text` / `page_num` | str / Optional | Source location |

### RecommendationSet

`guideline_name`, `guideline_year`, `organization`, `target_condition`, `target_population`, `recommendations` (list), `source_document`, `extraction_confidence`

---

## 12. Care Pathways

**Models**: `A_core/A17_care_pathway_models.py`

### CarePathway

| Field | Type | Description |
|-------|------|-------------|
| `pathway_id` | str | Unique identifier |
| `title` | str | Pathway title |
| `condition` | str | Target condition |
| `guideline_source` | Optional[str] | Source guideline |
| `phases` | List[str] | Treatment phases (e.g., ["Induction", "Maintenance"]) |
| `nodes` | List[CarePathwayNode] | Pathway steps |
| `edges` | List[CarePathwayEdge] | Transitions between steps |
| `entry_criteria` | Optional[str] | Who this applies to |
| `primary_drugs` / `alternative_drugs` | List[str] | Treatment drugs |
| `decision_points` | List[str] | Key decision summaries |
| `target_dose` / `target_timepoint` / `maintenance_duration` / `relapse_handling` | Optional[str] | Treatment details |
| `source_figure` / `source_page` | Optional | Source location |
| `extraction_confidence` | float | Confidence score |

### CarePathwayNode

`id`, `type` (NodeType: `start`, `end`, `action`, `decision`, `assessment`, `note`, `condition`), `text`, `phase`, `drugs`, `dose`, `duration`, `frequency`, `position`, `source_bbox`

### CarePathwayEdge

`source_id`, `target_id`, `condition`, `condition_type`, `label`

### TaperSchedule

`schedule_id`, `drug_name`, `regimen_name`, `schedule` (List[TaperSchedulePoint]: week/day/month + dose_mg + dose_unit), `starting_dose`, `target_dose`, `target_timepoint`, `source_figure`, `source_page`, `legend_color`, `legend_marker`, `linked_recommendation_text`

---

## 13. Visual Elements (Figures & Tables)

**Models**: `A_core/A13_visual_models.py`

### VisualCandidate

| Field | Type | Description |
|-------|------|-------------|
| `visual_type` | VisualType | `TABLE` or `FIGURE` |
| `page_range` | List[int] | Pages where visual appears |
| `bbox_pts_per_page` | List[PageBBox] | Bounding boxes per page |
| `caption_candidate` | Optional[CaptionCandidate] | Associated caption |
| `reference_number` | Optional[VisualReference] | "Table 1", "Figure 2" |
| `docling_table` / `validated_table` | Optional[TableStructure] | Table data (headers, rows, merged_cells, source) |
| `vlm_enrichment` | Optional[VLMEnrichmentResult] | VLM analysis |
| `image_path` | Optional[str] | Path to extracted image |
| `image_bytes` | Optional[bytes] | Raw image bytes |

### ExtractedVisual (Post-Validation)

`id`, `visual_type`, `page_range`, `bbox_pts_per_page`, `caption_text`, `caption_provenance` (`pdf_text`, `ocr`, `vlm`), `reference`, `docling_table`, `validated_table`, `image_path`, `vlm_title`, `vlm_description`, `layout_code`, `position_code`, `layout_filename`

### VLMEnrichmentResult

`classification` (type + confidence), `parsed_reference`, `extracted_caption`, `table_validation`, `is_continuation`, `continuation_of_reference`

---

## 14. Provenance Model

**Models**: `A_core/A01_domain_models.py`, `A_core/A03_provenance.py`

Core fields: `pipeline_version`, `run_id`, `doc_fingerprint` (SHA-256), `generator_name`, `rule_version`, `lexicon_source`, `lexicon_ids`, `prompt_bundle_hash`, `context_hash`, `llm_config`, `timestamp`.

Each entity type extends `BaseProvenanceMetadata` with specialized identifier lists (DiseaseIdentifier, DrugIdentifier, GeneIdentifier).

---

## 15. Output Structure

### Per-Document Folder

Processing `document.pdf` creates `document/` with timestamped JSON files per entity type (abbreviations, diseases, drugs, genes, recommendations, feasibility, figures, tables, metadata, authors, citations), extracted text, and PNG images.

### JSON Envelope

```json
{
  "document_id": "document.pdf",
  "extraction_timestamp": "2026-02-06T12:00:00Z",
  "pipeline_version": "v0.8",
  "schema_version": "1.0.0",
  "entity_count": 42,
  "entities": [...]
}
```

### Export Simplification

`ExportEntry` models flatten `Extracted*` models: drop UUIDs/provenance internals, flatten identifier lists to code strings, remove raw LLM responses.

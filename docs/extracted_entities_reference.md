# Extracted Entities Reference -- Complete Field Guide

> **Date**: February 2026
> **Pipeline version**: v0.8

Comprehensive reference of every entity type extracted by the ESE pipeline, with all fields, data sources, enums, and output formats.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Abbreviations](#2-abbreviations)
3. [Diseases](#3-diseases)
4. [Drugs](#4-drugs)
5. [Genes](#5-genes)
6. [Authors](#6-authors)
7. [Citations](#7-citations)
8. [Pharma Companies](#8-pharma-companies)
9. [Document Metadata](#9-document-metadata)
10. [Feasibility](#10-feasibility)
11. [Recommendations](#11-recommendations)
12. [Care Pathways](#12-care-pathways)
13. [Visual Elements (Figures & Tables)](#13-visual-elements-figures--tables)
14. [Provenance Model](#14-provenance-model)
15. [Output Structure](#15-output-structure)

---

## 1. Overview

The ESE pipeline extracts 14+ entity types from PDF documents. Every entity follows a common lifecycle:

```
Candidate (pre-validation)  -->  Extracted Entity (post-validation)  -->  Export Entry (JSON output)
```

**Shared patterns across all entity types:**
- UUID-based identification (`id`, `candidate_id`)
- Document binding (`doc_id`)
- Schema versioning (`schema_version: "1.0.0"`)
- Evidence tracking (`primary_evidence`, `supporting_evidence`)
- Validation status (`status`, `confidence_score`, `rejection_reason`)
- Mention frequency (`mention_count`, `pages_mentioned`)
- Provenance metadata (generator, lexicon source, timestamps)
- Audit trail (`raw_llm_response`)

---

## 2. Abbreviations

**Models**: `A_core/A01_domain_models.py`

### Candidate (Pre-Validation)

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Auto-generated unique identifier |
| `doc_id` | str | Source document identifier |
| `field_type` | FieldType | Detection method (see enum below) |
| `generator_type` | GeneratorType | Which generator produced this candidate |
| `short_form` | str | Abbreviation (e.g., "PAH") |
| `long_form` | Optional[str] | Definition (e.g., "pulmonary arterial hypertension") |
| `context_text` | str | Surrounding text sent to verifier |
| `context_location` | Coordinate | Page number and bounding box |
| `initial_confidence` | float | Pre-validation confidence (0.0-1.0) |
| `provenance` | ProvenanceMetadata | Full audit trail |

### ExtractedEntity (Post-Validation)

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Auto-generated unique identifier |
| `candidate_id` | UUID | Link to source candidate |
| `doc_id` | str | Source document identifier |
| `schema_version` | str | Output schema version ("1.0.0") |
| `field_type` | FieldType | Detection method |
| `short_form` | str | Abbreviation |
| `long_form` | Optional[str] | Definition |
| `normalized_value` | Optional[str/dict] | Post-normalization canonical form |
| `standard_id` | Optional[str] | External standard identifier |
| `primary_evidence` | EvidenceSpan | Best evidence for this entity |
| `supporting_evidence` | List[EvidenceSpan] | Additional evidence spans |
| `mention_count` | int | Times entity appears in document |
| `pages_mentioned` | List[int] | Page numbers where found |
| `status` | ValidationStatus | APPROVED / REJECTED / PENDING |
| `confidence_score` | float | Final confidence (0.0-1.0) |
| `rejection_reason` | Optional[str] | Why rejected (if applicable) |
| `validation_flags` | List[str] | Validation notes and warnings |
| `category` | Optional[AbbreviationCategory] | Semantic domain category |
| `provenance` | ProvenanceMetadata | Full audit trail |
| `raw_llm_response` | Optional[dict/str] | Raw LLM validation response |

### FieldType Enum

| Value | Description |
|-------|-------------|
| `DEFINITION_PAIR` | Both SF and LF found together (LF required) |
| `GLOSSARY_ENTRY` | From structured glossary section (LF required) |
| `SHORT_FORM_ONLY` | Only SF found, LF may be added later |

### GeneratorType Enum

| Value | Description |
|-------|-------------|
| `SYNTAX` | Schwartz-Hearst algorithm (C01) |
| `LEXICON_MATCH` | FlashText keyword matching (C04) |
| `GLOSSARY` | Structured glossary extraction (C05) |
| `REGEX` | Regular expression patterns |
| `NER` | scispacy named entity recognition |
| `LLM` | Language model extraction |

### AbbreviationCategory Enum

| Value | Description |
|-------|-------------|
| `ABBREV` | General abbreviation |
| `STATISTICAL` | Statistical term (CI, HR, OR, SD) |
| `DISEASE` | Disease abbreviation (PAH, IgAN) |
| `DRUG` | Drug abbreviation (RTX, CYC) |
| `GENE` | Gene symbol used as abbreviation |
| `STUDY` | Study/trial name abbreviation |
| `ORGANIZATION` | Organization abbreviation (WHO, FDA) |

---

## 3. Diseases

**Models**: `A_core/A05_disease_models.py`

### DiseaseCandidate (Pre-Validation)

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Auto-generated |
| `doc_id` | str | Source document |
| `matched_text` | str | Exact text matched in document |
| `preferred_label` | str | Canonical disease name from lexicon |
| `abbreviation` | Optional[str] | Disease abbreviation (PAH, IgAN) |
| `synonyms` | List[str] | Known synonyms |
| `field_type` | DiseaseFieldType | Detection method |
| `generator_type` | DiseaseGeneratorType | Source generator |
| `identifiers` | List[DiseaseIdentifier] | Medical codes |
| `context_text` | str | Surrounding text |
| `context_location` | Coordinate | Page and bbox |
| `is_rare_disease` | bool | Orphanet rare disease flag |
| `prevalence` | Optional[str] | e.g., "<1/1000000" |
| `parent_disease` | Optional[str] | Parent disease category |
| `disease_category` | Optional[str] | Clinical specialty (nephrology, etc.) |
| `initial_confidence` | float | Pre-validation confidence |
| `confidence_boost` | float | Context keyword boost (0.0-0.5) |
| `provenance` | DiseaseProvenanceMetadata | Audit trail with medical codes |

### ExtractedDisease (Post-Validation)

All DiseaseCandidate fields plus:

| Field | Type | Description |
|-------|------|-------------|
| `candidate_id` | UUID | Link to source candidate |
| `schema_version` | str | "1.0.0" |
| `icd10_code` | Optional[str] | ICD-10 code |
| `icd11_code` | Optional[str] | ICD-11 code |
| `snomed_code` | Optional[str] | SNOMED CT code |
| `mondo_id` | Optional[str] | MONDO identifier |
| `orpha_code` | Optional[str] | Orphanet code |
| `umls_cui` | Optional[str] | UMLS Concept Unique Identifier |
| `mesh_id` | Optional[str] | MeSH identifier |
| `primary_evidence` | EvidenceSpan | Best evidence |
| `supporting_evidence` | List[EvidenceSpan] | Additional evidence |
| `mention_count` | int | Occurrence count |
| `pages_mentioned` | List[int] | Pages where found |
| `status` | ValidationStatus | Validation result |
| `confidence_score` | float | Final confidence |
| `mesh_aliases` | List[str] | PubTator MeSH aliases |
| `pubtator_normalized_name` | Optional[str] | PubTator canonical name |
| `enrichment_source` | Optional[str] | Which API enriched this |

### DiseaseIdentifier

| Field | Type | Description |
|-------|------|-------------|
| `system` | str | Ontology: ICD-10, ICD-11, ICD-10-CM, SNOMED-CT, MONDO, ORPHA, MeSH, UMLS |
| `code` | str | Code value (e.g., "I27.0", "MONDO_0011055") |
| `display` | Optional[str] | Human-readable name |

### DiseaseFieldType Enum

| Value | Description |
|-------|-------------|
| `EXACT_MATCH` | Exact string match from lexicon |
| `PATTERN_MATCH` | Regex pattern match |
| `NER_DETECTION` | scispacy/NER detected |
| `ABBREVIATION_EXPAND` | Expanded from disease abbreviation |

### DiseaseGeneratorType Enum

| Value | Description |
|-------|-------------|
| `gen:disease_lexicon_specialized` | PAH, ANCA, IgAN specialized lexicons |
| `gen:disease_lexicon_general` | General disease lexicon (29K terms) |
| `gen:disease_lexicon_orphanet` | Orphanet rare diseases (9.5K terms) |
| `gen:disease_scispacy_ner` | scispacy entity recognition |

---

## 4. Drugs

**Models**: `A_core/A06_drug_models.py`

### DrugCandidate (Pre-Validation)

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Auto-generated |
| `doc_id` | str | Source document |
| `matched_text` | str | Exact text matched |
| `preferred_name` | str | Generic/canonical drug name |
| `brand_name` | Optional[str] | Brand name (e.g., FABHALTA) |
| `compound_id` | Optional[str] | Development code (e.g., LNP023) |
| `field_type` | DrugFieldType | Detection method |
| `generator_type` | DrugGeneratorType | Source generator |
| `identifiers` | List[DrugIdentifier] | Drug codes |
| `context_text` | str | Surrounding text |
| `context_location` | Coordinate | Page and bbox |
| `drug_class` | Optional[str] | e.g., "Factor B inhibitor" |
| `mechanism` | Optional[str] | Mechanism of action |
| `development_phase` | Optional[str] | "Phase 3", "Approved" |
| `is_investigational` | bool | Whether drug is investigational |
| `sponsor` | Optional[str] | Pharmaceutical company |
| `conditions` | List[str] | Target indications |
| `nct_id` | Optional[str] | ClinicalTrials.gov NCT ID |
| `dosage_form` | Optional[str] | "TABLET", "INJECTABLE" |
| `route` | Optional[str] | "ORAL", "INJECTION" |
| `marketing_status` | Optional[str] | "Prescription", "Discontinued" |
| `initial_confidence` | float | Pre-validation confidence |
| `provenance` | DrugProvenanceMetadata | Audit trail |

### ExtractedDrug (Post-Validation)

All DrugCandidate fields plus:

| Field | Type | Description |
|-------|------|-------------|
| `candidate_id` | UUID | Link to source candidate |
| `schema_version` | str | "1.0.0" |
| `rxcui` | Optional[str] | RxNorm Concept Unique ID |
| `mesh_id` | Optional[str] | MeSH identifier |
| `ndc_code` | Optional[str] | National Drug Code |
| `drugbank_id` | Optional[str] | DrugBank identifier |
| `unii` | Optional[str] | FDA Unique Ingredient Identifier |
| `primary_evidence` | EvidenceSpan | Best evidence |
| `supporting_evidence` | List[EvidenceSpan] | Additional evidence |
| `mention_count` | int | Occurrence count |
| `pages_mentioned` | List[int] | Pages where found |
| `status` | ValidationStatus | Validation result |
| `confidence_score` | float | Final confidence |
| `mesh_aliases` | List[str] | PubTator MeSH aliases |
| `pubtator_normalized_name` | Optional[str] | PubTator canonical name |

### DrugIdentifier

| Field | Type | Description |
|-------|------|-------------|
| `system` | str | Database: RxCUI, NDC, MeSH, DrugBank, ChEMBL, NCT, UNII |
| `code` | str | Code value |
| `display` | Optional[str] | Human-readable name |

### DevelopmentPhase Enum

| Value | Description |
|-------|-------------|
| `Preclinical` | Before human trials |
| `Phase 1` | First-in-human safety |
| `Phase 2` | Dose-finding efficacy |
| `Phase 3` | Pivotal confirmatory |
| `Approved` | Regulatory approved |
| `Withdrawn` | Market withdrawal |
| `Unknown` | Phase not determined |

### DrugGeneratorType Enum

| Value | Description |
|-------|-------------|
| `gen:drug_lexicon_alexion` | Alexion pipeline drugs |
| `gen:drug_lexicon_investigational` | Clinical trial drugs |
| `gen:drug_lexicon_fda` | FDA approved drugs (23K) |
| `gen:drug_lexicon_rxnorm` | RxNorm general terms (132K) |
| `gen:drug_pattern_compound` | Compound ID regex patterns |
| `gen:drug_scispacy_ner` | scispacy CHEMICAL detection |

---

## 5. Genes

**Models**: `A_core/A19_gene_models.py`

### GeneCandidate (Pre-Validation)

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Auto-generated |
| `doc_id` | str | Source document |
| `matched_text` | str | Exact text matched |
| `hgnc_symbol` | str | Official HGNC symbol (canonical) |
| `full_name` | Optional[str] | Full gene name |
| `is_alias` | bool | Whether matched via alias/synonym |
| `alias_of` | Optional[str] | Canonical symbol if alias (required when is_alias=True) |
| `field_type` | GeneFieldType | Detection method |
| `generator_type` | GeneGeneratorType | Source generator |
| `identifiers` | List[GeneIdentifier] | Gene codes |
| `context_text` | str | Surrounding text |
| `context_location` | Coordinate | Page and bbox |
| `locus_type` | Optional[str] | "protein-coding", "ncRNA", "pseudogene" |
| `chromosome` | Optional[str] | Chromosomal location |
| `associated_diseases` | List[GeneDiseaseLinkage] | Rare disease associations (Orphadata) |
| `initial_confidence` | float | Pre-validation confidence |
| `provenance` | GeneProvenanceMetadata | Audit trail |

### ExtractedGene (Post-Validation)

All GeneCandidate fields plus:

| Field | Type | Description |
|-------|------|-------------|
| `candidate_id` | UUID | Link to source candidate |
| `schema_version` | str | "1.0.0" |
| `hgnc_id` | Optional[str] | HGNC identifier (e.g., "HGNC:1100") |
| `entrez_id` | Optional[str] | NCBI Entrez Gene ID |
| `ensembl_id` | Optional[str] | Ensembl gene ID |
| `omim_id` | Optional[str] | OMIM number |
| `uniprot_id` | Optional[str] | UniProt accession |
| `primary_evidence` | EvidenceSpan | Best evidence |
| `supporting_evidence` | List[EvidenceSpan] | Additional evidence |
| `mention_count` | int | Occurrence count |
| `pages_mentioned` | List[int] | Pages where found |
| `status` | ValidationStatus | Validation result |
| `confidence_score` | float | Final confidence |
| `pubtator_normalized_name` | Optional[str] | PubTator canonical name |
| `pubtator_aliases` | List[str] | PubTator known aliases |
| `enrichment_source` | Optional[EnrichmentSource] | Which API enriched this |

### GeneIdentifier

| Field | Type | Description |
|-------|------|-------------|
| `system` | str | Database: HGNC, ENTREZ, ENSEMBL, OMIM, UNIPROT, ORPHACODE |
| `code` | str | Code value (e.g., "HGNC:1100", "672") |
| `display` | Optional[str] | Human-readable name |

### GeneDiseaseLinkage

| Field | Type | Description |
|-------|------|-------------|
| `orphacode` | str | Orphanet disease code |
| `disease_name` | str | Disease name |
| `association_type` | Optional[str] | Type of gene-disease association |
| `association_status` | Optional[str] | Status of the association |

---

## 6. Authors

**Models**: `A_core/A10_author_models.py`

### AuthorCandidate

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Auto-generated |
| `doc_id` | str | Source document |
| `full_name` | str | Author full name |
| `first_name` | Optional[str] | First/given name |
| `last_name` | Optional[str] | Last/family name |
| `role` | AuthorRoleType | Author's role |
| `affiliation` | Optional[str] | Institutional affiliation |
| `orcid` | Optional[str] | ORCID identifier (0000-0000-0000-000X) |
| `email` | Optional[str] | Contact email |
| `generator_type` | AuthorGeneratorType | Source generator |
| `context_text` | str | Surrounding text |
| `context_location` | Coordinate | Page and bbox |
| `provenance` | AuthorProvenanceMetadata | Audit trail |

### AuthorRoleType Enum

| Value | Description |
|-------|-------------|
| `author` | General author |
| `principal_investigator` | Principal Investigator (PI) |
| `co_investigator` | Co-Investigator |
| `corresponding_author` | Corresponding author |
| `steering_committee` | Steering committee member |
| `study_chair` | Study chair |
| `data_safety_board` | Data safety monitoring board |
| `unknown` | Role not determined |

### AuthorGeneratorType Enum

| Value | Description |
|-------|-------------|
| `gen:author_header` | From document header/title section |
| `gen:author_affiliation` | From affiliation section |
| `gen:author_contribution` | From contributions section |
| `gen:author_investigator` | From investigator listing |
| `gen:author_regex` | Regex pattern match |

---

## 7. Citations

**Models**: `A_core/A11_citation_models.py`

### CitationCandidate

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Auto-generated |
| `doc_id` | str | Source document |
| `pmid` | Optional[str] | PubMed identifier |
| `pmcid` | Optional[str] | PubMed Central identifier |
| `doi` | Optional[str] | Digital Object Identifier |
| `nct_id` | Optional[str] | ClinicalTrials.gov NCT number |
| `url` | Optional[str] | URL if present |
| `citation_text` | Optional[str] | Full citation text |
| `citation_number` | Optional[int] | Reference number in document |
| `identifier_types` | List[CitationIdentifierType] | Types of identifiers found |
| `generator_type` | CitationGeneratorType | Source generator |
| `context_text` | str | Surrounding text |
| `context_location` | Coordinate | Page and bbox |
| `provenance` | CitationProvenanceMetadata | Audit trail |

### CitationIdentifierType Enum

| Value | Description |
|-------|-------------|
| `pmid` | PubMed identifier |
| `pmcid` | PubMed Central identifier |
| `doi` | Digital Object Identifier |
| `nct` | ClinicalTrials.gov NCT number |
| `url` | Generic URL |

### CitationGeneratorType Enum

| Value | Description |
|-------|-------------|
| `gen:citation_regex` | Regex pattern match |
| `gen:citation_reference` | From reference section |
| `gen:citation_inline` | Inline citation marker |

### CitationValidation

| Field | Type | Description |
|-------|------|-------------|
| `identifier_type` | str | Type of identifier validated |
| `identifier_value` | str | The identifier value |
| `is_valid` | bool | Whether identifier resolves |
| `resolved_url` | Optional[str] | Resolved URL if valid |
| `title` | Optional[str] | Article/study title if retrieved |
| `error_message` | Optional[str] | Why validation failed |

---

## 8. Pharma Companies

**Models**: `A_core/A09_pharma_models.py`

### PharmaCandidate

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Auto-generated |
| `doc_id` | str | Source document |
| `matched_text` | str | Exact text matched |
| `canonical_name` | str | Official company name |
| `headquarters` | Optional[str] | Company headquarters location |
| `parent_company` | Optional[str] | Parent/holding company |
| `context_text` | str | Surrounding text |
| `context_location` | Coordinate | Page and bbox |
| `provenance` | PharmaProvenanceMetadata | Audit trail |

---

## 9. Document Metadata

**Models**: `A_core/A08_document_metadata_models.py`

### FileMetadata

| Field | Type | Description |
|-------|------|-------------|
| `filename` | str | PDF filename |
| `file_size_bytes` | int | File size |
| `file_hash` | str | SHA-256 hash of PDF bytes |
| `created_at` | Optional[datetime] | File creation date |
| `modified_at` | Optional[datetime] | File modification date |

### PDFMetadata

| Field | Type | Description |
|-------|------|-------------|
| `title` | Optional[str] | PDF title property |
| `author` | Optional[str] | PDF author property |
| `creator` | Optional[str] | PDF creator application |
| `producer` | Optional[str] | PDF producer application |
| `creation_date` | Optional[datetime] | PDF creation date |
| `modification_date` | Optional[datetime] | PDF modification date |
| `page_count` | int | Number of pages |
| `pdf_version` | Optional[str] | PDF specification version |

### DocumentClassification

| Field | Type | Description |
|-------|------|-------------|
| `document_type` | str | LLM-classified type (e.g., "clinical_trial_protocol") |
| `confidence` | float | Classification confidence |
| `sub_type` | Optional[str] | More specific type |
| `therapeutic_area` | Optional[str] | Medical specialty |

### DateExtractionResult

| Field | Type | Description |
|-------|------|-------------|
| `primary_date` | Optional[datetime] | Best-determined document date |
| `date_source` | Optional[str] | Where the date came from |
| `all_dates` | List[dict] | All dates found with sources |
| `fallback_chain` | List[str] | Priority order of date sources tried |

---

## 10. Feasibility

**Models**: `A_core/A07_feasibility_models.py`, `A_core/A21_clinical_criteria.py`, `A_core/A22_logical_expressions.py`

The feasibility model is the most complex, with deeply nested sub-structures.

### FeasibilityFieldType Enum

| Value | Description |
|-------|-------------|
| `ELIGIBILITY_INCLUSION` | Inclusion criterion |
| `ELIGIBILITY_EXCLUSION` | Exclusion criterion |
| `EPIDEMIOLOGY_PREVALENCE` | Prevalence data |
| `EPIDEMIOLOGY_INCIDENCE` | Incidence data |
| `EPIDEMIOLOGY_DEMOGRAPHICS` | Demographics statistics |
| `PATIENT_JOURNEY_PHASE` | Trial phase definition |
| `STUDY_ENDPOINT` | Primary/secondary endpoint |
| `STUDY_SITE` | Study site information |
| `STUDY_FOOTPRINT` | Geographic distribution |
| `STUDY_DESIGN` | Phase, blinding, arms |
| `STUDY_DURATION` | Trial duration |
| `TREATMENT_PATHWAY` | Treatment pathway description |
| `SCREENING_YIELD` | Screen-to-randomization yield |
| `SCREENING_FLOW` | CONSORT flow data |
| `VACCINATION_REQUIREMENT` | Vaccination requirements |
| `OPERATIONAL_BURDEN` | Visit/procedure burden |
| `INVASIVE_PROCEDURE` | Invasive procedures |
| `BACKGROUND_THERAPY` | Concomitant medications |
| `LAB_CRITERION` | Lab value criterion |

### EligibilityCriterion

| Field | Type | Description |
|-------|------|-------------|
| `criterion_type` | CriterionType | `inclusion` or `exclusion` |
| `text` | str | Original criterion text |
| `category` | Optional[str] | Category (age, diagnosis, lab, etc.) |
| `parsed_value` | Optional[dict] | Structured parsed data |
| `lab_criterion` | Optional[LabCriterion] | Parsed lab value with units/ranges |
| `diagnosis_confirmation` | Optional[DiagnosisConfirmation] | Diagnosis requirements |
| `severity_grade` | Optional[SeverityGrade] | Severity grading |
| `logical_expression` | Optional[LogicalExpression] | AND/OR grouping of sub-criteria |

### LabCriterion

| Field | Type | Description |
|-------|------|-------------|
| `test_name` | str | Lab test name (e.g., "eGFR") |
| `value_min` | Optional[float] | Minimum value |
| `value_max` | Optional[float] | Maximum value |
| `unit` | Optional[str] | Unit (e.g., "mL/min/1.73m2") |
| `timepoint` | Optional[LabTimepoint] | When the test should be done |

### ScreeningFlow

| Field | Type | Description |
|-------|------|-------------|
| `screened` | Optional[int] | Number screened |
| `screen_failures` | Optional[int] | Number of screen failures |
| `randomized` | Optional[int] | Number randomized |
| `screen_failure_rate` | Optional[float] | Calculated failure rate |
| `failure_reasons` | List[dict] | Reasons for screen failure |

### OperationalBurden

| Field | Type | Description |
|-------|------|-------------|
| `total_visits` | Optional[int] | Total study visits |
| `visit_frequency` | Optional[str] | Visit frequency description |
| `procedures_per_visit` | Optional[int] | Average procedures per visit |
| `blood_draw_volume_ml` | Optional[float] | Total blood draw volume |
| `invasive_procedures` | List[InvasiveProcedure] | List of invasive procedures |
| `visit_schedule` | Optional[VisitSchedule] | Detailed visit schedule |

### StudyDesign

| Field | Type | Description |
|-------|------|-------------|
| `phase` | Optional[str] | Trial phase (1, 2, 3, 4) |
| `blinding` | Optional[str] | Blinding type (open, single, double) |
| `randomization_ratio` | Optional[str] | Randomization ratio (e.g., "1:1") |
| `treatment_arms` | List[str] | Treatment arm descriptions |
| `comparator` | Optional[str] | Comparator (placebo, active, SOC) |
| `primary_endpoint` | Optional[str] | Primary endpoint description |
| `secondary_endpoints` | List[str] | Secondary endpoints |
| `study_duration` | Optional[str] | Total study duration |

### EpidemiologyData

| Field | Type | Description |
|-------|------|-------------|
| `prevalence` | Optional[str] | Disease prevalence |
| `incidence` | Optional[str] | Disease incidence |
| `geographic_distribution` | Optional[str] | Geographic patterns |
| `demographics` | Optional[dict] | Age, sex, ethnicity data |

### PatientJourneyPhaseType Enum

| Value | Description |
|-------|-------------|
| `screening` | Initial screening period |
| `run_in` | Run-in/wash-out period |
| `randomization` | Randomization visit |
| `treatment` | Active treatment period |
| `follow_up` | Post-treatment follow-up |
| `extension` | Open-label extension |

### LogicalExpression

| Field | Type | Description |
|-------|------|-------------|
| `operator` | LogicalOperator | `AND` or `OR` |
| `children` | List[CriterionNode] | Sub-criteria linked by operator |
| `text` | Optional[str] | Original text of the expression |

---

## 11. Recommendations

**Models**: `A_core/A18_recommendation_models.py`

### GuidelineRecommendation

| Field | Type | Description |
|-------|------|-------------|
| `recommendation_id` | str | Unique identifier |
| `recommendation_type` | RecommendationType | Category of recommendation |
| `population` | str | Target patient population |
| `condition` | Optional[str] | Clinical condition context |
| `severity` | Optional[str] | Disease severity qualification |
| `action` | str | Recommended action |
| `action_description` | Optional[str] | Detailed action description |
| `preferred` | Optional[str] | Preferred option if alternatives exist |
| `alternatives` | List[str] | Alternative treatments |
| `dosing` | List[DrugDosingInfo] | Dosing information |
| `taper_target` | Optional[str] | Taper target (e.g., "5 mg/day by 4-5 months") |
| `duration` | Optional[str] | Treatment duration (e.g., "24-48 months") |
| `stop_window` | Optional[str] | When to stop (e.g., "6-12 months") |
| `evidence_level` | EvidenceLevel | Evidence quality |
| `strength` | RecommendationStrength | Recommendation strength |
| `references` | List[str] | Supporting references (PMID, DOI) |
| `source` | str | Where found ("Table 2", "Figure 1") |
| `source_text` | Optional[str] | Original text |
| `page_num` | Optional[int] | Page number |

### RecommendationSet

| Field | Type | Description |
|-------|------|-------------|
| `guideline_name` | str | Guideline name (e.g., "2022 EULAR recommendations for AAV") |
| `guideline_year` | Optional[int] | Publication year |
| `organization` | Optional[str] | Issuing organization (EULAR, ACR, FDA) |
| `target_condition` | str | Target condition |
| `target_population` | Optional[str] | Target population |
| `recommendations` | List[GuidelineRecommendation] | All recommendations |
| `source_document` | Optional[str] | Source document |
| `extraction_confidence` | float | Overall extraction confidence |

### DrugDosingInfo

| Field | Type | Description |
|-------|------|-------------|
| `drug_name` | str | Drug name |
| `dose_range` | Optional[str] | Dose range (e.g., "50-75 mg/day") |
| `starting_dose` | Optional[str] | Starting dose |
| `maintenance_dose` | Optional[str] | Maintenance dose |
| `max_dose` | Optional[str] | Maximum dose |
| `route` | Optional[str] | Route (oral, IV, SC) |
| `frequency` | Optional[str] | Frequency (daily, weekly) |

### EvidenceLevel Enum

| Value | Description |
|-------|-------------|
| `high` | High-quality evidence |
| `moderate` | Moderate evidence |
| `low` | Low-quality evidence |
| `very_low` | Very low evidence |
| `expert_opinion` | Expert consensus only |
| `unknown` | Level not determined |

### RecommendationStrength Enum

| Value | Description |
|-------|-------------|
| `strong` | Strong recommendation |
| `conditional` | Conditional/weak recommendation |
| `weak` | Weak recommendation |
| `unknown` | Strength not determined |

### RecommendationType Enum

| Value | Description |
|-------|-------------|
| `treatment` | Treatment recommendation |
| `dosing` | Dosing guidance |
| `duration` | Treatment duration |
| `monitoring` | Monitoring recommendation |
| `contraindication` | Contraindication warning |
| `alternative` | Alternative treatment |
| `preference` | Preference statement |
| `other` | Other recommendation |

---

## 12. Care Pathways

**Models**: `A_core/A17_care_pathway_models.py`

### CarePathway

| Field | Type | Description |
|-------|------|-------------|
| `pathway_id` | str | Unique identifier |
| `title` | str | Pathway title |
| `condition` | str | Target condition (e.g., "Active GPA/MPA") |
| `guideline_source` | Optional[str] | Source guideline |
| `phases` | List[str] | Treatment phases (["Induction", "Maintenance"]) |
| `nodes` | List[CarePathwayNode] | Pathway steps |
| `edges` | List[CarePathwayEdge] | Transitions between steps |
| `entry_criteria` | Optional[str] | Who this pathway applies to |
| `primary_drugs` | List[str] | Main treatment drugs |
| `alternative_drugs` | List[str] | Alternative drugs |
| `decision_points` | List[str] | Key decision summaries |
| `target_dose` | Optional[str] | Target dose (e.g., "5 mg/day") |
| `target_timepoint` | Optional[str] | Target timepoint |
| `maintenance_duration` | Optional[str] | Maintenance duration |
| `relapse_handling` | Optional[str] | Relapse management |
| `source_figure` | Optional[str] | Source figure ID |
| `source_page` | Optional[int] | Page number |
| `extraction_confidence` | float | Confidence score |

### CarePathwayNode

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Unique node identifier |
| `type` | NodeType | Node type (see enum) |
| `text` | str | Raw text from flowchart box/diamond |
| `phase` | Optional[str] | Treatment phase (induction, maintenance) |
| `drugs` | List[str] | Drug names mentioned |
| `dose` | Optional[str] | Dosing information |
| `duration` | Optional[str] | Duration |
| `frequency` | Optional[str] | Frequency |
| `position` | Optional[dict] | Position for visualization |
| `source_bbox` | Optional[List[float]] | Bounding box in figure |

### CarePathwayEdge

| Field | Type | Description |
|-------|------|-------------|
| `source_id` | str | Source node ID |
| `target_id` | str | Target node ID |
| `condition` | Optional[str] | Transition condition ("Yes", "No", etc.) |
| `condition_type` | Optional[str] | Condition category |
| `label` | Optional[str] | Additional edge label |

### NodeType Enum

| Value | Description |
|-------|-------------|
| `start` | Pathway start point |
| `end` | Pathway end point |
| `action` | Treatment action (prescribe drug, perform procedure) |
| `decision` | Decision point (diamond shape) |
| `assessment` | Clinical assessment |
| `note` | Informational note |
| `condition` | Entry condition/prerequisite |

### TaperSchedule

| Field | Type | Description |
|-------|------|-------------|
| `schedule_id` | str | Unique identifier |
| `drug_name` | str | Drug being tapered (e.g., "prednisone") |
| `regimen_name` | str | Regimen name (e.g., "Protocol target") |
| `schedule` | List[TaperSchedulePoint] | Timepoint-dose pairs |
| `starting_dose` | Optional[str] | Starting dose |
| `target_dose` | Optional[str] | Target dose |
| `target_timepoint` | Optional[str] | Target timepoint |
| `source_figure` | Optional[str] | Source figure |
| `source_page` | Optional[int] | Page number |
| `legend_color` | Optional[str] | Chart legend color |
| `legend_marker` | Optional[str] | Chart legend marker |
| `linked_recommendation_text` | Optional[str] | Linked recommendation |

### TaperSchedulePoint

| Field | Type | Description |
|-------|------|-------------|
| `week` | Optional[int] | Week number |
| `day` | Optional[int] | Day number |
| `month` | Optional[float] | Month number |
| `dose_mg` | float | Dose in milligrams |
| `dose_unit` | str | Unit (default: "mg/day") |

---

## 13. Visual Elements (Figures & Tables)

**Models**: `A_core/A13_visual_models.py`

### VisualCandidate (Pre-Validation)

| Field | Type | Description |
|-------|------|-------------|
| `visual_type` | VisualType | TABLE or FIGURE |
| `page_range` | List[int] | Pages where visual appears |
| `bbox_pts_per_page` | List[PageBBox] | Bounding boxes per page |
| `caption_candidate` | Optional[CaptionCandidate] | Associated caption |
| `reference_number` | Optional[VisualReference] | "Table 1", "Figure 2" |
| `docling_table` | Optional[TableStructure] | Docling-extracted table data |
| `validated_table` | Optional[TableStructure] | VLM-validated table data |
| `vlm_enrichment` | Optional[VLMEnrichmentResult] | VLM analysis results |
| `image_path` | Optional[str] | Path to extracted image |
| `image_bytes` | Optional[bytes] | Raw image bytes |

### ExtractedVisual (Post-Validation)

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Auto-generated |
| `visual_type` | VisualType | TABLE or FIGURE |
| `page_range` | List[int] | Pages where visual appears |
| `bbox_pts_per_page` | List[PageBBox] | Bounding boxes per page |
| `caption_text` | Optional[str] | Caption text |
| `caption_provenance` | Optional[CaptionProvenance] | How caption was found |
| `reference` | Optional[VisualReference] | Parsed reference |
| `docling_table` | Optional[TableStructure] | Docling table structure |
| `validated_table` | Optional[TableStructure] | VLM-validated table |
| `image_path` | Optional[str] | Path to PNG file |
| `vlm_title` | Optional[str] | VLM-generated title |
| `vlm_description` | Optional[str] | VLM-generated description |
| `layout_code` | Optional[str] | Page layout pattern |
| `position_code` | Optional[str] | Position in layout |
| `layout_filename` | Optional[str] | Layout-encoded filename |

### VisualType Enum

| Value | Description |
|-------|-------------|
| `TABLE` | Data table |
| `FIGURE` | Figure, chart, diagram, or photograph |

### CaptionProvenance Enum

| Value | Description |
|-------|-------------|
| `pdf_text` | From PDF text layer |
| `ocr` | From OCR extraction |
| `vlm` | From Vision LLM analysis |

### TableStructure

| Field | Type | Description |
|-------|------|-------------|
| `headers` | List[str] | Column headers |
| `rows` | List[List[str]] | Row data |
| `merged_cells` | List[dict] | Merged cell definitions |
| `source` | str | Extraction source ("docling", "vlm") |

### VLMEnrichmentResult

| Field | Type | Description |
|-------|------|-------------|
| `classification` | VLMClassificationResult | Type + confidence |
| `parsed_reference` | Optional[VisualReference] | Parsed "Table 1" etc. |
| `extracted_caption` | Optional[str] | VLM-extracted caption |
| `table_validation` | Optional[VLMTableValidation] | Table quality check |
| `is_continuation` | bool | Whether this continues a previous visual |
| `continuation_of_reference` | Optional[str] | Reference it continues |

---

## 14. Provenance Model

**Models**: `A_core/A01_domain_models.py` (ProvenanceMetadata), `A_core/A03_provenance.py`

Every extracted entity carries provenance metadata for reproducibility and audit compliance.

### ProvenanceMetadata (Abbreviations)

| Field | Type | Description |
|-------|------|-------------|
| `pipeline_version` | str | Git commit hash or version tag |
| `run_id` | str | Unique run identifier (e.g., "RUN_20250101_120000_ab12cd34ef56") |
| `doc_fingerprint` | str | SHA-256 hash of source PDF bytes |
| `generator_name` | GeneratorType | Which generator produced the entity |
| `rule_version` | Optional[str] | Version of extraction rule |
| `lexicon_source` | Optional[str] | Dictionary file the match came from |
| `lexicon_ids` | Optional[List[LexiconIdentifier]] | External IDs (Orphanet, MONDO, etc.) |
| `prompt_bundle_hash` | Optional[str] | Hash of LLM prompt used |
| `context_hash` | Optional[str] | Hash of context text sent to LLM |
| `llm_config` | Optional[LLMParameters] | LLM model and settings used |
| `timestamp` | datetime | When extraction occurred |

### Entity-Specific Provenance

Each entity type extends `BaseProvenanceMetadata` with:

| Entity | Provenance Class | Specialized Fields |
|--------|-----------------|-------------------|
| Abbreviations | ProvenanceMetadata | LexiconIdentifier list |
| Diseases | DiseaseProvenanceMetadata | DiseaseIdentifier list |
| Drugs | DrugProvenanceMetadata | DrugIdentifier list |
| Genes | GeneProvenanceMetadata | GeneIdentifier list |
| Authors | AuthorProvenanceMetadata | (no lexicon IDs) |
| Citations | CitationProvenanceMetadata | (no lexicon IDs) |

### Shared Provenance Fields

| Field | Type | Description |
|-------|------|-------------|
| `pipeline_version` | str | For reproducibility |
| `run_id` | str | Links all entities from same processing run |
| `doc_fingerprint` | str | Ensures entity matches source document |
| `generator_name` | Enum | Tracks which strategy found the entity |
| `timestamp` | datetime | Exact extraction time |

---

## 15. Output Structure

### Per-Document Folder Layout

Processing `document.pdf` creates a folder alongside the PDF:

```
document/
├── abbreviations_document_YYYYMMDD_HHMMSS.json
├── diseases_document_YYYYMMDD_HHMMSS.json
├── drugs_document_YYYYMMDD_HHMMSS.json
├── genes_document_YYYYMMDD_HHMMSS.json
├── recommendations_document_YYYYMMDD_HHMMSS.json
├── feasibility_document_YYYYMMDD_HHMMSS.json
├── figures_document_YYYYMMDD_HHMMSS.json
├── tables_document_YYYYMMDD_HHMMSS.json
├── metadata_document_YYYYMMDD_HHMMSS.json
├── authors_document_YYYYMMDD_HHMMSS.json
├── citations_document_YYYYMMDD_HHMMSS.json
├── document_extracted_text_YYYYMMDD_HHMMSS.txt
├── document_figure_page3_1.png
├── document_flowchart_page5_1.png
└── document_YYYYMMDD_HHMMSS.txt          # Console log
```

### JSON File Naming Convention

```
{entity_type}_{document_stem}_{YYYYMMDD}_{HHMMSS}.json
```

- `entity_type`: abbreviations, diseases, drugs, genes, recommendations, feasibility, figures, tables, metadata, authors, citations
- `document_stem`: PDF filename without extension
- `YYYYMMDD_HHMMSS`: Timestamp of extraction run

### JSON Structure

All entity JSON files follow a common envelope:

```json
{
  "document_id": "document.pdf",
  "extraction_timestamp": "2026-02-06T12:00:00Z",
  "pipeline_version": "v0.8",
  "schema_version": "1.0.0",
  "entity_count": 42,
  "entities": [
    {
      "matched_text": "...",
      "preferred_label": "...",
      "confidence": 0.95,
      "identifiers": [...],
      "evidence": {...},
      "pages": [1, 3, 7]
    }
  ]
}
```

### Export Entry Simplification

Each entity type has a simplified `ExportEntry` model that flattens the full `Extracted*` model for JSON output:

| Full Model | Export Model | Key Simplifications |
|------------|-------------|---------------------|
| ExtractedEntity | AbbreviationExportEntry | Drops UUID, provenance internals |
| ExtractedDisease | DiseaseExportEntry | Flattens identifiers to code strings |
| ExtractedDrug | DrugExportEntry | Flattens codes, drops raw LLM response |
| ExtractedGene | GeneExportEntry | Flattens identifiers, keeps disease links |
| ExtractedAuthor | AuthorExportEntry | Drops provenance, keeps role/affiliation |
| ExtractedCitation | CitationExportEntry | Adds validation results |

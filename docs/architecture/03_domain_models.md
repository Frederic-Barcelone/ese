# Domain Models

All domain models are defined in `corpus_metadata/A_core/` using Pydantic v2 with strict validation (`extra="forbid"`, frozen where appropriate). This document covers the key models, enums, and their relationships.

---

## Core Abbreviation Models (A01)

### Candidate

Pre-verification abbreviation candidate. High recall, noisy.

```python
class Candidate(BaseModel):
    id: uuid.UUID
    doc_id: str
    field_type: FieldType
    generator_type: GeneratorType
    short_form: str                     # Always required, non-empty
    long_form: Optional[str]            # Required for DEFINITION_PAIR and GLOSSARY_ENTRY
    context_text: str
    context_location: Coordinate
    initial_confidence: float           # [0.0, 1.0]
    provenance: ProvenanceMetadata
```

### ExtractedEntity

Post-verification abbreviation. Suitable for export and audit.

```python
class ExtractedEntity(BaseModel):
    id: uuid.UUID
    candidate_id: uuid.UUID
    doc_id: str
    schema_version: str                 # "1.0.0"
    field_type: FieldType
    short_form: str
    long_form: Optional[str]
    normalized_value: Optional[str | dict]
    standard_id: Optional[str]
    primary_evidence: EvidenceSpan
    supporting_evidence: list[EvidenceSpan]
    mention_count: int                  # >= 1
    pages_mentioned: list[int]
    status: ValidationStatus
    confidence_score: float             # [0.0, 1.0]
    rejection_reason: Optional[str]
    validation_flags: list[str]
    category: Optional[AbbreviationCategory]
    provenance: ProvenanceMetadata
    raw_llm_response: Optional[dict | str]
```

### FieldType Enum

How the abbreviation was presented in the document.

| Value | Description | Example |
|-------|-------------|---------|
| `DEFINITION_PAIR` | Parenthetical definition | "Tumor Necrosis Factor (TNF)" |
| `GLOSSARY_ENTRY` | Table/section glossary | "AE \| Adverse Event" |
| `SHORT_FORM_ONLY` | No definition in document | "The patient received TNF..." |

### GeneratorType Enum

Which C_generators strategy produced the candidate.

| Value | Strategy | Module |
|-------|----------|--------|
| `gen:syntax_pattern` | Schwartz-Hearst abbreviations | C01 |
| `gen:glossary_table` | Glossary tables | C01 |
| `gen:rigid_pattern` | DOI, trial IDs, doses | C02 |
| `gen:table_layout` | Spatial extraction | C03 |
| `gen:lexicon_match` | Dictionary matching | C04 |
| `gen:inline_definition` | Explicit inline definitions (SF=LF) | C04 |

### ValidationStatus Enum

| Value | Meaning |
|-------|---------|
| `VALIDATED` | Confirmed by LLM or heuristic rule |
| `REJECTED` | Determined to be a false positive |
| `AMBIGUOUS` | Uncertain; kept with lower confidence |

### AbbreviationCategory Enum

Semantic domain for abbreviation classification (separate from entity type):

`ABBREV`, `STATISTICAL`, `DISEASE`, `DRUG`, `GENE`, `STUDY`, `ORGANIZATION`, `UNKNOWN`

---

## Supporting Types (A01)

### BoundingBox

Strict coordinate representation for PDF locations.

```python
class BoundingBox(BaseModel):      # frozen=True
    coords: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    page_width: Optional[float]
    page_height: Optional[float]
    is_normalized: bool             # True if coords in [0, 1]
```

### Coordinate

Minimal audit-friendly location. Page number is 1-based.

```python
class Coordinate(BaseModel):       # frozen=True
    page_num: int                   # >= 1
    block_id: Optional[str]
    table_id: Optional[str]
    cell_row: Optional[int]
    cell_col: Optional[int]
    bbox: Optional[BoundingBox]
```

### EvidenceSpan

Proof snippet with character offsets scoped to a known text context.

```python
class EvidenceSpan(BaseModel):     # frozen=True
    text: str
    location: Coordinate
    scope_ref: str                  # Hash of context window or block_id
    start_char_offset: int          # >= 0
    end_char_offset: int            # >= start_char_offset
```

### ProvenanceMetadata

Audit trail for reproducibility and compliance.

```python
class ProvenanceMetadata(BaseModel):  # frozen=True
    pipeline_version: str           # Git commit hash
    run_id: str                     # "RUN_20250101_120000_ab12cd34ef56"
    doc_fingerprint: str            # SHA256 of source PDF bytes
    generator_name: GeneratorType
    rule_version: Optional[str]
    lexicon_source: Optional[str]
    lexicon_ids: Optional[list[LexiconIdentifier]]
    prompt_bundle_hash: Optional[str]
    context_hash: Optional[str]
    llm_config: Optional[LLMParameters]
    timestamp: datetime
```

---

## DocumentGraph (B02)

Top-level container for parsed PDF document structure.

```python
class DocumentGraph(BaseModel):
    doc_id: str
    pages: Dict[int, Page] = Field(default_factory=dict)
    metadata: Dict[str, str] = Field(default_factory=dict)

    # Methods:
    def get_page(self, page_num: int) -> Page: ...
    def iter_images(self) -> Iterator[ImageBlock]: ...
    def iter_linear_blocks(self, skip_header_footer: bool = True) -> Iterator[TextBlock]: ...
    def iter_tables(self, table_type: Optional[TableType] = None) -> Iterator[Table]: ...
```

### Page

```python
class Page(BaseModel):
    number: int                     # 1-based
    width: float
    height: float
    blocks: list[TextBlock]
    tables: list[Table]
    images: list[ImageBlock]
```

### TextBlock

Atomic text unit with content role and bounding box.

```python
class TextBlock(BaseModel):        # frozen=True
    id: str
    text: str
    page_num: int                   # 1-based
    reading_order_index: int        # 0-based per page
    role: ContentRole               # BODY_TEXT, SECTION_HEADER, PAGE_HEADER, etc.
    bbox: BoundingBox
```

### ContentRole Enum

`BODY_TEXT`, `SECTION_HEADER`, `PAGE_HEADER`, `PAGE_FOOTER`, `TABLE_CAPTION`, `TABLE_CELL`

---

## Disease Models (A05)

### DiseaseCandidate

```python
class DiseaseCandidate(BaseModel):
    id: uuid.UUID
    doc_id: str
    matched_text: str               # Exact text in document
    preferred_label: str            # Canonical disease name
    abbreviation: Optional[str]     # Disease abbreviation (PAH, IgAN)
    synonyms: list[str]
    field_type: DiseaseFieldType    # EXACT_MATCH, PATTERN_MATCH, NER_DETECTION, ABBREV_EXPAND
    generator_type: DiseaseGeneratorType
    identifiers: list[DiseaseIdentifier]
    context_text: str
    context_location: Coordinate
    is_rare_disease: bool
    prevalence: Optional[str]       # "<1/1000000"
    disease_category: Optional[str] # "nephrology", "pulmonology"
    provenance: DiseaseProvenanceMetadata
```

### ExtractedDisease

Validated disease entity with all medical codes populated.

```python
class ExtractedDisease(BaseModel):
    # ... (id, candidate_id, doc_id, schema_version)
    matched_text: str
    preferred_label: str
    identifiers: list[DiseaseIdentifier]
    icd10_code: Optional[str]       # Primary ICD-10 code
    icd11_code: Optional[str]
    snomed_code: Optional[str]
    mondo_id: Optional[str]
    orpha_code: Optional[str]
    umls_cui: Optional[str]
    mesh_id: Optional[str]
    status: ValidationStatus
    confidence_score: float
    is_rare_disease: bool
    mesh_aliases: list[str]         # From PubTator3
    pubtator_normalized_name: Optional[str]
    provenance: DiseaseProvenanceMetadata
```

### DiseaseIdentifier

```python
class DiseaseIdentifier(BaseModel):  # frozen=True
    system: str     # "ICD-10", "ICD-11", "SNOMED-CT", "MONDO", "ORPHA", "MeSH", "UMLS"
    code: str       # "I27.0", "MONDO_0011055", "ORPHA:182090"
    display: Optional[str]
```

---

## Drug Models (A06)

### DrugCandidate

```python
class DrugCandidate(BaseModel):
    id: uuid.UUID
    doc_id: str
    matched_text: str
    preferred_name: str             # Generic/canonical drug name
    brand_name: Optional[str]       # e.g., FABHALTA
    compound_id: Optional[str]      # e.g., LNP023
    field_type: DrugFieldType       # EXACT_MATCH, PATTERN_MATCH, NER_DETECTION
    generator_type: DrugGeneratorType
    identifiers: list[DrugIdentifier]
    drug_class: Optional[str]       # "Factor B inhibitor"
    development_phase: Optional[str] # "Phase 3", "Approved"
    is_investigational: bool
    sponsor: Optional[str]
    conditions: list[str]           # Target indications
    provenance: DrugProvenanceMetadata
```

### ExtractedDrug

```python
class ExtractedDrug(BaseModel):
    # ... (id, candidate_id, doc_id, schema_version)
    matched_text: str
    preferred_name: str
    brand_name: Optional[str]
    compound_id: Optional[str]
    identifiers: list[DrugIdentifier]
    rxcui: Optional[str]            # RxNorm Concept Unique Identifier
    mesh_id: Optional[str]
    drugbank_id: Optional[str]
    unii: Optional[str]             # FDA Unique Ingredient Identifier
    status: ValidationStatus
    confidence_score: float
    development_phase: Optional[str]
    is_investigational: bool
    provenance: DrugProvenanceMetadata
```

### DevelopmentPhase Enum

`Preclinical`, `Phase 1`, `Phase 2`, `Phase 3`, `Approved`, `Withdrawn`, `Unknown`

---

## Gene Models (A19)

### GeneCandidate

```python
class GeneCandidate(BaseModel):
    id: uuid.UUID
    doc_id: str
    matched_text: str
    hgnc_symbol: str                # Official HGNC symbol (canonical)
    full_name: Optional[str]
    is_alias: bool                  # True if matched via alias/synonym
    alias_of: Optional[str]         # Canonical symbol if alias
    field_type: GeneFieldType       # EXACT_MATCH, PATTERN_MATCH, NER_DETECTION
    generator_type: GeneGeneratorType
    identifiers: list[GeneIdentifier]
    locus_type: Optional[str]       # "protein-coding", "ncRNA", "pseudogene"
    chromosome: Optional[str]
    associated_diseases: list[GeneDiseaseLinkage]
    provenance: GeneProvenanceMetadata
```

### ExtractedGene

```python
class ExtractedGene(BaseModel):
    # ... (id, candidate_id, doc_id, schema_version)
    matched_text: str
    hgnc_symbol: str
    identifiers: list[GeneIdentifier]
    hgnc_id: Optional[str]          # "HGNC:1100"
    entrez_id: Optional[str]        # "672"
    ensembl_id: Optional[str]       # "ENSG00000012048"
    omim_id: Optional[str]
    uniprot_id: Optional[str]
    associated_diseases: list[GeneDiseaseLinkage]
    pubtator_normalized_name: Optional[str]  # PubTator3 canonical name
    pubtator_aliases: list[str]              # PubTator3 aliases
    enrichment_source: Optional[EnrichmentSource]
    status: ValidationStatus
    confidence_score: float
    provenance: GeneProvenanceMetadata
```

### GeneDiseaseLinkage

Links a gene to an associated rare disease from Orphadata.

```python
class GeneDiseaseLinkage(BaseModel):  # frozen=True
    orphacode: str
    disease_name: str
    association_type: Optional[str]   # "Disease-causing germline mutation(s) in", etc.
    association_status: Optional[str]
```

### GeneAssociationType Enum

`DISEASE_CAUSING`, `DISEASE_CAUSING_SOMATIC`, `MAJOR_SUSCEPTIBILITY`, `MODIFYING`, `ROLE_PATHOGENESIS`, `CANDIDATE`, `BIOMARKER`, `UNKNOWN`

---

## Feasibility Models (A07)

### FeasibilityCandidate

Top-level container for all feasibility data types.

```python
class FeasibilityCandidate(BaseModel):
    id: uuid.UUID
    doc_id: str
    field_type: FeasibilityFieldType
    generator_type: FeasibilityGeneratorType
    matched_text: str
    context_text: str
    page_number: Optional[int]
    section_name: Optional[str]
    evidence: list[EvidenceSpan]
    extraction_method: Optional[ExtractionMethod]  # REGEX, TABLE, LLM, HYBRID
    # One of these is populated based on field_type:
    eligibility_criterion: Optional[EligibilityCriterion]
    screening_flow: Optional[ScreeningFlow]
    study_design: Optional[StudyDesign]
    operational_burden: Optional[OperationalBurden]
    epidemiology_data: Optional[EpidemiologyData]
    study_footprint: Optional[StudyFootprint]
    # ... (and more)
    confidence: float
    provenance: FeasibilityProvenanceMetadata
```

### Key Sub-Models

- **EligibilityCriterion**: Inclusion/exclusion with lab values (`LabCriterion`), severity grades, logical expressions (`AND`/`OR` trees)
- **ScreeningFlow**: CONSORT flow metrics (screened, failures, randomized, completed), screen fail reasons with overlap tracking
- **StudyDesign**: Phase, blinding, randomization ratio, treatment arms, study periods
- **OperationalBurden**: Invasive procedures, visit schedule, vaccination requirements, background therapy, central lab requirements
- **StudyFootprint**: Sites total, countries total, geographic distribution
- **TrialIdentifier**: NCT, EudraCT, CTIS, ISRCTN registry identifiers

---

## Author Models (A10)

### AuthorCandidate / ExtractedAuthor

```python
class AuthorCandidate(BaseModel):
    id: uuid.UUID
    doc_id: str
    full_name: str
    role: AuthorRoleType            # AUTHOR, PRINCIPAL_INVESTIGATOR, CORRESPONDING_AUTHOR, etc.
    affiliation: Optional[str]
    email: Optional[str]
    orcid: Optional[str]            # ORCID identifier
    generator_type: AuthorGeneratorType
    provenance: AuthorProvenanceMetadata
```

### AuthorRoleType Enum

`AUTHOR`, `PRINCIPAL_INVESTIGATOR`, `CO_INVESTIGATOR`, `CORRESPONDING_AUTHOR`, `STEERING_COMMITTEE`, `STUDY_CHAIR`, `DATA_SAFETY_BOARD`, `UNKNOWN`

---

## Citation Models (A11)

### CitationCandidate / ExtractedCitation

```python
class CitationCandidate(BaseModel):
    id: uuid.UUID
    doc_id: str
    pmid: Optional[str]
    pmcid: Optional[str]
    doi: Optional[str]
    nct: Optional[str]
    url: Optional[str]
    citation_text: str
    citation_number: Optional[int]  # Reference number [1], [2], etc.
    generator_type: CitationGeneratorType
    identifier_types: list[CitationIdentifierType]
    provenance: CitationProvenanceMetadata
```

### CitationIdentifierType Enum

`PMID`, `PMCID`, `DOI`, `NCT`, `URL`

---

## Visual Models (A13)

### ExtractedVisual

Unified model for extracted tables and figures after all pipeline stages.

```python
class ExtractedVisual(BaseModel):
    visual_id: str
    visual_type: VisualType         # TABLE, FIGURE, OTHER
    confidence: float
    page_range: list[int]
    bbox_pts_per_page: list[PageLocation]
    caption_text: Optional[str]
    caption_provenance: Optional[CaptionProvenance]  # PDF_TEXT, OCR, VLM
    reference: Optional[VisualReference]
    image_base64: str               # Base64-encoded PNG
    image_format: str               # Default "png"
    render_dpi: int                 # Default 300
    source_file: str
    docling_table: Optional[TableStructure]
    validated_table: Optional[TableStructure]
    table_extraction_mode: Optional[TableExtractionMode]
    relationships: VisualRelationships
    extraction_method: str          # "docling+vlm", "docling_only", "vlm_only"
    vlm_title: Optional[str]
    vlm_description: Optional[str]
    triage_decision: Optional[TriageDecision]
    triage_reason: Optional[str]
    layout_code: Optional[str]      # Layout pattern code from B18
    position_code: Optional[str]    # Position within layout
    layout_filename: Optional[str]  # Generated filename from B21
```

### VisualType Enum

`TABLE`, `FIGURE`, `OTHER`

### TriageDecision Enum

`SKIP` (noise), `CHEAP_PATH` (minimal processing), `VLM_REQUIRED` (full VLM enrichment)

---

## Extraction Result (A14)

Universal output contract for deterministic pipeline results. Uses `dataclass(frozen=True)` instead of Pydantic for immutability and hash-based IDs.

### EntityType Enum

All extractable entity types:

| Value | Description |
|-------|-------------|
| `abbreviation` | Abbreviations/acronyms with definitions |
| `disease` | Disease mentions with medical codes |
| `drug` | Drug/chemical entities |
| `gene` | Gene/protein entities |
| `pharma_company` | Pharmaceutical companies |
| `author` | Authors/investigators |
| `citation` | Bibliographic references |
| `feasibility` | Clinical trial feasibility data |
| `metadata` | Document metadata |

### Provenance (A14)

Immutable location and audit trail (distinct from A01 ProvenanceMetadata).

```python
@dataclass(frozen=True)
class Provenance:
    page_num: int
    strategy_id: str
    bbox: Optional[tuple[float, ...]]
    node_ids: tuple[str, ...]
    char_span: Optional[tuple[int, int]]
    strategy_version: str
    doc_fingerprint: str
    lexicon_source: Optional[str]
    pipeline_version: str
    run_id: str
    timestamp: Optional[datetime]
```

---

## Pipeline Metrics (A16)

Single source of truth for all pipeline observability.

```python
class PipelineMetrics(BaseModel):
    run_id: str
    doc_id: str
    generation: GenerationMetrics       # Candidate counts from C_generators
    heuristics: HeuristicsMetrics       # PASO rule breakdown
    validation: ValidationMetrics       # LLM results, SF-only extraction
    normalization: NormalizationMetrics  # Disambiguation, deduplication
    export: ExportMetrics               # Final counts by entity type
    scoring: Optional[ScoringMetrics]   # Precision/recall/F1, TP/FP/FN
```

Includes `validate_invariants()` to check cross-stage consistency (e.g., `TP + FP == exported_validated`).

---

## Care Pathway Models (A17)

For clinical treatment algorithms extracted from guideline flowcharts.

```python
class CarePathway(BaseModel):
    pathway_id: str
    title: str
    condition: str                  # Target condition
    guideline_source: Optional[str]
    phases: list[str]               # ["Induction", "Maintenance"]
    nodes: list[CarePathwayNode]    # Action, decision, assessment nodes
    edges: list[CarePathwayEdge]    # Transitions with conditions
    primary_drugs: list[str]
    decision_points: list[str]
```

### NodeType Enum

`START`, `END`, `ACTION`, `DECISION`, `ASSESSMENT`, `NOTE`, `CONDITION`

### TaperSchedule

Structured medication dose reduction schedule (e.g., glucocorticoid tapers).

```python
class TaperSchedule(BaseModel):
    schedule_id: str
    drug_name: str
    regimen_name: str
    schedule: list[TaperSchedulePoint]  # week/day + dose_mg
    starting_dose: Optional[str]
    target_dose: Optional[str]
    target_timepoint: Optional[str]
```

---

## Guideline Recommendation Models (A18)

For structured treatment recommendations from clinical guidelines.

```python
class GuidelineRecommendation(BaseModel):
    recommendation_id: str
    recommendation_type: RecommendationType
    population: str                 # "GPA/MPA organ-threatening"
    action: str                     # "GC + RTX or CYC"
    evidence_level: EvidenceLevel
    strength: RecommendationStrength
    dosing: list[DrugDosingInfo]
    taper_target: Optional[str]
    duration: Optional[str]
```

### EvidenceLevel Enum

`HIGH`, `MODERATE`, `LOW`, `VERY_LOW`, `EXPERT_OPINION`, `UNKNOWN`

### RecommendationStrength Enum

`STRONG`, `CONDITIONAL`, `WEAK`, `UNKNOWN`

### RecommendationType Enum

`TREATMENT`, `DOSING`, `DURATION`, `MONITORING`, `CONTRAINDICATION`, `ALTERNATIVE`, `PREFERENCE`, `OTHER`

---

## Document Metadata Models (A08)

### DocumentMetadata

Complete metadata container for a processed document.

```python
class DocumentMetadata(BaseModel):
    file_metadata: FileMetadata             # File system properties (size, dates)
    pdf_metadata: PDFMetadata               # PDF properties (title, author, DOI, page count)
    classification: DocumentClassification  # Document type with confidence
    description: DocumentDescription        # LLM-generated title and descriptions
    date_extraction: DateExtractionResult   # Dates with fallback chain
    provenance: DocumentMetadataProvenance
```

### Key Sub-Models

- **DocumentType**: Classification result with `code`, `name`, `confidence` (e.g., DLA, CRM, CSR)
- **DocumentClassification**: Primary type + alternative types ranked by confidence
- **DocumentDescription**: LLM-generated `title`, `short_description`, `long_description`
- **DateExtractionResult**: All extracted dates with `DateSourceType` fallback chain (FILENAME, CONTENT, PDF_METADATA, FILE_SYSTEM)

---

## Pharma Models (A09)

### PharmaCandidate / ExtractedPharma

```python
class PharmaCandidate(BaseModel):
    id: uuid.UUID
    doc_id: str
    matched_text: str
    canonical_name: str
    full_name: Optional[str]
    headquarters: Optional[str]
    parent_company: Optional[str]
    subsidiaries: list[str]
    generator_type: PharmaGeneratorType    # LEXICON_MATCH
    provenance: PharmaProvenanceMetadata
```

---

## Exceptions (A12)

All pipeline exceptions inherit from `ESEPipelineError`:

| Exception | Parent | Purpose |
|-----------|--------|---------|
| `ESEPipelineError` | `Exception` | Base for all pipeline errors |
| `ConfigurationError` | `ESEPipelineError` | Invalid config (config_key, expected_type, actual_value) |
| `ParsingError` | `ESEPipelineError` | PDF parsing failures (file_path, page_number) |
| `ExtractionError` | `ESEPipelineError` | NER/regex/lexicon failures (extractor_name, entity_type) |
| `EnrichmentError` | `ESEPipelineError` | Database lookup failures (enricher_name, entity_id) |
| `APIError` | `EnrichmentError` | HTTP failures (status_code, response_body, endpoint) |
| `RateLimitError` | `APIError` | HTTP 429 rate limit (retry_after) |
| `ValidationError` | `ESEPipelineError` | Entity validation failures (entity_id, field_name) |
| `CacheError` | `ESEPipelineError` | Cache read/write/corruption failures (cache_key, operation) |
| `EvaluationError` | `ESEPipelineError` | Scoring and gold standard failures (metric_name) |

---

## Domain Profile (A15)

Configurable tuning profiles for domain-specific confidence adjustments.

```python
@dataclass
class DomainProfile:
    name: str
    priority_diseases: list[str]
    priority_journals: list[str]
    noise_terms: list[str]
    physiological_systems: list[str]
    generic_terms: list[str]
    adjustments: ConfidenceAdjustments
```

Built-in profiles: `"generic"`, `"nephrology"`, `"oncology"`, `"pulmonology"`.

`ConfidenceAdjustments` provides tunable penalty/boost values (e.g., `generic_disease_term`, `short_match_no_context`, `priority_disease_boost`).

---

## Unicode Utilities (A20)

PDF-aware text normalization functions for handling encoding artifacts.

- `normalize_sf()` -- NFKC normalization, mojibake fix, hyphen normalization
- `normalize_sf_key()` -- Uppercase key normalization for dictionary lookups
- `normalize_context()` -- Lowercase normalization for context matching
- `clean_long_form()` -- Repair PDF extraction artifacts (line-break hyphenation, truncation)
- `is_truncated_term()` -- Detect PDF truncation patterns
- `MOJIBAKE_MAP` -- Common PDF encoding issues (Greek letters, ligatures)

---

## Clinical Criteria (A21)

Computable clinical criteria models for eligibility analysis.

### LabCriterion

```python
class LabCriterion(BaseModel):
    analyte: str                    # "eGFR", "hemoglobin"
    operator: str                   # ">=", "<", "between"
    value: Optional[float]
    unit: Optional[str]
    min_value: Optional[float]      # For "between" ranges
    max_value: Optional[float]
    specimen: Optional[str]
    timepoints: list[LabTimepoint]
    normalization: Optional[EntityNormalization]  # LOINC, SNOMED codes
```

### SeverityGrade

Normalized clinical severity grades.

```python
class SeverityGrade(BaseModel):
    grade_type: SeverityGradeType   # NYHA, ECOG, CKD, CHILD_PUGH, MELD, etc.
    raw_value: str
    numeric_value: Optional[float]
    min_value: Optional[float]
    max_value: Optional[float]
    operator: Optional[str]
```

---

## Logical Expressions (A22)

AND/OR/NOT tree structures for eligibility criteria composition.

```python
class CriterionNode(BaseModel):
    criterion_id: Optional[str]
    operator: Optional[LogicalOperator]  # AND, OR, NOT
    children: list[CriterionNode]
    raw_text: Optional[str]
    confidence: float
```

```python
class LogicalExpression(BaseModel):
    root: CriterionNode
    raw_text: str
    criteria_refs: dict[str, str]   # criterion_id -> raw_text
```

Supports `evaluate()` for programmatic evaluation and `to_sql_where()` for SQL generation.

---

## Model Relationships

```
                    ProvenanceMetadata
                          |
               +----------+----------+
               |                     |
          Candidate            ExtractedEntity
          (pre-LLM)            (post-LLM)
               |                     |
               +--------> EvidenceSpan <--------+
                              |
                          Coordinate
                              |
                          BoundingBox

    DiseaseCandidate -----> ExtractedDisease
         |                       |
    DiseaseIdentifier       DiseaseIdentifier
    (ICD-10, SNOMED,        (enriched via PubTator3)
     MONDO, ORPHA)

    DrugCandidate --------> ExtractedDrug
         |                       |
    DrugIdentifier          DrugIdentifier
    (RxCUI, MeSH,           (enriched)
     DrugBank)

    GeneCandidate --------> ExtractedGene
         |                       |
    GeneIdentifier          GeneIdentifier
    GeneDiseaseLinkage      (enriched via Orphadata)

    FeasibilityCandidate
         |
         +---> EligibilityCriterion ---> LabCriterion
         +---> ScreeningFlow ---------> ScreenFailReason
         +---> StudyDesign -----------> TreatmentArm
         +---> OperationalBurden -----> InvasiveProcedure, VisitSchedule
         +---> StudyFootprint

    VisualCandidate ------> ExtractedVisual
         |                       |
    CaptionCandidate        TableStructure
    TriageResult            VisualRelationships

    CarePathway
         |
         +---> CarePathwayNode
         +---> CarePathwayEdge
         +---> TaperSchedule ---> TaperSchedulePoint

    RecommendationSet
         |
         +---> GuidelineRecommendation ---> DrugDosingInfo
```

---

## Related Documentation

- [Pipeline Overview](01_overview.md) -- layer philosophy and directory structure
- [Data Flow](02_data_flow.md) -- entity lifecycle and per-type flows

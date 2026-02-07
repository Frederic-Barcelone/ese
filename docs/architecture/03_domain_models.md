# Domain Models

All domain models in `corpus_metadata/A_core/` use Pydantic v2 with strict validation (`extra="forbid"`, frozen where appropriate).

---

## Core Abbreviation Models (A01)

### Candidate

Pre-verification abbreviation candidate (high recall, noisy).

```python
class Candidate(BaseModel):
    id: uuid.UUID
    doc_id: str
    field_type: FieldType
    generator_type: GeneratorType
    short_form: str
    long_form: Optional[str]
    context_text: str
    context_location: Coordinate
    initial_confidence: float           # [0.0, 1.0]
    provenance: ProvenanceMetadata
```

### ExtractedEntity

Post-verification abbreviation, suitable for export.

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
    mention_count: int
    pages_mentioned: list[int]
    status: ValidationStatus
    confidence_score: float
    rejection_reason: Optional[str]
    validation_flags: list[str]
    category: Optional[AbbreviationCategory]
    provenance: ProvenanceMetadata
    raw_llm_response: Optional[dict | str]
```

### Key Enums

**FieldType** -- How the abbreviation appeared in the document:

| Value | Example |
|-------|---------|
| `DEFINITION_PAIR` | "Tumor Necrosis Factor (TNF)" |
| `GLOSSARY_ENTRY` | "AE \| Adverse Event" |
| `SHORT_FORM_ONLY` | "The patient received TNF..." |

**GeneratorType** -- Which strategy produced the candidate:

| Value | Module |
|-------|--------|
| `gen:syntax_pattern` | C01 |
| `gen:glossary_table` | C01 |
| `gen:rigid_pattern` | C02 |
| `gen:table_layout` | C03 |
| `gen:lexicon_match` | C04 |
| `gen:inline_definition` | C04 |

**ValidationStatus**: `VALIDATED`, `REJECTED`, `AMBIGUOUS`

**AbbreviationCategory**: `ABBREV`, `STATISTICAL`, `DISEASE`, `DRUG`, `GENE`, `STUDY`, `ORGANIZATION`, `UNKNOWN`

---

## Supporting Types (A01)

### BoundingBox

```python
class BoundingBox(BaseModel):      # frozen=True
    coords: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    page_width: Optional[float]
    page_height: Optional[float]
    is_normalized: bool
```

### Coordinate

Page number is 1-based.

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

```python
class EvidenceSpan(BaseModel):     # frozen=True
    text: str
    location: Coordinate
    scope_ref: str
    start_char_offset: int
    end_char_offset: int
```

### ProvenanceMetadata

```python
class ProvenanceMetadata(BaseModel):  # frozen=True
    pipeline_version: str           # Git commit hash
    run_id: str
    doc_fingerprint: str            # SHA256 of source PDF
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

```python
class DocumentGraph(BaseModel):
    doc_id: str
    pages: Dict[int, Page]
    metadata: Dict[str, str]

    def get_page(self, page_num: int) -> Page: ...
    def iter_images(self) -> Iterator[ImageBlock]: ...
    def iter_linear_blocks(self, skip_header_footer: bool = True) -> Iterator[TextBlock]: ...
    def iter_tables(self, table_type: Optional[TableType] = None) -> Iterator[Table]: ...
```

### Page / TextBlock

```python
class Page(BaseModel):
    number: int                     # 1-based
    width: float
    height: float
    blocks: list[TextBlock]
    tables: list[Table]
    images: list[ImageBlock]

class TextBlock(BaseModel):        # frozen=True
    id: str
    text: str
    page_num: int
    reading_order_index: int
    role: ContentRole               # BODY_TEXT, SECTION_HEADER, PAGE_HEADER, etc.
    bbox: BoundingBox
```

---

## Disease Models (A05)

```python
class DiseaseCandidate(BaseModel):
    id: uuid.UUID
    doc_id: str
    matched_text: str
    preferred_label: str
    abbreviation: Optional[str]
    synonyms: list[str]
    field_type: DiseaseFieldType
    generator_type: DiseaseGeneratorType
    identifiers: list[DiseaseIdentifier]
    context_text: str
    context_location: Coordinate
    is_rare_disease: bool
    prevalence: Optional[str]
    disease_category: Optional[str]
    provenance: DiseaseProvenanceMetadata

class ExtractedDisease(BaseModel):
    matched_text: str
    preferred_label: str
    identifiers: list[DiseaseIdentifier]
    icd10_code: Optional[str]
    icd11_code: Optional[str]
    snomed_code: Optional[str]
    mondo_id: Optional[str]
    orpha_code: Optional[str]
    umls_cui: Optional[str]
    mesh_id: Optional[str]
    status: ValidationStatus
    confidence_score: float
    is_rare_disease: bool
    mesh_aliases: list[str]
    pubtator_normalized_name: Optional[str]
    provenance: DiseaseProvenanceMetadata

class DiseaseIdentifier(BaseModel):  # frozen=True
    system: str     # "ICD-10", "SNOMED-CT", "MONDO", "ORPHA", "MeSH", "UMLS"
    code: str
    display: Optional[str]
```

---

## Drug Models (A06)

```python
class DrugCandidate(BaseModel):
    matched_text: str
    preferred_name: str
    brand_name: Optional[str]
    compound_id: Optional[str]
    identifiers: list[DrugIdentifier]
    drug_class: Optional[str]
    development_phase: Optional[str]
    is_investigational: bool
    sponsor: Optional[str]
    conditions: list[str]
    provenance: DrugProvenanceMetadata

class ExtractedDrug(BaseModel):
    matched_text: str
    preferred_name: str
    rxcui: Optional[str]
    mesh_id: Optional[str]
    drugbank_id: Optional[str]
    unii: Optional[str]
    development_phase: Optional[str]
    is_investigational: bool
    provenance: DrugProvenanceMetadata
```

**DevelopmentPhase**: `Preclinical`, `Phase 1`, `Phase 2`, `Phase 3`, `Approved`, `Withdrawn`, `Unknown`

---

## Gene Models (A19)

```python
class GeneCandidate(BaseModel):
    matched_text: str
    hgnc_symbol: str
    full_name: Optional[str]
    is_alias: bool
    alias_of: Optional[str]
    identifiers: list[GeneIdentifier]
    locus_type: Optional[str]
    chromosome: Optional[str]
    associated_diseases: list[GeneDiseaseLinkage]
    provenance: GeneProvenanceMetadata

class ExtractedGene(BaseModel):
    hgnc_symbol: str
    identifiers: list[GeneIdentifier]
    hgnc_id: Optional[str]
    entrez_id: Optional[str]
    ensembl_id: Optional[str]
    omim_id: Optional[str]
    uniprot_id: Optional[str]
    associated_diseases: list[GeneDiseaseLinkage]
    provenance: GeneProvenanceMetadata

class GeneDiseaseLinkage(BaseModel):  # frozen=True
    orphacode: str
    disease_name: str
    association_type: Optional[str]
    association_status: Optional[str]
```

**GeneAssociationType**: `DISEASE_CAUSING`, `DISEASE_CAUSING_SOMATIC`, `MAJOR_SUSCEPTIBILITY`, `MODIFYING`, `ROLE_PATHOGENESIS`, `CANDIDATE`, `BIOMARKER`, `UNKNOWN`

---

## Feasibility Models (A07)

```python
class FeasibilityCandidate(BaseModel):
    id: uuid.UUID
    doc_id: str
    field_type: FeasibilityFieldType
    matched_text: str
    context_text: str
    evidence: list[EvidenceSpan]
    extraction_method: Optional[ExtractionMethod]  # REGEX, TABLE, LLM, HYBRID
    # One populated based on field_type:
    eligibility_criterion: Optional[EligibilityCriterion]
    screening_flow: Optional[ScreeningFlow]
    study_design: Optional[StudyDesign]
    operational_burden: Optional[OperationalBurden]
    epidemiology_data: Optional[EpidemiologyData]
    study_footprint: Optional[StudyFootprint]
    confidence: float
    provenance: FeasibilityProvenanceMetadata
```

Key sub-models: `EligibilityCriterion` (with `LabCriterion`, severity grades, logical expressions), `ScreeningFlow` (CONSORT metrics), `StudyDesign` (phase, blinding, arms), `OperationalBurden`, `StudyFootprint`, `TrialIdentifier`.

---

## Other Entity Models

### Author (A10)
```python
class AuthorCandidate(BaseModel):
    full_name: str
    role: AuthorRoleType  # AUTHOR, PI, CO_INVESTIGATOR, CORRESPONDING_AUTHOR, etc.
    affiliation: Optional[str]
    email: Optional[str]
    orcid: Optional[str]
```

### Citation (A11)
```python
class CitationCandidate(BaseModel):
    pmid: Optional[str]
    pmcid: Optional[str]
    doi: Optional[str]
    nct: Optional[str]
    url: Optional[str]
    citation_text: str
    citation_number: Optional[int]
    identifier_types: list[CitationIdentifierType]  # PMID, PMCID, DOI, NCT, URL
```

### Visual (A13)
```python
class ExtractedVisual(BaseModel):
    visual_id: str
    visual_type: VisualType         # TABLE, FIGURE, OTHER
    confidence: float
    page_range: list[int]
    caption_text: Optional[str]
    image_base64: str
    docling_table: Optional[TableStructure]
    validated_table: Optional[TableStructure]
    triage_decision: Optional[TriageDecision]  # SKIP, CHEAP_PATH, VLM_REQUIRED
    vlm_title: Optional[str]
    vlm_description: Optional[str]
```

### Pharma (A09)
```python
class PharmaCandidate(BaseModel):
    matched_text: str
    canonical_name: str
    full_name: Optional[str]
    headquarters: Optional[str]
    parent_company: Optional[str]
    subsidiaries: list[str]
```

---

## Infrastructure Models

### Extraction Result (A14)

Universal output contract using `dataclass(frozen=True)`:

**EntityType**: `abbreviation`, `disease`, `drug`, `gene`, `pharma_company`, `author`, `citation`, `feasibility`, `metadata`

### Pipeline Metrics (A16)

```python
class PipelineMetrics(BaseModel):
    run_id: str
    doc_id: str
    generation: GenerationMetrics
    heuristics: HeuristicsMetrics
    validation: ValidationMetrics
    normalization: NormalizationMetrics
    export: ExportMetrics
    scoring: Optional[ScoringMetrics]
```

Includes `validate_invariants()` for cross-stage consistency checks.

### Care Pathway (A17)

```python
class CarePathway(BaseModel):
    pathway_id: str
    title: str
    condition: str
    phases: list[str]
    nodes: list[CarePathwayNode]    # START, END, ACTION, DECISION, ASSESSMENT, etc.
    edges: list[CarePathwayEdge]
    primary_drugs: list[str]
    decision_points: list[str]
```

### Guideline Recommendation (A18)

```python
class GuidelineRecommendation(BaseModel):
    recommendation_type: RecommendationType  # TREATMENT, DOSING, MONITORING, etc.
    population: str
    action: str
    evidence_level: EvidenceLevel    # HIGH, MODERATE, LOW, VERY_LOW, EXPERT_OPINION
    strength: RecommendationStrength # STRONG, CONDITIONAL, WEAK
    dosing: list[DrugDosingInfo]
```

### Exceptions (A12)

All inherit from `ESEPipelineError`: `ConfigurationError`, `ParsingError`, `ExtractionError`, `EnrichmentError`, `APIError`, `RateLimitError`, `ValidationError`, `CacheError`, `EvaluationError`.

### Domain Profile (A15)

Configurable confidence adjustments per domain. Built-in profiles: `"generic"`, `"nephrology"`, `"oncology"`, `"pulmonology"`.

### Clinical Criteria (A21) / Logical Expressions (A22)

`LabCriterion` (analyte, operator, value, unit), `SeverityGrade` (NYHA, ECOG, CKD, etc.), `LogicalExpression` with AND/OR/NOT trees for eligibility criteria. Supports `evaluate()` and `to_sql_where()`.

---

## Model Relationships

```
                    ProvenanceMetadata
                          |
               +----------+----------+
               |                     |
          Candidate            ExtractedEntity
               |                     |
               +--------> EvidenceSpan <--------+
                              |
                          Coordinate --> BoundingBox

    DiseaseCandidate -----> ExtractedDisease (with DiseaseIdentifier)
    DrugCandidate --------> ExtractedDrug (with DrugIdentifier)
    GeneCandidate --------> ExtractedGene (with GeneIdentifier, GeneDiseaseLinkage)

    FeasibilityCandidate
         +---> EligibilityCriterion ---> LabCriterion
         +---> ScreeningFlow, StudyDesign, OperationalBurden, StudyFootprint

    CarePathway ---> CarePathwayNode/Edge, TaperSchedule
    RecommendationSet ---> GuidelineRecommendation ---> DrugDosingInfo
```

---

## Related Documentation

- [Pipeline Overview](01_overview.md) -- layer philosophy and directory structure
- [Data Flow](02_data_flow.md) -- entity lifecycle and per-type flows

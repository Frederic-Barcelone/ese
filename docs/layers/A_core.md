# Layer A: Core Domain Models

## Purpose

`A_core/` defines the shared vocabulary for the entire ESE pipeline. All Pydantic models, enums, interfaces, exceptions, and utilities used across layers live here. No layer imports domain types from another -- all cross-layer contracts go through `A_core/`.

24 modules, dependency-free with respect to other pipeline layers (B through J). Depends only on Pydantic v2, stdlib, and `typing_extensions`.

> **Note**: The `A13` prefix is shared by `A13_ner_models.py` and `A13_visual_models.py` (historical artifact; independent modules).

See also: [Domain Models reference](../architecture/03_domain_models.md) | [Pipeline Overview](../architecture/01_overview.md)

---

## Key Modules

### Logging

| Module | Description |
|--------|-------------|
| `A00_logging.py` | `get_logger()`, `configure_logging()`, `timed` decorator. |

### Core Abbreviation Models

| Module | Description |
|--------|-------------|
| `A01_domain_models.py` | `Candidate`, `ExtractedEntity`, `FieldType`, `GeneratorType`, `ValidationStatus`, `AbbreviationCategory`, `EvidenceSpan`, `ProvenanceMetadata`, `BoundingBox`, `Coordinate`. |

### Interfaces

| Module | Description |
|--------|-------------|
| `A02_interfaces.py` | Abstract base classes for all pipeline components. See [Public Interfaces](#public-interfaces). |

### Provenance and Heuristics

| Module | Description |
|--------|-------------|
| `A03_provenance.py` | `generate_run_id()`, `compute_doc_fingerprint()`, `hash_bytes()`, `compute_prompt_hash()`. |
| `A04_heuristics_config.py` | PASO heuristic rules from `config.yaml`: `HeuristicsConfig`, `HeuristicsCounters`. |

### Entity-Specific Models

| Module | Description |
|--------|-------------|
| `A05_disease_models.py` | `DiseaseCandidate`, `ExtractedDisease`, `DiseaseIdentifier` (ICD-10, SNOMED, MONDO, ORPHA). |
| `A06_drug_models.py` | `DrugCandidate`, `ExtractedDrug`, `DrugIdentifier` (RxCUI, MeSH, DrugBank). |
| `A07_feasibility_models.py` | `FeasibilityCandidate`, `NERCandidate`, `EligibilityCriterion`, `ScreeningFlow`, `StudyDesign`, `TrialIdentifier`, `OperationalBurden`, `StudyFootprint`. |
| `A08_document_metadata_models.py` | `DocumentMetadata`, `DocumentClassification`, `DateExtractionResult`, `PDFMetadata`. |
| `A09_pharma_models.py` | `PharmaCandidate`, `ExtractedPharma`. |
| `A10_author_models.py` | `AuthorCandidate`, `ExtractedAuthor`, `AuthorRoleType`. |
| `A11_citation_models.py` | `CitationCandidate`, `ExtractedCitation`, `CitationIdentifierType`. |
| `A19_gene_models.py` | `GeneCandidate`, `ExtractedGene`, `GeneIdentifier` (HGNC, Entrez, Ensembl), `GeneDiseaseLinkage`. |

### Infrastructure Models

| Module | Description |
|--------|-------------|
| `A12_exceptions.py` | `ESEPipelineError`, `ConfigurationError`, `ParsingError`, `APIError`, `RateLimitError`, `ValidationError`. |
| `A13_ner_models.py` | `NEREntity`, `NERResult` -- unified NER output format. |
| `A13_visual_models.py` | `ExtractedVisual`, `VisualType`, `TableStructure`, `TriageResult`. |
| `A14_extraction_result.py` | `EntityType` enum, `Provenance` dataclass (frozen), deterministic ID generation. |
| `A15_domain_profile.py` | `DomainProfile`, `ConfidenceAdjustments` for domain-specific tuning. |
| `A16_pipeline_metrics.py` | `PipelineMetrics` with `validate_invariants()`. |

### Specialized Models

| Module | Description |
|--------|-------------|
| `A17_care_pathway_models.py` | `CarePathwayNode`, `CarePathwayEdge`, `CarePathway`, `TaperSchedule`. |
| `A18_recommendation_models.py` | `GuidelineRecommendation`, `RecommendationSet`, `EvidenceLevel`, `RecommendationStrength`. |
| `A20_unicode_utils.py` | `normalize_sf()`, `normalize_context()`, mojibake mapping, ligature handling. |
| `A21_clinical_criteria.py` | `LabCriterion`, `SeverityGrade` with `evaluate()` for eligibility logic. |
| `A22_logical_expressions.py` | `LogicalExpression`, `CriterionNode` (AND, OR, NOT) for eligibility criteria trees. |

---

## Public Interfaces

Defined in `A02_interfaces.py`.

### ExecutionContext

Shared context passed through extraction steps.

```python
@dataclass
class ExecutionContext:
    plan_id: str
    doc_id: str
    doc_fingerprint: str
    outputs_by_step: Dict[str, List[RawExtraction]]
    def get_prior_outputs(self, strategy_id: str) -> List[RawExtraction]: ...
```

### RawExtraction

Strategy output with features (confidence computed separately by `UnifiedConfidenceCalculator` in `B06_confidence.py`).

```python
@dataclass
class RawExtraction:
    doc_id: str
    entity_type: EntityType
    field_name: str
    value: str
    page_num: int
    strategy_id: str
    lexicon_matched: bool = False
    externally_validated: bool = False
    pattern_strength: float = 0.0
    negated: bool = False
```

### Key Interfaces

```python
class BaseExtractor(ABC):
    def extract(self, doc_graph, ctx: ExecutionContext, config: Dict) -> List[RawExtraction]: ...

class BaseCandidateGenerator(ABC):
    def extract(self, doc_structure: DocumentModel) -> List[Candidate]: ...

class BaseVerifier(ABC):
    def verify(self, candidate: Candidate) -> ExtractedEntity: ...

class BaseNormalizer(ABC):
    def normalize(self, entity: ExtractedEntity) -> ExtractedEntity: ...

class BaseEnricher(ABC, Generic[InputT, OutputT]):
    def enrich(self, entity: InputT) -> OutputT: ...
    def enrich_batch(self, entities: List[InputT]) -> List[OutputT]: ...

class BaseEvaluationMetric(ABC):
    def compute(self, predictions, ground_truth) -> Dict[str, float]: ...
```

---

## Usage Patterns

### Creating a New Entity Model

1. Define `*Candidate` (pre-verification) and `*Extracted*` (post-verification) models.
2. Define `*Identifier` if the entity has standard codes.
3. Use `frozen=True` where immutability matters.
4. Include provenance on every model.
5. Register in `A14_extraction_result.EntityType`.

### Model Design Rules

- All Pydantic models use `ConfigDict(arbitrary_types_allowed=True)`.
- Provenance tracking mandatory on every entity.
- No optional fields without explicit defaults.
- Type progression: `Candidate -> ExtractedEntity`.

### Provenance Tracking

Every entity carries `ProvenanceMetadata`: `pipeline_version` (git hash), `run_id`, `doc_fingerprint` (SHA-256), `generator_name`, `prompt_bundle_hash`, `timestamp`.

# Layer A: Core Domain Models

## Purpose

`A_core/` defines the shared vocabulary for the entire ESE pipeline. Every Pydantic model, enum, interface, exception, and utility used across layers lives here. No layer imports domain types from another layer; all cross-layer contracts are mediated through `A_core/`.

This layer contains 24 modules and is strictly dependency-free with respect to other pipeline layers (B through J). It depends only on Pydantic v2, the standard library, and `typing_extensions`.

> **Note**: The `A13` prefix is shared by two modules: `A13_ner_models.py` (NER output format) and `A13_visual_models.py` (figure/table models). This is a historical artifact — both modules are independent and do not conflict.

See also: [Domain Models reference](../architecture/03_domain_models.md) | [Pipeline Overview](../architecture/01_overview.md)

---

## Key Modules

### Logging

| Module | Description |
|--------|-------------|
| `A00_logging.py` | Centralized colored logging. Provides `get_logger()`, `configure_logging()`, and `timed` decorator for performance measurement. |

### Core Abbreviation Models

| Module | Description |
|--------|-------------|
| `A01_domain_models.py` | Foundational models shared by abbreviation extraction: `Candidate` (pre-verification), `ExtractedEntity` (post-verification), `FieldType`, `GeneratorType`, `ValidationStatus`, `AbbreviationCategory`, `EvidenceSpan`, `ProvenanceMetadata`, `BoundingBox`, `Coordinate`. |

### Interfaces

| Module | Description |
|--------|-------------|
| `A02_interfaces.py` | Abstract base classes defining contracts for all pipeline components. See [Public Interfaces](#public-interfaces) below. |

### Provenance and Heuristics

| Module | Description |
|--------|-------------|
| `A03_provenance.py` | Deterministic hashing: `generate_run_id()`, `compute_doc_fingerprint()`, `hash_bytes()`, `compute_prompt_hash()`. Every pipeline run is uniquely identified for reproducibility. |
| `A04_heuristics_config.py` | PASO heuristic rules loaded from `config.yaml`: `HeuristicsConfig` (statistical abbreviation whitelist, short-form blacklist, context rules), `HeuristicsCounters` for tracking rule application. |

### Entity-Specific Models

| Module | Description |
|--------|-------------|
| `A05_disease_models.py` | `DiseaseCandidate`, `ExtractedDisease`, `DiseaseIdentifier` (ICD-10, SNOMED, MONDO, ORPHA). |
| `A06_drug_models.py` | `DrugCandidate`, `ExtractedDrug`, `DrugIdentifier` (RxCUI, MeSH, DrugBank), `DevelopmentPhase` enum. |
| `A07_feasibility_models.py` | `FeasibilityCandidate`, `EligibilityCriterion`, `ScreeningFlow`, `StudyDesign`, `TrialIdentifier`, `OperationalBurden`, `StudyFootprint`. |
| `A08_document_metadata_models.py` | `DocumentMetadata`, `DocumentClassification`, `DateExtractionResult`, `PDFMetadata`. |
| `A09_pharma_models.py` | `PharmaCandidate`, `ExtractedPharma` for pharmaceutical company entities. |
| `A10_author_models.py` | `AuthorCandidate`, `ExtractedAuthor`, `AuthorRoleType` (PI, corresponding, steering committee). |
| `A11_citation_models.py` | `CitationCandidate`, `ExtractedCitation`, `CitationIdentifierType` (PMID, DOI, NCT, URL). |
| `A19_gene_models.py` | `GeneCandidate`, `ExtractedGene`, `GeneIdentifier` (HGNC, Entrez, Ensembl), `GeneDiseaseLinkage`. |

### Infrastructure Models

| Module | Description |
|--------|-------------|
| `A12_exceptions.py` | Exception hierarchy: `ESEPipelineError`, `ConfigurationError`, `ParsingError`, `APIError`, `RateLimitError`, `ValidationError`. |
| `A13_ner_models.py` | `NEREntity`, `NERResult` -- unified output format from multiple NER backends (scispacy, biomedical-ner-all, ZeroShotBioNER). |
| `A13_visual_models.py` | `ExtractedVisual`, `VisualType`, `TableStructure`, `TriageResult` for figure/table pipeline. |
| `A14_extraction_result.py` | `EntityType` enum, `Provenance` dataclass (frozen), deterministic ID generation. Universal output contract. |
| `A15_domain_profile.py` | `DomainProfile`, `ConfidenceAdjustments`, `load_domain_profile()` for domain-specific confidence tuning. |
| `A16_pipeline_metrics.py` | `PipelineMetrics`, `GenerationMetrics`, `HeuristicsMetrics`, `ValidationMetrics`, `ExportMetrics` with `validate_invariants()`. |

### Specialized Models

| Module | Description |
|--------|-------------|
| `A17_care_pathway_models.py` | `CarePathwayNode`, `CarePathwayEdge`, `CarePathway`, `TaperSchedule` for clinical treatment algorithm extraction. |
| `A18_recommendation_models.py` | `GuidelineRecommendation`, `RecommendationSet`, `EvidenceLevel`, `RecommendationType`, `RecommendationStrength`. |
| `A20_unicode_utils.py` | `normalize_sf()`, `normalize_context()`, mojibake mapping, ligature handling for consistent text normalization. |
| `A21_clinical_criteria.py` | `LabCriterion`, `DiagnosisConfirmation`, `SeverityGrade` with `evaluate()` methods for eligibility logic. |
| `A22_logical_expressions.py` | `LogicalExpression`, `CriterionNode`, `LogicalOperator` (AND, OR, NOT) for structured eligibility criteria trees. |

---

## Public Interfaces

All pipeline components implement abstract base classes defined in `A02_interfaces.py`.

### ExecutionContext

Shared context passed through all extraction steps. Strategies can read prior outputs but must never gate execution on them.

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

Strategy output with features (confidence computed separately by `UnifiedConfidenceCalculator`). Strategies must not set confidence directly; they emit features instead.

```python
@dataclass
class RawExtraction:
    doc_id: str
    entity_type: EntityType
    field_name: str
    value: str
    page_num: int
    strategy_id: str
    # Features for confidence calculation
    lexicon_matched: bool = False
    externally_validated: bool = False
    pattern_strength: float = 0.0
    negated: bool = False
    # ...
```

### BaseExtractor

Universal interface for all extraction strategies.

```python
class BaseExtractor(ABC):
    @property
    def strategy_id(self) -> str: ...
    @property
    def strategy_version(self) -> str: ...
    @property
    def entity_type(self) -> EntityType: ...

    def extract(self, doc_graph, ctx: ExecutionContext, config: Dict) -> List[RawExtraction]: ...
```

### BaseCandidateGenerator

Interface for high-recall candidate generation strategies.

```python
class BaseCandidateGenerator(ABC):
    @property
    def generator_type(self) -> GeneratorType: ...

    def extract(self, doc_structure: DocumentModel) -> List[Candidate]: ...
```

### BaseVerifier

Interface for LLM or deterministic verification.

```python
class BaseVerifier(ABC):
    def verify(self, candidate: Candidate) -> ExtractedEntity: ...
```

### BaseNormalizer

Interface for post-verification standardization.

```python
class BaseNormalizer(ABC):
    def normalize(self, entity: ExtractedEntity) -> ExtractedEntity: ...
```

### BaseEnricher[InputT, OutputT]

Generic interface for entity enrichment (PubTator, NCT, genetic databases).

```python
class BaseEnricher(ABC, Generic[InputT, OutputT]):
    def __init__(self, config: Optional[Dict] = None): ...

    @property
    def enricher_name(self) -> str: ...

    def enrich(self, entity: InputT) -> OutputT: ...
    def enrich_batch(self, entities: List[InputT]) -> List[OutputT]: ...
```

### BaseEvaluationMetric

Interface for precision/recall metric computation.

```python
class BaseEvaluationMetric(ABC):
    def compute(self, predictions: List[ExtractedEntity], ground_truth: List[Dict]) -> Dict[str, float]: ...
```

---

## Usage Patterns

### Creating a New Entity Model

1. Define a `*Candidate` model (pre-verification, high-recall fields).
2. Define an `*Extracted*` model (post-verification, enriched fields).
3. Define an `*Identifier` model if the entity has standard codes.
4. Use `ConfigDict(arbitrary_types_allowed=True)` and `frozen=True` where immutability matters.
5. Include `ProvenanceMetadata` or a domain-specific provenance variant on every model.
6. Register the entity type in `A14_extraction_result.EntityType`.

### Model Design Rules

- All Pydantic models use `ConfigDict(arbitrary_types_allowed=True)`.
- Provenance tracking is mandatory on every entity.
- Use immutable tuples for collections (`tuple[str, ...]` not `list[str]`).
- Use enum-based classification for deterministic logic.
- No optional fields without explicit default values.
- Strong type hierarchy: `Candidate -> ExtractedEntity` progression.

### Provenance Tracking

Every entity carries a `ProvenanceMetadata` recording:
- `pipeline_version` (git commit hash)
- `run_id` (deterministic, timestamp-based)
- `doc_fingerprint` (SHA-256 of source PDF)
- `generator_name` (which strategy produced the candidate)
- `prompt_bundle_hash` (for LLM-based extraction)
- `timestamp`

This enables full audit trail from extracted entity back to source PDF, extraction strategy, and LLM prompt version.

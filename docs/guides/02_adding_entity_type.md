# Adding a New Entity Type

This guide walks through the process of adding a new entity type to the ESE pipeline. Each step references the established patterns in the codebase. Disease extraction is used as a concrete example throughout.

## Overview

Adding a new entity type touches multiple layers of the pipeline:

```
A_core/ (models) -> C_generators/ (detection) -> E_normalization/ (enrichment)
    -> H_pipeline/ (factory) -> I_extraction/ (processing) -> J_export/ (output)
```

## Step 1: Create Domain Model in A_core/

Create `A_core/AXX_entity_models.py` with Pydantic models for the new entity type.

**Required models:**

1. **Identifier model** -- represents codes from standard ontologies.
2. **Provenance model** -- extends `BaseProvenanceMetadata` for audit trails.
3. **Candidate model** -- pre-validation entity with context and detection metadata.
4. **Extracted model** -- validated entity with all codes and evidence.
5. **Export models** -- simplified structures for JSON output.

**Example from `A05_disease_models.py`:**

```python
from pydantic import BaseModel, ConfigDict, Field, model_validator
from A_core.A01_domain_models import (
    BaseProvenanceMetadata, Coordinate, EvidenceSpan, ValidationStatus,
)

class DiseaseIdentifier(BaseModel):
    """Medical code from a standard ontology."""
    system: str   # "ICD-10", "SNOMED-CT", "MONDO", "ORPHA"
    code: str     # "I27.0", "MONDO_0011055"
    display: Optional[str] = None
    model_config = ConfigDict(frozen=True, extra="forbid")

class DiseaseCandidate(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    doc_id: str
    matched_text: str
    preferred_label: str
    identifiers: List[DiseaseIdentifier] = Field(default_factory=list)
    field_type: DiseaseFieldType
    generator_type: DiseaseGeneratorType
    context_text: str
    context_location: Coordinate
    is_rare_disease: bool = False
    initial_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    provenance: DiseaseProvenanceMetadata
    model_config = ConfigDict(extra="forbid")

class ExtractedDisease(BaseModel):
    candidate_id: uuid.UUID
    doc_id: str
    matched_text: str
    preferred_label: str
    identifiers: List[DiseaseIdentifier] = Field(default_factory=list)
    primary_evidence: EvidenceSpan
    status: ValidationStatus
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    provenance: DiseaseProvenanceMetadata
    model_config = ConfigDict(extra="forbid")
```

**Key patterns to follow:**

- Use `ConfigDict(extra="forbid")` to catch typos in field names.
- Include `model_validator(mode="after")` for non-empty field checks.
- Add provenance fields and confidence scores on all entities.
- No optional fields without explicit default values.
- Reference existing models: `A05_disease_models.py`, `A06_drug_models.py`, `A19_gene_models.py`.

## Step 2: Create Generator in C_generators/

Create `C_generators/CXX_strategy_entity.py` implementing the detection logic.

**Interface:** Generators implement one of two interfaces from `A_core/A02_interfaces.py` (see [A_core interfaces](../layers/A_core.md#public-interfaces) for full signatures):

- **`BaseCandidateGenerator`** -- For abbreviation-style strategies. Implement `generator_type` property and `extract(doc_structure) -> List[Candidate]`.
- **`BaseExtractor`** -- For entity-specific strategies (recommended for new types). Implement `strategy_id`, `entity_type` properties and `extract(doc_graph, ctx, config) -> List[RawExtraction]`.

**Design principles for generators:**

- Optimize for **high recall** -- false positives are acceptable at this stage.
- Use **FlashText** for lexicon matching (not regex for large vocabularies). The pipeline loads 600K+ terms via FlashText for fast O(n) matching.
- Consider multiple detection layers: lexicon matching + NER + regex patterns.
- Iterate over the document using `doc_graph.iter_linear_blocks()` or `doc_graph.iter_tables()`.

**Example from disease detection (`C06_strategy_disease.py`):**

The `DiseaseDetector` loads multiple lexicons (general diseases, Orphanet rare diseases, specialized lexicons for PAH/ANCA/IgAN) and uses FlashText keyword processors to scan document text blocks. It also integrates scispacy NER for additional coverage.

**Optionally add a false positive filter** in `CXX_entity_fp_filter.py`:

- Confidence-based adjustments (not hard filtering for most cases).
- Hard filtering only for catastrophic false positives (e.g., chromosome patterns in chromosome context).
- See `C24_disease_fp_filter.py` and `C25_drug_fp_filter.py` for patterns.

## Step 3: Add Enricher in E_normalization/ (if needed)

Create `E_normalization/EXX_entity_enricher.py` implementing the `BaseEnricher[InputT, OutputT]` interface from `A_core/A02_interfaces.py` (see [A_core interfaces](../layers/A_core.md#public-interfaces) for full signature). Key methods to implement:

- `enricher_name` property -- unique identifier for logging/metrics
- `enrich(entity: InputT) -> OutputT` -- enrich a single entity
- Optionally override `enrich_batch()` for optimized batch processing

**Common enrichment patterns:**

- **PubTator3 API** (`E04_pubtator_enricher.py`): Adds MeSH codes, aliases, and normalized names for diseases.
- **NCT/ClinicalTrials.gov** (`E06_nct_enricher.py`): Enriches trial identifiers with official metadata.
- **Disk caching**: Use `Z_utils/Z01_api_client.py` for caching API responses. Configure TTL in `config.yaml` under the relevant `cache` section.
- **Deduplication**: Implement deduplication logic based on canonical identifiers, not string matching. See `E17_entity_deduplicator.py`.
- Handle API failures gracefully -- return the original entity unmodified on error.

## Step 4: Register in H_pipeline/H01_component_factory.py

Add a creation method to `ComponentFactory`:

```python
def create_entity_detector(self) -> "EntityDetector":
    """Create entity detection component."""
    from C_generators.CXX_strategy_entity import EntityDetector
    return EntityDetector(
        config={
            "run_id": self.run_id,
            "lexicon_base_path": str(self.dict_path),
        }
    )
```

If you added an enricher, also add:

```python
def create_entity_enricher(self) -> Optional["EntityEnricher"]:
    """Create entity enricher if enabled."""
    if not self.use_pubtator_enrichment:
        return None
    from E_normalization.EXX_entity_enricher import EntityEnricher
    return EntityEnricher(config={"run_id": self.run_id})
```

Add the appropriate TYPE_CHECKING import at the top of the file.

## Step 5: Add Processing Method to I_extraction/I01_entity_processors.py

Add the new entity type to `EntityProcessor`:

1. Add the detector and optional enricher as constructor parameters.
2. Add a `process_entities()` method that coordinates: detection -> FP filtering -> validation -> enrichment -> deduplication.

**Pattern from disease processing:**

```python
def process_diseases(self, doc: "DocumentGraph", pdf_path: Path) -> List["ExtractedDisease"]:
    if self.disease_detector is None:
        return []

    # 1. Generate candidates (high recall)
    candidates = self.disease_detector.extract(doc)

    # 2. Convert to extracted entities with validation status
    results = []
    for candidate in candidates:
        entity = self._create_disease_entity(candidate, ...)
        results.append(entity)

    # 3. Normalize (optional)
    if self.disease_normalizer is not None:
        results = self.disease_normalizer.normalize_batch(results)

    # 4. Enrich (optional, e.g., PubTator)
    if self.disease_enricher is not None:
        results = self.disease_enricher.enrich_batch(results, verbose=True)

    # 5. Deduplicate
    deduplicator = EntityDeduplicator()
    results = deduplicator.deduplicate_diseases(results)

    return results
```

## Step 6: Add Export Handler in J_export/

Add an export function in `J01a_entity_exporters.py` (for entity types) or `J01b_metadata_exporters.py` (for metadata types).

**Pattern:**

```python
def export_entity_results(
    out_dir: Path,
    pdf_path: Path,
    results: List["ExtractedEntity"],
    run_id: str,
    pipeline_version: str,
) -> None:
    """Export entity results to JSON file."""
    # Convert to export models
    # Write to: {entity_type}_{doc_name}_{timestamp}.json
```

Follow the existing serialization patterns in `J01a_entity_exporters.py` using the export models defined in Step 1 (e.g., `DiseaseExportEntry`, `DiseaseExportDocument`).

## Step 7: Wire Up in orchestrator.py

1. Create the detector and enricher via `ComponentFactory` in the orchestrator initialization.
2. Pass them to `EntityProcessor`.
3. Add the extraction call in the main processing loop.
4. Add the export call to write results.
5. Add the entity type to the relevant extraction presets in `config.yaml`.
6. Add a toggle under `extraction_pipeline.extractors` in `config.yaml`:

```yaml
extraction_pipeline:
  extractors:
    your_entity: true
```

## Step 8: Add Tests in K_tests/

Write tests covering:

- **Model validation**: Ensure Pydantic models accept valid data and reject invalid data.
- **Generator detection**: Test with sample document text to verify candidates are produced.
- **FP filter**: Test that known false positives are filtered or downweighted.
- **Enricher**: Test with mocked API responses.
- **Deduplication**: Test that duplicate entities are merged correctly.
- **Export**: Test that JSON output matches the expected schema.

Run the full test suite:

```bash
cd corpus_metadata && python -m pytest K_tests/ -v
```

## Reference Files

| Step | Example Files |
|------|--------------|
| Domain models | `A_core/A05_disease_models.py`, `A_core/A06_drug_models.py`, `A_core/A19_gene_models.py` |
| Generators | `C_generators/C06_strategy_disease.py`, `C_generators/C07_strategy_drug.py`, `C_generators/C16_strategy_gene.py` |
| FP filters | `C_generators/C24_disease_fp_filter.py`, `C_generators/C25_drug_fp_filter.py`, `C_generators/C34_gene_fp_filter.py` |
| Enrichers | `E_normalization/E04_pubtator_enricher.py`, `E_normalization/E05_drug_enricher.py`, `E_normalization/E18_gene_enricher.py` |
| Deduplication | `E_normalization/E17_entity_deduplicator.py` |
| Factory | `H_pipeline/H01_component_factory.py` |
| Processors | `I_extraction/I01_entity_processors.py` |
| Exporters | `J_export/J01a_entity_exporters.py`, `J_export/J01b_metadata_exporters.py` |

# A_core Refactoring Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor A_core folder to improve DRY, reduce file sizes, and enhance maintainability.

**Architecture:** Split large files (A04, A07) into focused modules. Extract common provenance pattern into a generic base class in A01. Fix minor documentation issues.

**Tech Stack:** Python 3.12+, Pydantic v2, dataclasses

---

## Task 1: Create BaseProvenanceMetadata Generic Class

**Files:**
- Modify: `corpus_metadata/A_core/A01_domain_models.py:217-243`

**Step 1: Write a test for the generic provenance base class**

Create test file:

```python
# corpus_metadata/tests/test_a01_base_provenance.py
"""Tests for BaseProvenanceMetadata generic class."""
import pytest
from datetime import datetime
from enum import Enum

from A_core.A01_domain_models import (
    BaseProvenanceMetadata,
    GeneratorType,
    LLMParameters,
)


class TestGeneratorType(str, Enum):
    """Test generator type for testing."""
    TEST_GEN = "gen:test"


def test_base_provenance_with_generator_type():
    """Test that BaseProvenanceMetadata works with any generator enum."""
    prov = BaseProvenanceMetadata(
        pipeline_version="abc123",
        run_id="RUN_20250202_120000_test",
        doc_fingerprint="sha256hash",
        generator_name=GeneratorType.SYNTAX_PATTERN,
    )
    assert prov.pipeline_version == "abc123"
    assert prov.generator_name == GeneratorType.SYNTAX_PATTERN
    assert prov.timestamp is not None


def test_base_provenance_with_llm_config():
    """Test provenance with LLM config."""
    llm = LLMParameters(
        model_name="claude-sonnet-4-20250514",
        temperature=0.0,
        max_tokens=1024,
        top_p=1.0,
    )
    prov = BaseProvenanceMetadata(
        pipeline_version="abc123",
        run_id="RUN_20250202",
        doc_fingerprint="sha256",
        generator_name=GeneratorType.LEXICON_MATCH,
        llm_config=llm,
    )
    assert prov.llm_config.model_name == "claude-sonnet-4-20250514"


def test_base_provenance_is_frozen():
    """Test that provenance metadata is immutable."""
    prov = BaseProvenanceMetadata(
        pipeline_version="abc123",
        run_id="RUN_20250202",
        doc_fingerprint="sha256",
        generator_name=GeneratorType.SYNTAX_PATTERN,
    )
    with pytest.raises(Exception):  # ValidationError for frozen model
        prov.pipeline_version = "changed"
```

**Step 2: Run test to verify it fails**

```bash
cd corpus_metadata && python -m pytest tests/test_a01_base_provenance.py -v
```

Expected: FAIL (BaseProvenanceMetadata not found)

**Step 3: Implement BaseProvenanceMetadata**

Add after `LLMParameters` class (around line 206) in `A01_domain_models.py`:

```python
class BaseProvenanceMetadata(BaseModel):
    """
    Base provenance metadata for all entity types.

    Provides common audit trail fields. Entity-specific provenance classes
    should inherit from this and specify their own generator_name type.

    The generator_name field accepts any Enum value, allowing each entity type
    to use its own GeneratorType enum while sharing the common structure.
    """

    pipeline_version: str  # Git commit hash
    run_id: str  # e.g., RUN_20250101_120000_ab12cd34ef56
    doc_fingerprint: str  # SHA256 of source PDF bytes

    generator_name: Enum  # Entity-specific generator type (accepts any Enum)
    rule_version: Optional[str] = None

    # Lexicon provenance
    lexicon_source: Optional[str] = None  # e.g., "disease_lexicon_pah.json"

    # Populated during verification
    prompt_bundle_hash: Optional[str] = None
    context_hash: Optional[str] = None
    llm_config: Optional[LLMParameters] = None

    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(frozen=True, extra="forbid")
```

**Step 4: Run test to verify it passes**

```bash
cd corpus_metadata && python -m pytest tests/test_a01_base_provenance.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add corpus_metadata/A_core/A01_domain_models.py corpus_metadata/tests/test_a01_base_provenance.py
git commit -m "$(cat <<'EOF'
feat(A_core): add BaseProvenanceMetadata generic class

Provides common audit trail structure for all entity types.
Accepts any Enum as generator_name for flexibility.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Migrate Disease Provenance to Use Base Class

**Files:**
- Modify: `corpus_metadata/A_core/A05_disease_models.py:70-95`

**Step 1: Write migration test**

```python
# corpus_metadata/tests/test_a05_disease_provenance_migration.py
"""Tests for DiseaseProvenanceMetadata inheriting from base."""
import pytest
from datetime import datetime

from A_core.A05_disease_models import (
    DiseaseProvenanceMetadata,
    DiseaseGeneratorType,
    DiseaseIdentifier,
)
from A_core.A01_domain_models import BaseProvenanceMetadata, LLMParameters


def test_disease_provenance_has_base_fields():
    """Verify DiseaseProvenanceMetadata has all base fields."""
    prov = DiseaseProvenanceMetadata(
        pipeline_version="abc123",
        run_id="RUN_20250202",
        doc_fingerprint="sha256",
        generator_name=DiseaseGeneratorType.LEXICON_ORPHANET,
    )
    # Base fields exist
    assert prov.pipeline_version == "abc123"
    assert prov.run_id == "RUN_20250202"
    assert prov.doc_fingerprint == "sha256"
    assert prov.timestamp is not None


def test_disease_provenance_with_lexicon_ids():
    """Test disease-specific lexicon_ids field."""
    ids = [DiseaseIdentifier(system="MONDO", code="MONDO_0011055")]
    prov = DiseaseProvenanceMetadata(
        pipeline_version="abc123",
        run_id="RUN_20250202",
        doc_fingerprint="sha256",
        generator_name=DiseaseGeneratorType.LEXICON_ORPHANET,
        lexicon_ids=ids,
    )
    assert len(prov.lexicon_ids) == 1
    assert prov.lexicon_ids[0].system == "MONDO"


def test_disease_provenance_is_frozen():
    """Test immutability is preserved."""
    prov = DiseaseProvenanceMetadata(
        pipeline_version="abc123",
        run_id="RUN_20250202",
        doc_fingerprint="sha256",
        generator_name=DiseaseGeneratorType.SCISPACY_NER,
    )
    with pytest.raises(Exception):
        prov.run_id = "changed"
```

**Step 2: Run test (should pass with current code)**

```bash
cd corpus_metadata && python -m pytest tests/test_a05_disease_provenance_migration.py -v
```

Expected: PASS (validates current behavior before migration)

**Step 3: Simplify DiseaseProvenanceMetadata**

Replace `DiseaseProvenanceMetadata` class in A05 with:

```python
class DiseaseProvenanceMetadata(BaseProvenanceMetadata):
    """
    Provenance metadata for disease detection.

    Extends BaseProvenanceMetadata with disease-specific lexicon_ids type.
    """

    generator_name: DiseaseGeneratorType  # Override with specific type
    lexicon_ids: Optional[List[DiseaseIdentifier]] = None  # Disease-specific codes

    model_config = ConfigDict(frozen=True, extra="forbid")
```

Update import to include `BaseProvenanceMetadata`:

```python
from A_core.A01_domain_models import (
    BaseProvenanceMetadata,
    Coordinate,
    EvidenceSpan,
    LLMParameters,
    ValidationStatus,
)
```

**Step 4: Run tests**

```bash
cd corpus_metadata && python -m pytest tests/test_a05_disease_provenance_migration.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add corpus_metadata/A_core/A05_disease_models.py corpus_metadata/tests/test_a05_disease_provenance_migration.py
git commit -m "$(cat <<'EOF'
refactor(A05): migrate DiseaseProvenanceMetadata to inherit from base

Reduces ~20 lines of duplication by inheriting common provenance fields.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Migrate Drug Provenance to Use Base Class

**Files:**
- Modify: `corpus_metadata/A_core/A06_drug_models.py:85-109`

**Step 1: Simplify DrugProvenanceMetadata**

Replace class with:

```python
class DrugProvenanceMetadata(BaseProvenanceMetadata):
    """Provenance metadata for drug detection."""

    generator_name: DrugGeneratorType  # Override with specific type
    lexicon_ids: Optional[List[DrugIdentifier]] = None  # Drug-specific codes

    model_config = ConfigDict(frozen=True, extra="forbid")
```

Update import:

```python
from A_core.A01_domain_models import (
    BaseProvenanceMetadata,
    Coordinate,
    EvidenceSpan,
    LLMParameters,
    ValidationStatus,
)
```

**Step 2: Run existing tests**

```bash
cd corpus_metadata && python -m pytest tests/ -k drug -v
```

Expected: PASS (or no drug-specific tests)

**Step 3: Commit**

```bash
git add corpus_metadata/A_core/A06_drug_models.py
git commit -m "$(cat <<'EOF'
refactor(A06): migrate DrugProvenanceMetadata to inherit from base

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Migrate Gene Provenance to Use Base Class

**Files:**
- Modify: `corpus_metadata/A_core/A19_gene_models.py:99-121`

**Step 1: Simplify GeneProvenanceMetadata**

Replace class with:

```python
class GeneProvenanceMetadata(BaseProvenanceMetadata):
    """Provenance metadata for gene detection."""

    generator_name: GeneGeneratorType  # Override with specific type
    lexicon_ids: Optional[List[GeneIdentifier]] = None  # Gene-specific codes

    model_config = ConfigDict(frozen=True, extra="forbid")
```

Update import:

```python
from A_core.A01_domain_models import (
    BaseProvenanceMetadata,
    Coordinate,
    EvidenceSpan,
    LLMParameters,
    ValidationStatus,
)
```

**Step 2: Commit**

```bash
git add corpus_metadata/A_core/A19_gene_models.py
git commit -m "$(cat <<'EOF'
refactor(A19): migrate GeneProvenanceMetadata to inherit from base

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Migrate Remaining Provenance Classes (Pharma, Author, Citation)

**Files:**
- Modify: `corpus_metadata/A_core/A09_pharma_models.py:36-47`
- Modify: `corpus_metadata/A_core/A10_author_models.py:53-63`
- Modify: `corpus_metadata/A_core/A11_citation_models.py:48-58`

**Step 1: Simplify PharmaProvenanceMetadata**

```python
from A_core.A01_domain_models import (
    BaseProvenanceMetadata,
    Coordinate,
    EvidenceSpan,
    ValidationStatus,
)

class PharmaProvenanceMetadata(BaseProvenanceMetadata):
    """Provenance metadata for pharma company detection."""

    generator_name: PharmaGeneratorType

    model_config = ConfigDict(frozen=True, extra="forbid")
```

**Step 2: Simplify AuthorProvenanceMetadata**

```python
from A_core.A01_domain_models import (
    BaseProvenanceMetadata,
    Coordinate,
    EvidenceSpan,
    ValidationStatus,
)

class AuthorProvenanceMetadata(BaseProvenanceMetadata):
    """Provenance metadata for author detection."""

    generator_name: AuthorGeneratorType

    model_config = ConfigDict(frozen=True, extra="forbid")
```

**Step 3: Simplify CitationProvenanceMetadata**

```python
from A_core.A01_domain_models import (
    BaseProvenanceMetadata,
    Coordinate,
    EvidenceSpan,
    ValidationStatus,
)

class CitationProvenanceMetadata(BaseProvenanceMetadata):
    """Provenance metadata for citation detection."""

    generator_name: CitationGeneratorType

    model_config = ConfigDict(frozen=True, extra="forbid")
```

**Step 4: Run all tests**

```bash
cd corpus_metadata && python -m pytest tests/ -v
```

**Step 5: Commit**

```bash
git add corpus_metadata/A_core/A09_pharma_models.py corpus_metadata/A_core/A10_author_models.py corpus_metadata/A_core/A11_citation_models.py
git commit -m "$(cat <<'EOF'
refactor(A_core): migrate Pharma/Author/Citation provenance to base class

Completes DRY refactoring of provenance metadata across all entity types.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Split A04_heuristics_config.py - Extract Unicode Utilities

**Files:**
- Create: `corpus_metadata/A_core/A04a_unicode_utils.py`
- Modify: `corpus_metadata/A_core/A04_heuristics_config.py`

**Step 1: Create A04a_unicode_utils.py**

Move lines 27-178 from A04_heuristics_config.py:

```python
# corpus_metadata/A_core/A04a_unicode_utils.py
"""
Unicode normalization utilities for PDF text extraction.

Handles mojibake, ligatures, variant hyphens, and truncation detection.
"""

import re
import unicodedata

# Various Unicode hyphen/dash characters to normalize
HYPHENS_PATTERN = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2212\u00ad]")

# Common mojibake substitutions (e.g., from PDF extraction)
MOJIBAKE_MAP = {
    "Î±": "α",  # Greek alpha (common PDF encoding issue)
    "Î²": "β",  # Greek beta
    "Î³": "γ",  # Greek gamma
    "Î´": "δ",  # Greek delta
    "ﬁ": "fi",  # fi ligature
    "ﬂ": "fl",  # fl ligature
    "ﬀ": "ff",  # ff ligature
    "ﬃ": "ffi",  # ffi ligature
    "ﬄ": "ffl",  # ffl ligature
}


def normalize_sf(sf: str) -> str:
    """
    Normalize a short form for display/storage.

    - Applies NFKC Unicode normalization
    - Fixes common mojibake issues
    - Normalizes hyphens to standard ASCII hyphen
    - Collapses whitespace
    """
    s = unicodedata.normalize("NFKC", sf).strip()
    # Fix mojibake
    for bad, good in MOJIBAKE_MAP.items():
        s = s.replace(bad, good)
    # Normalize hyphens
    s = HYPHENS_PATTERN.sub("-", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_sf_key(sf: str) -> str:
    """
    Normalize a short form for use as dictionary/set key (comparison).

    Returns uppercase normalized form for consistent matching.
    """
    return normalize_sf(sf).upper()


def normalize_context(ctx: str) -> str:
    """
    Normalize context text for matching.

    - Applies NFKC Unicode normalization
    - Normalizes hyphens
    - Returns lowercase for case-insensitive matching
    """
    c = unicodedata.normalize("NFKC", ctx)
    c = HYPHENS_PATTERN.sub("-", c)
    return c.lower()


def clean_long_form(lf: str) -> str:
    """
    Clean up long form text extracted from PDFs.

    Fixes common PDF parsing artifacts:
    - Line-break hyphenation: "gastro-\\nintestinal" -> "gastrointestinal"
    - Truncation detection: returns empty string if truncated
    - Extra whitespace: collapses multiple spaces
    - Mojibake: fixes common encoding issues

    Args:
        lf: Raw long form string from PDF extraction

    Returns:
        Cleaned long form, or empty string if invalid/truncated
    """
    if not lf:
        return ""

    # Apply NFKC normalization
    lf = unicodedata.normalize("NFKC", lf).strip()

    # Fix mojibake
    for bad, good in MOJIBAKE_MAP.items():
        lf = lf.replace(bad, good)

    # Fix line-break hyphenation: "gastro-\\nintestinal" -> "gastrointestinal"
    lf = re.sub(r"-\s*\n\s*([a-z])", r"\1", lf)
    lf = re.sub(r"-\s+([a-z])", r"\1", lf)

    # Normalize all hyphens to standard ASCII
    lf = HYPHENS_PATTERN.sub("-", lf)

    # Collapse whitespace
    lf = re.sub(r"\s+", " ", lf).strip()

    # Detect truncation
    words = lf.split()
    if words:
        last_word = words[-1].lower()
        if len(last_word) >= 3:
            truncation_endings = [
                "stin", "culi", "liti", "niti", "rati", "mati",
                "gica", "logi", "path", "neur", "chem", "phar",
            ]
            for ending in truncation_endings:
                if last_word.endswith(ending) and len(last_word) < 8:
                    return ""

    return lf


def is_truncated_term(term: str) -> bool:
    """
    Check if a term appears to be truncated from PDF extraction.

    Returns True if term looks incomplete (likely PDF artifact).
    """
    if not term or len(term) < 4:
        return False

    term_lower = term.lower().strip()

    incomplete_patterns = [
        r"vasculit?i?$",
        r"glomeru?l?o?$",
        r"nephropa?t?h?$",
        r"encepha?l?o?$",
        r"myopa?t?h?$",
        r"neuropa?t?h?$",
        r"cardio?m?y?o?$",
        r"throm?b?o?$",
        r"pancre?a?t?$",
        r"hepat?i?t?$",
    ]

    for pattern in incomplete_patterns:
        if re.search(pattern, term_lower):
            return True

    return False
```

**Step 2: Update A04_heuristics_config.py imports**

Replace the Unicode section (lines 27-178) with:

```python
# Unicode utilities moved to A04a_unicode_utils.py for maintainability
from A_core.A04a_unicode_utils import (
    HYPHENS_PATTERN,
    MOJIBAKE_MAP,
    normalize_sf,
    normalize_sf_key,
    normalize_context,
    clean_long_form,
    is_truncated_term,
)
```

**Step 3: Run tests**

```bash
cd corpus_metadata && python -m pytest tests/ -v
```

**Step 4: Commit**

```bash
git add corpus_metadata/A_core/A04a_unicode_utils.py corpus_metadata/A_core/A04_heuristics_config.py
git commit -m "$(cat <<'EOF'
refactor(A04): extract Unicode utilities to A04a_unicode_utils.py

Reduces A04_heuristics_config.py by ~150 lines.
Improves testability of Unicode normalization functions.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Split A07_feasibility_models.py - Extract Clinical Criteria

**Files:**
- Create: `corpus_metadata/A_core/A07a_clinical_criteria.py`
- Modify: `corpus_metadata/A_core/A07_feasibility_models.py`

**Step 1: Create A07a_clinical_criteria.py**

Move clinical criteria models (LabCriterion, DiagnosisConfirmation, SeverityGrade):

```python
# corpus_metadata/A_core/A07a_clinical_criteria.py
"""
Clinical criteria models for eligibility computability.

Contains:
- Lab criteria with evaluation logic
- Diagnosis confirmation requirements
- Severity grade normalization (NYHA, ECOG, CKD, etc.)
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class LabTimepoint(BaseModel):
    """Structured timepoint for lab requirements."""

    day: Optional[int] = None  # relative to randomization
    visit_name: Optional[str] = None  # "screening", "baseline"
    window_days: Optional[int] = None  # +/- days

    model_config = ConfigDict(extra="forbid")


class EntityNormalization(BaseModel):
    """Normalized coding for drugs, conditions, labs."""

    system: str  # "LOINC", "RxNorm", "ATC", "Orphanet", "SNOMED", "ICD-10"
    code: str
    label: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class LabCriterion(BaseModel):
    """Fully computable lab eligibility criterion."""

    analyte: str  # "UPCR", "eGFR", "C3"
    operator: str  # ">=", "<=", ">", "<", "==", "between"
    value: float
    unit: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    specimen: Optional[str] = None
    timepoints: List[LabTimepoint] = Field(default_factory=list)
    normalization: Optional[EntityNormalization] = None

    model_config = ConfigDict(extra="forbid")

    def evaluate(self, actual_value: float) -> bool:
        """Evaluate if actual_value satisfies this criterion."""
        if self.operator == ">=":
            return actual_value >= self.value
        elif self.operator == "<=":
            return actual_value <= self.value
        elif self.operator == ">":
            return actual_value > self.value
        elif self.operator == "<":
            return actual_value < self.value
        elif self.operator == "==":
            return actual_value == self.value
        elif self.operator == "between":
            if self.min_value is not None and self.max_value is not None:
                return self.min_value <= actual_value <= self.max_value
        return False


class DiagnosisConfirmation(BaseModel):
    """Structured diagnosis confirmation requirement."""

    method: str  # "biopsy", "genetic_testing", "clinical_criteria"
    window_months: Optional[int] = None
    assessor: Optional[str] = None
    findings: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class SeverityGradeType(str, Enum):
    """Standard clinical severity grading systems."""

    NYHA = "nyha"
    ECOG = "ecog"
    CKD = "ckd"
    CHILD_PUGH = "child_pugh"
    MELD = "meld"
    BCLC = "bclc"
    TNM = "tnm"
    EDSS = "edss"


class SeverityGrade(BaseModel):
    """Normalized severity grade for clinical scales."""

    grade_type: SeverityGradeType
    raw_value: str
    numeric_value: Optional[int] = None
    min_value: Optional[int] = None
    max_value: Optional[int] = None
    operator: Optional[str] = None

    model_config = ConfigDict(extra="forbid")

    def evaluate(self, actual_grade: int) -> bool:
        """Evaluate if actual_grade satisfies this criterion."""
        if self.operator == "<=":
            return actual_grade <= (self.numeric_value or self.max_value or 0)
        elif self.operator == ">=":
            return actual_grade >= (self.numeric_value or self.min_value or 0)
        elif self.operator == "==":
            return actual_grade == self.numeric_value
        elif self.operator == "between" and self.min_value is not None and self.max_value is not None:
            return self.min_value <= actual_grade <= self.max_value
        return False


# Severity grade normalization mappings
SEVERITY_GRADE_MAPPINGS = {
    SeverityGradeType.NYHA: {
        "i": 1, "1": 1, "class i": 1, "class 1": 1,
        "ii": 2, "2": 2, "class ii": 2, "class 2": 2,
        "iii": 3, "3": 3, "class iii": 3, "class 3": 3,
        "iv": 4, "4": 4, "class iv": 4, "class 4": 4,
    },
    SeverityGradeType.ECOG: {
        "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
    },
    SeverityGradeType.CKD: {
        "1": 1, "2": 2, "3": 3, "3a": 3, "3b": 3, "4": 4, "5": 5,
        "stage 1": 1, "stage 2": 2, "stage 3": 3, "stage 3a": 3,
        "stage 3b": 3, "stage 4": 4, "stage 5": 5,
    },
    SeverityGradeType.CHILD_PUGH: {
        "a": 1, "b": 2, "c": 3,
        "class a": 1, "class b": 2, "class c": 3,
        "5": 1, "6": 1, "7": 2, "8": 2, "9": 2, "10": 3,
        "11": 3, "12": 3, "13": 3, "14": 3, "15": 3,
    },
}
```

**Step 2: Update A07_feasibility_models.py imports**

Replace clinical criteria definitions with imports:

```python
# Clinical criteria models moved to A07a_clinical_criteria.py
from A_core.A07a_clinical_criteria import (
    LabTimepoint,
    EntityNormalization,
    LabCriterion,
    DiagnosisConfirmation,
    SeverityGradeType,
    SeverityGrade,
    SEVERITY_GRADE_MAPPINGS,
)
```

Remove lines 141-286 (LabTimepoint through SEVERITY_GRADE_MAPPINGS).

**Step 3: Run tests**

```bash
cd corpus_metadata && python -m pytest tests/ -v
```

**Step 4: Commit**

```bash
git add corpus_metadata/A_core/A07a_clinical_criteria.py corpus_metadata/A_core/A07_feasibility_models.py
git commit -m "$(cat <<'EOF'
refactor(A07): extract clinical criteria to A07a_clinical_criteria.py

Reduces A07_feasibility_models.py by ~150 lines.
Clinical criteria (LabCriterion, SeverityGrade) now in focused module.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Split A07_feasibility_models.py - Extract Logical Expressions

**Files:**
- Create: `corpus_metadata/A_core/A07b_logical_expressions.py`
- Modify: `corpus_metadata/A_core/A07_feasibility_models.py`

**Step 1: Create A07b_logical_expressions.py**

Move logical expression models (LogicalOperator, CriterionNode, LogicalExpression):

```python
# corpus_metadata/A_core/A07b_logical_expressions.py
"""
Logical expression models for eligibility criteria computability.

Supports structured representation of eligibility criteria with AND/OR/NOT logic:
"(Age >= 18 AND Age <= 75) AND (eGFR >= 30 OR on_dialysis)"
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from A_core.A07_feasibility_models import EligibilityCriterion


class LogicalOperator(str, Enum):
    """Logical operators for combining eligibility criteria."""

    AND = "AND"
    OR = "OR"
    NOT = "NOT"


class CriterionNode(BaseModel):
    """
    Node in a logical expression tree for eligibility criteria.

    Can be:
    - A leaf node containing a single criterion
    - An internal node with an operator and children
    """

    # For leaf nodes
    criterion: Optional[Any] = None  # EligibilityCriterion (forward ref)
    criterion_id: Optional[str] = None

    # For internal nodes
    operator: Optional[LogicalOperator] = None
    children: List["CriterionNode"] = Field(default_factory=list)

    # Metadata
    raw_text: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    model_config = ConfigDict(extra="forbid")

    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return self.criterion is not None or self.criterion_id is not None

    def evaluate(self, criterion_values: Dict[str, bool]) -> bool:
        """
        Evaluate the logical expression given criterion truth values.

        Args:
            criterion_values: Dict mapping criterion_id to True/False

        Returns:
            Boolean result of evaluating the expression
        """
        if self.is_leaf():
            if self.criterion_id:
                return criterion_values.get(self.criterion_id, False)
            return False

        if self.operator == LogicalOperator.AND:
            return all(child.evaluate(criterion_values) for child in self.children)
        elif self.operator == LogicalOperator.OR:
            return any(child.evaluate(criterion_values) for child in self.children)
        elif self.operator == LogicalOperator.NOT:
            if self.children:
                return not self.children[0].evaluate(criterion_values)
        return False


class LogicalExpression(BaseModel):
    """
    Complete logical expression for eligibility criteria.

    Represents structured eligibility like:
    "(Age >= 18 AND Age <= 75) AND (eGFR >= 30 OR on_dialysis)"
    """

    root: CriterionNode
    raw_text: str
    criteria_refs: Dict[str, Any] = Field(default_factory=dict)  # EligibilityCriterion

    model_config = ConfigDict(extra="forbid")

    def evaluate(self, patient_data: Dict[str, Any]) -> bool:
        """
        Evaluate eligibility for a patient.

        Args:
            patient_data: Dict with patient values (age, eGFR, etc.)

        Returns:
            True if patient meets eligibility criteria
        """
        criterion_values = {}
        for crit_id, criterion in self.criteria_refs.items():
            criterion_values[crit_id] = self._evaluate_criterion(criterion, patient_data)
        return self.root.evaluate(criterion_values)

    def _evaluate_criterion(self, criterion: Any, patient_data: Dict[str, Any]) -> bool:
        """Evaluate a single criterion against patient data."""
        if hasattr(criterion, 'lab_criterion') and criterion.lab_criterion:
            lab = criterion.lab_criterion
            actual = patient_data.get(lab.analyte.lower())
            if actual is not None:
                result = lab.evaluate(actual)
                return not result if criterion.is_negated else result
        return False

    def to_sql_where(self) -> str:
        """Convert expression to SQL WHERE clause (for EHR queries)."""
        return self._node_to_sql(self.root)

    def _node_to_sql(self, node: CriterionNode) -> str:
        """Convert a node to SQL."""
        if node.is_leaf():
            crit = self.criteria_refs.get(node.criterion_id) if node.criterion_id else None
            if crit and hasattr(crit, 'lab_criterion') and crit.lab_criterion:
                lab = crit.lab_criterion
                col = lab.analyte.lower()
                if lab.operator == "between":
                    return f"({col} BETWEEN {lab.min_value} AND {lab.max_value})"
                return f"({col} {lab.operator} {lab.value})"
            return "TRUE"

        if node.operator == LogicalOperator.AND:
            parts = [self._node_to_sql(c) for c in node.children]
            return f"({' AND '.join(parts)})"
        elif node.operator == LogicalOperator.OR:
            parts = [self._node_to_sql(c) for c in node.children]
            return f"({' OR '.join(parts)})"
        elif node.operator == LogicalOperator.NOT:
            if node.children:
                return f"(NOT {self._node_to_sql(node.children[0])})"
        return "TRUE"


# Rebuild forward references
CriterionNode.model_rebuild()
LogicalExpression.model_rebuild()
```

**Step 2: Update A07_feasibility_models.py imports**

Replace logical expression definitions with imports:

```python
# Logical expression models moved to A07b_logical_expressions.py
from A_core.A07b_logical_expressions import (
    LogicalOperator,
    CriterionNode,
    LogicalExpression,
)
```

Remove lines 289-424 (LogicalOperator through LogicalExpression) and lines 456-458 (model_rebuild calls).

**Step 3: Run tests**

```bash
cd corpus_metadata && python -m pytest tests/ -v
```

**Step 4: Commit**

```bash
git add corpus_metadata/A_core/A07b_logical_expressions.py corpus_metadata/A_core/A07_feasibility_models.py
git commit -m "$(cat <<'EOF'
refactor(A07): extract logical expressions to A07b_logical_expressions.py

Reduces A07_feasibility_models.py by ~140 lines.
Logical expression tree (AND/OR/NOT) for eligibility now in focused module.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Fix Header Comments

**Files:**
- Modify: `corpus_metadata/A_core/A04_heuristics_config.py:1`
- Modify: `corpus_metadata/A_core/A05_disease_models.py:1`
- Modify: `corpus_metadata/A_core/A06_drug_models.py:1`
- Modify: `corpus_metadata/A_core/A08_document_metadata_models.py:1`

**Step 1: Fix duplicate path in headers**

Change `# corpus_metadata/corpus_metadata/A_core/` to `# corpus_metadata/A_core/` in each file.

**Step 2: Commit**

```bash
git add corpus_metadata/A_core/A04_heuristics_config.py corpus_metadata/A_core/A05_disease_models.py corpus_metadata/A_core/A06_drug_models.py corpus_metadata/A_core/A08_document_metadata_models.py
git commit -m "$(cat <<'EOF'
fix(A_core): correct duplicate path in file header comments

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Final Verification

**Step 1: Run full test suite**

```bash
cd corpus_metadata && python -m pytest tests/ -v
```

**Step 2: Run type checking**

```bash
mypy corpus_metadata/A_core/
```

**Step 3: Run linting**

```bash
ruff check corpus_metadata/A_core/
```

**Step 4: Verify imports work**

```bash
cd corpus_metadata && python -c "
from A_core.A01_domain_models import BaseProvenanceMetadata
from A_core.A04a_unicode_utils import normalize_sf, normalize_sf_key
from A_core.A07a_clinical_criteria import LabCriterion, SeverityGrade
from A_core.A07b_logical_expressions import LogicalExpression, CriterionNode
from A_core.A05_disease_models import DiseaseProvenanceMetadata
from A_core.A06_drug_models import DrugProvenanceMetadata
print('All imports successful!')
"
```

---

## Summary of Changes

| File | Before | After | Reduction |
|------|--------|-------|-----------|
| A01_domain_models.py | 355 | ~385 | +30 (BaseProvenanceMetadata) |
| A04_heuristics_config.py | 1,073 | ~920 | -153 |
| A04a_unicode_utils.py | - | ~130 | new |
| A05_disease_models.py | 302 | ~285 | -17 |
| A06_drug_models.py | 336 | ~315 | -21 |
| A07_feasibility_models.py | 998 | ~710 | -288 |
| A07a_clinical_criteria.py | - | ~150 | new |
| A07b_logical_expressions.py | - | ~140 | new |
| A09_pharma_models.py | 163 | ~155 | -8 |
| A10_author_models.py | 171 | ~163 | -8 |
| A11_citation_models.py | 205 | ~197 | -8 |
| A19_gene_models.py | 310 | ~290 | -20 |

**Net reduction:** ~150 lines of duplication removed
**Maintainability:** Large files split into focused modules
**DRY:** Single BaseProvenanceMetadata replaces 7 near-identical classes

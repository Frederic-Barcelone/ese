# corpus_metadata/corpus_metadata/A_core/A15_domain_profile.py
"""
Domain Profile System for Configurable Extraction Priors.

Addresses overfitting by externalizing domain-specific assumptions:
- Disease/condition vocabularies
- Journal name patterns
- Confidence adjustments
- Noise term filters

Instead of hardcoding nephrology-specific terms, profiles allow:
1. Switching domains without code changes
2. A/B testing different configurations
3. Clear separation of generic vs domain-specific logic

Usage:
    from A_core.A15_domain_profile import DomainProfile, load_domain_profile

    profile = load_domain_profile("nephrology", config)
    adjustment = profile.get_confidence_adjustment("IgA nephropathy")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import yaml


@dataclass
class ConfidenceAdjustments:
    """Confidence score adjustments for different entity types."""

    # Global penalties (apply to all domains)
    generic_disease_term: float = -0.30  # "disease", "syndrome"
    physiological_system: float = -0.40  # "CNS", "immune system"
    short_match_no_context: float = -0.25  # Short matches without disease context
    chromosome_pattern: float = -0.50  # Chromosome patterns like "10p"
    journal_citation: float = -0.40  # Text in citation context

    # Domain-specific boosts (positive = more relevant)
    priority_disease_boost: float = 0.15  # Domain-priority diseases
    priority_journal_boost: float = 0.10  # Domain-priority journals
    domain_noise_penalty: float = -0.20  # Domain-specific noise terms


@dataclass
class DomainProfile:
    """
    Domain-specific configuration for extraction priors.

    Profiles separate generic extraction logic from domain assumptions.
    This prevents corpus overfitting by making domain coupling explicit.
    """

    name: str
    description: str = ""

    # Priority diseases: boost confidence for these
    # These are diseases of high interest in this domain
    priority_diseases: Set[str] = field(default_factory=set)

    # Priority journals: boost confidence for mentions in these
    priority_journals: Set[str] = field(default_factory=set)

    # Noise terms: penalize confidence (not hard filter)
    # Domain-specific terms that are often false positives
    noise_terms: Set[str] = field(default_factory=set)

    # Physiological systems to penalize (often confused with diseases)
    physiological_systems: Set[str] = field(default_factory=set)

    # Generic terms to penalize (too broad to be useful)
    generic_terms: Set[str] = field(default_factory=set)

    # Journal patterns that indicate citation context (penalize)
    journal_citation_patterns: Set[str] = field(default_factory=set)

    # Confidence adjustments
    adjustments: ConfidenceAdjustments = field(default_factory=ConfidenceAdjustments)

    def get_confidence_adjustment(
        self,
        matched_text: str,
        context: str = "",
        is_short_match: bool = False,
        is_citation_context: bool = False,
    ) -> float:
        """
        Calculate confidence adjustment for a match.

        Returns a value between -1.0 and +0.3 to add to base confidence.
        Negative = likely false positive, Positive = domain-relevant.
        """
        adjustment = 0.0
        matched_lower = matched_text.lower().strip()
        ctx_lower = context.lower()

        # Global penalties (truly generic, apply everywhere)
        if matched_lower in self.generic_terms:
            adjustment += self.adjustments.generic_disease_term

        if matched_lower in self.physiological_systems:
            adjustment += self.adjustments.physiological_system

        if is_short_match and not self._has_disease_context(ctx_lower):
            adjustment += self.adjustments.short_match_no_context

        if is_citation_context:
            adjustment += self.adjustments.journal_citation

        # Domain-specific adjustments
        if matched_lower in self.priority_diseases:
            adjustment += self.adjustments.priority_disease_boost

        if matched_lower in self.noise_terms:
            adjustment += self.adjustments.domain_noise_penalty

        # Check if context mentions priority journals
        if any(j in ctx_lower for j in self.priority_journals):
            adjustment += self.adjustments.priority_journal_boost

        # Clamp to reasonable range
        return max(-1.0, min(0.3, adjustment))

    def should_hard_filter(
        self,
        matched_text: str,
        context: str = "",
    ) -> tuple[bool, str]:
        """
        Determine if match should be hard-filtered (catastrophic FP).

        Most cases should use confidence adjustment, not hard filtering.
        Only filter truly catastrophic false positives.

        Returns:
            (should_filter, reason)
        """
        matched_lower = matched_text.lower().strip()
        ctx_lower = context.lower()

        # Only hard filter if:
        # 1. Very generic term AND
        # 2. No disease context at all
        if matched_lower in self.generic_terms:
            if not self._has_disease_context(ctx_lower):
                # Still allow if it's a complete phrase
                if len(matched_text.split()) <= 1:
                    return True, "generic_term_no_context"

        return False, ""

    def _has_disease_context(self, ctx_lower: str) -> bool:
        """Check if context contains disease-related keywords."""
        disease_keywords = [
            "syndrome", "disease", "disorder", "condition",
            "patient", "diagnosis", "diagnosed", "treatment",
            "therapy", "symptom", "clinical", "prognosis",
            "affected", "prevalence", "incidence", "rare",
            "orphan", "trial", "study",
        ]
        return any(kw in ctx_lower for kw in disease_keywords)

    def is_priority_disease(self, disease_name: str) -> bool:
        """Check if disease is a priority for this domain."""
        return disease_name.lower().strip() in self.priority_diseases

    def is_noise_term(self, term: str) -> bool:
        """Check if term is domain-specific noise."""
        return term.lower().strip() in self.noise_terms


# =============================================================================
# BUILT-IN PROFILES
# =============================================================================


def _create_generic_profile() -> DomainProfile:
    """Create a generic profile with minimal assumptions."""
    return DomainProfile(
        name="generic",
        description="Minimal assumptions, maximum generalization",
        # Truly generic terms (apply to all medical domains)
        generic_terms={
            "disease", "diseases", "syndrome", "syndromes",
            "disorder", "disorders", "condition", "conditions",
            "neoplasm", "neoplasms", "tumor", "tumors",
            "abnormality", "abnormalities", "anomaly", "anomalies",
            "malformation", "deficiency", "insufficiency",
        },
        physiological_systems={
            "cns", "central nervous system",
            "pns", "peripheral nervous system",
            "ans", "autonomic nervous system",
            "hpa axis", "hypothalamic-pituitary-adrenal axis",
            "immune system", "complement system",
        },
        journal_citation_patterns={
            "et al", "doi:", "pmid:", "vol.", "pp.",
        },
    )


def _create_nephrology_profile() -> DomainProfile:
    """Create nephrology-specific profile (kidney/rare disease focus)."""
    return DomainProfile(
        name="nephrology",
        description="Nephrology and rare kidney disease focus",
        # Priority diseases for nephrology
        priority_diseases={
            # IgA Nephropathy
            "iga nephropathy", "igan", "berger's disease",
            # C3 Glomerulopathy
            "c3 glomerulopathy", "c3g", "c3 glomerulonephritis",
            "dense deposit disease", "ddd",
            # ANCA-associated vasculitis
            "anca-associated vasculitis", "aav",
            "granulomatosis with polyangiitis", "gpa",
            "microscopic polyangiitis", "mpa",
            "eosinophilic granulomatosis with polyangiitis", "egpa",
            # FSGS
            "focal segmental glomerulosclerosis", "fsgs",
            # Membranous nephropathy
            "membranous nephropathy", "mn",
            # PKD
            "polycystic kidney disease", "pkd", "adpkd",
            # aHUS
            "atypical hemolytic uremic syndrome", "ahus",
            # Lupus nephritis
            "lupus nephritis", "ln",
            # General CKD
            "chronic kidney disease", "ckd",
            "end-stage renal disease", "esrd", "eskd",
        },
        # Priority journals for nephrology
        priority_journals={
            "kidney int", "kidney international",
            "j am soc nephrol", "jasn",
            "clin j am soc nephrol", "cjasn",
            "nephrol dial transplant", "ndt",
            "am j kidney dis", "ajkd",
        },
        # Noise terms specific to nephrology corpus
        noise_terms={
            "kidney diseases",  # Too generic
            "renal disease",  # Too generic
            "glomerular disease",  # Too generic
            "kidney failure",  # Status, not disease
        },
        # Generic terms (inherit from generic + additions)
        generic_terms={
            "disease", "diseases", "syndrome", "syndromes",
            "disorder", "disorders", "condition", "conditions",
            "neoplasm", "neoplasms", "tumor", "tumors",
            "abnormality", "abnormalities", "anomaly", "anomalies",
            "malformation", "deficiency", "insufficiency",
            # Nephrology-generic
            "nephritis", "glomerulonephritis",
        },
        physiological_systems={
            "cns", "central nervous system",
            "pns", "peripheral nervous system",
            "ans", "autonomic nervous system",
            "hpa axis",
            "immune system", "complement system",
            # Nephrology-specific systems
            "raas", "ras", "renin-angiotensin system",
            "renin-angiotensin-aldosterone system",
        },
        journal_citation_patterns={
            "et al", "doi:", "pmid:", "vol.", "pp.",
            # Nephrology journal abbreviations (citation context)
            "kidney int", "j am soc nephrol", "nephrol dial transplant",
        },
    )


def _create_oncology_profile() -> DomainProfile:
    """Create oncology-specific profile."""
    return DomainProfile(
        name="oncology",
        description="Oncology and cancer focus",
        priority_diseases={
            # Common cancers
            "non-small cell lung cancer", "nsclc",
            "small cell lung cancer", "sclc",
            "breast cancer",
            "colorectal cancer", "crc",
            "pancreatic cancer",
            "hepatocellular carcinoma", "hcc",
            "renal cell carcinoma", "rcc",
            "melanoma",
            "multiple myeloma", "mm",
            "acute myeloid leukemia", "aml",
            "chronic lymphocytic leukemia", "cll",
            "diffuse large b-cell lymphoma", "dlbcl",
            "glioblastoma", "gbm",
        },
        priority_journals={
            "j clin oncol", "jco",
            "lancet oncol",
            "ann oncol",
            "cancer", "cancer res",
            "clin cancer res",
        },
        noise_terms={
            "malignancy",  # Too generic
            "solid tumor",  # Too generic
        },
        generic_terms={
            "disease", "diseases", "syndrome", "syndromes",
            "disorder", "disorders", "condition", "conditions",
            "neoplasm", "neoplasms", "tumor", "tumors",
            "cancer", "cancers",  # Too generic without qualifier
            "carcinoma", "sarcoma", "lymphoma", "leukemia",  # Generic types
        },
        physiological_systems={
            "cns", "central nervous system",
            "immune system",
            "lymphatic system",
        },
        journal_citation_patterns={
            "et al", "doi:", "pmid:", "vol.", "pp.",
        },
    )


def _create_pulmonology_profile() -> DomainProfile:
    """Create pulmonology-specific profile (PAH focus)."""
    return DomainProfile(
        name="pulmonology",
        description="Pulmonology and pulmonary arterial hypertension focus",
        priority_diseases={
            # PAH
            "pulmonary arterial hypertension", "pah",
            "idiopathic pulmonary arterial hypertension", "ipah",
            "heritable pulmonary arterial hypertension", "hpah",
            "chronic thromboembolic pulmonary hypertension", "cteph",
            "pulmonary hypertension", "ph",
            # IPF
            "idiopathic pulmonary fibrosis", "ipf",
            # COPD
            "chronic obstructive pulmonary disease", "copd",
            # Asthma
            "severe asthma",
            # Cystic fibrosis
            "cystic fibrosis", "cf",
        },
        priority_journals={
            "chest",
            "am j respir crit care med",
            "eur respir j",
            "lancet respir med",
        },
        noise_terms={
            "lung disease",  # Too generic
            "respiratory disease",  # Too generic
        },
        generic_terms={
            "disease", "diseases", "syndrome", "syndromes",
            "disorder", "disorders", "condition", "conditions",
            "neoplasm", "neoplasms", "tumor", "tumors",
            "abnormality", "abnormalities",
        },
        physiological_systems={
            "cns", "central nervous system",
            "pns", "peripheral nervous system",
            "ans", "autonomic nervous system",
            "immune system",
        },
        journal_citation_patterns={
            "et al", "doi:", "pmid:", "vol.", "pp.",
        },
    )


# Profile registry
_BUILTIN_PROFILES: Dict[str, DomainProfile] = {
    "generic": _create_generic_profile(),
    "nephrology": _create_nephrology_profile(),
    "oncology": _create_oncology_profile(),
    "pulmonology": _create_pulmonology_profile(),
}


# =============================================================================
# LOADING AND CONFIGURATION
# =============================================================================


def load_domain_profile(
    profile_name: str,
    config: Optional[Dict[str, Any]] = None,
) -> DomainProfile:
    """
    Load a domain profile by name.

    Args:
        profile_name: Name of profile ("generic", "nephrology", etc.)
                     or path to custom YAML profile
        config: Optional config dict with domain_profile overrides

    Returns:
        DomainProfile instance
    """
    # Check if it's a path to a custom profile
    if profile_name.endswith(".yaml") or profile_name.endswith(".yml"):
        return _load_profile_from_yaml(Path(profile_name))

    # Get built-in profile
    profile = _BUILTIN_PROFILES.get(profile_name.lower())
    if profile is None:
        # Fall back to generic
        profile = _BUILTIN_PROFILES["generic"]

    # Apply config overrides if provided
    if config and "domain_profile" in config:
        profile = _apply_config_overrides(profile, config["domain_profile"])

    return profile


def _load_profile_from_yaml(path: Path) -> DomainProfile:
    """Load profile from YAML file."""
    if not path.exists():
        raise FileNotFoundError(f"Domain profile not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return DomainProfile(
        name=data.get("name", path.stem),
        description=data.get("description", ""),
        priority_diseases=set(data.get("priority_diseases", [])),
        priority_journals=set(data.get("priority_journals", [])),
        noise_terms=set(data.get("noise_terms", [])),
        physiological_systems=set(data.get("physiological_systems", [])),
        generic_terms=set(data.get("generic_terms", [])),
        journal_citation_patterns=set(data.get("journal_citation_patterns", [])),
        adjustments=ConfidenceAdjustments(**data.get("adjustments", {})),
    )


def _apply_config_overrides(
    profile: DomainProfile,
    overrides: Dict[str, Any],
) -> DomainProfile:
    """Apply config overrides to a profile."""
    # Create a copy with overrides
    return DomainProfile(
        name=overrides.get("name", profile.name),
        description=overrides.get("description", profile.description),
        priority_diseases=set(overrides.get("priority_diseases", profile.priority_diseases)),
        priority_journals=set(overrides.get("priority_journals", profile.priority_journals)),
        noise_terms=set(overrides.get("noise_terms", profile.noise_terms)),
        physiological_systems=set(overrides.get("physiological_systems", profile.physiological_systems)),
        generic_terms=set(overrides.get("generic_terms", profile.generic_terms)),
        journal_citation_patterns=set(overrides.get("journal_citation_patterns", profile.journal_citation_patterns)),
        adjustments=ConfidenceAdjustments(
            **{**profile.adjustments.__dict__, **overrides.get("adjustments", {})}
        ),
    )


def get_available_profiles() -> List[str]:
    """Get list of available built-in profiles."""
    return list(_BUILTIN_PROFILES.keys())

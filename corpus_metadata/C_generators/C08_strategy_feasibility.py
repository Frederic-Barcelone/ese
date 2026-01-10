# corpus_metadata/corpus_metadata/C_generators/C08_strategy_feasibility.py
"""
Clinical trial feasibility information extraction strategy.

Extracts key information needed for clinical trial feasibility assessment:
1. Eligibility criteria (inclusion/exclusion)
2. Epidemiology data (prevalence, incidence, demographics)
3. Patient journey phases (screening → treatment → follow-up)
4. Study endpoints (primary, secondary)
5. Site/country information

Uses a combination of:
- Section detection (Methods, Eligibility, Study Design)
- Pattern matching for structured criteria
- Regex for epidemiology statistics
- Keyword detection for patient journey phases
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from A_core.A03_provenance import generate_run_id, get_git_revision_hash
from A_core.A07_feasibility_models import (
    CriterionType,
    EligibilityCriterion,
    EndpointType,
    EpidemiologyData,
    FeasibilityCandidate,
    FeasibilityFieldType,
    FeasibilityGeneratorType,
    FeasibilityProvenanceMetadata,
    PatientJourneyPhase,
    PatientJourneyPhaseType,
    StudyEndpoint,
    StudySite,
)
from B_parsing.B02_doc_graph import DocumentGraph


# =============================================================================
# SECTION PATTERNS
# =============================================================================

# Patterns to identify relevant sections in clinical trial documents
SECTION_PATTERNS = {
    "eligibility": [
        r"eligibility\s*criteria",
        r"inclusion\s*(?:and\s*)?exclusion\s*criteria",
        r"patient\s*selection",
        r"study\s*population",
        r"participants",
    ],
    "methods": [
        r"methods",
        r"study\s*design",
        r"trial\s*design",
        r"materials?\s*and\s*methods",
    ],
    "epidemiology": [
        r"epidemiology",
        r"prevalence",
        r"incidence",
        r"demographics",
        r"background",
        r"introduction",
    ],
    "endpoints": [
        r"endpoints?",
        r"outcomes?",
        r"efficacy",
        r"primary\s*(?:end)?points?",
        r"secondary\s*(?:end)?points?",
    ],
    "patient_journey": [
        r"study\s*procedures?",
        r"treatment\s*period",
        r"follow[\-\s]?up",
        r"screening",
        r"study\s*visits?",
    ],
}


# =============================================================================
# ELIGIBILITY CRITERIA PATTERNS
# =============================================================================

INCLUSION_MARKERS = [
    r"inclusion\s*criteria",
    r"patients?\s*(?:were|are)\s*eligible\s*if",
    r"eligible\s*(?:patients?|subjects?|participants?)",
    r"key\s*inclusion",
    r"to\s*be\s*eligible",
]

EXCLUSION_MARKERS = [
    r"exclusion\s*criteria",
    r"patients?\s*(?:were|are)\s*excluded\s*if",
    r"ineligible\s*if",
    r"key\s*exclusion",
    r"not\s*eligible\s*if",
]

# Common criterion categories
CRITERION_CATEGORIES = {
    "age": r"(?:age|year[s]?\s*old|\d+\s*(?:to|–|-)\s*\d+\s*years?)",
    "diagnosis": r"(?:diagnosis|diagnosed|confirmed|documented)\s+(?:of|with)",
    "disease_severity": r"(?:mild|moderate|severe|advanced|early[- ]stage|late[- ]stage)",
    "prior_treatment": r"(?:prior|previous|history\s*of)\s+(?:treatment|therapy|medication)",
    "lab_value": r"(?:egfr|gfr|creatinine|hemoglobin|platelet|wbc|alt|ast|bilirubin)",
    "comorbidity": r"(?:comorbid|concurrent|concomitant|coexisting)\s+(?:disease|condition)",
    "pregnancy": r"(?:pregnan|breast[\-\s]?feed|lactat|contracept)",
    "consent": r"(?:informed\s*consent|willing\s*to\s*participate)",
}


# =============================================================================
# EPIDEMIOLOGY PATTERNS
# =============================================================================

PREVALENCE_PATTERNS = [
    # "prevalence of 1 in 10,000" or "prevalence: 1/10000"
    r"prevalence\s*(?:of|:|\s)\s*([\d.,]+\s*(?:in|per|/)\s*[\d.,]+(?:\s*(?:million|thousand|100,?000|10,?000|1,?000))?)",
    # "affects 1-2 per million"
    r"affects?\s*([\d.,]+(?:\s*[-–to]+\s*[\d.,]+)?\s*(?:per|in)\s*(?:million|[\d,]+))",
    # "rare disease with <1/1000000"
    r"(<?\s*[\d.,]+\s*/\s*[\d.,]+)",
    # "3.5% of patients"
    r"([\d.,]+\s*%)\s*(?:of\s*)?(?:patients?|population|adults?|children)",
]

INCIDENCE_PATTERNS = [
    # "incidence of 2.5 per 100,000 person-years"
    r"incidence\s*(?:of|:|\s)\s*([\d.,]+(?:\s*[-–to]+\s*[\d.,]+)?\s*(?:per|/)\s*[\d.,]+(?:\s*person[\-\s]?years?)?)",
    # "2-3 new cases per million per year"
    r"([\d.,]+(?:\s*[-–to]+\s*[\d.,]+)?\s*(?:new\s*)?cases?\s*(?:per|/)\s*(?:million|[\d,]+)(?:\s*per\s*year)?)",
]

DEMOGRAPHICS_PATTERNS = [
    # "median age 45 years" or "mean age of 52"
    r"((?:median|mean)\s*age\s*(?:of\s*)?[\d.,]+(?:\s*years?)?)",
    # "age range 18-75"
    r"(age\s*(?:range|:)\s*[\d]+\s*[-–to]+\s*[\d]+)",
    # "65% female" or "male:female ratio 1:2"
    r"([\d.,]+\s*%\s*(?:male|female|women|men))",
    r"((?:male|female)[:\s]+(?:male|female)\s*ratio\s*[\d.,]+\s*:\s*[\d.,]+)",
]


# =============================================================================
# PATIENT JOURNEY PATTERNS
# =============================================================================

JOURNEY_PHASE_KEYWORDS = {
    PatientJourneyPhaseType.SCREENING: [
        "screening period",
        "screening phase",
        "screening visit",
        "pre-treatment",
        "baseline assessment",
        "wash-out",
        "washout period",
    ],
    PatientJourneyPhaseType.RUN_IN: [
        "run-in period",
        "run in phase",
        "lead-in period",
        "stabilization period",
    ],
    PatientJourneyPhaseType.RANDOMIZATION: [
        "randomization",
        "randomised",
        "randomized",
        "random assignment",
        "treatment allocation",
    ],
    PatientJourneyPhaseType.TREATMENT: [
        "treatment period",
        "treatment phase",
        "active treatment",
        "intervention period",
        "dosing period",
        "induction phase",
        "maintenance phase",
    ],
    PatientJourneyPhaseType.FOLLOW_UP: [
        "follow-up period",
        "follow up phase",
        "post-treatment",
        "safety follow-up",
        "observation period",
        "long-term follow-up",
    ],
    PatientJourneyPhaseType.EXTENSION: [
        "extension study",
        "open-label extension",
        "OLE period",
        "continued access",
    ],
}

# Duration patterns
DURATION_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:[-–to]+\s*\d+(?:\.\d+)?\s*)?"
    r"(weeks?|months?|days?|years?)",
    re.IGNORECASE,
)


# =============================================================================
# ENDPOINT PATTERNS
# =============================================================================

ENDPOINT_PATTERNS = {
    EndpointType.PRIMARY: [
        r"primary\s*(?:end)?point[s]?\s*(?:is|was|were|:)",
        r"primary\s*(?:efficacy\s*)?(?:end)?point",
        r"main\s*(?:end)?point",
    ],
    EndpointType.SECONDARY: [
        r"secondary\s*(?:end)?point[s]?",
        r"key\s*secondary",
    ],
    EndpointType.EXPLORATORY: [
        r"exploratory\s*(?:end)?point[s]?",
        r"tertiary\s*(?:end)?point[s]?",
    ],
    EndpointType.SAFETY: [
        r"safety\s*(?:end)?point[s]?",
        r"adverse\s*event[s]?",
        r"tolerability",
    ],
}


# =============================================================================
# SITE/COUNTRY PATTERNS
# =============================================================================

COUNTRY_PATTERNS = [
    # "conducted in 15 countries"
    r"conducted\s*(?:in|across)\s*(\d+)\s*(?:countries|sites|centers)",
    # "United States, Germany, France"
    r"(?:countries?|sites?|centers?)\s*(?:included?|:)\s*([A-Z][a-z]+(?:,?\s*(?:and\s*)?[A-Z][a-z]+)*)",
]

# Known country names for extraction
COUNTRIES = {
    "united states", "usa", "us", "germany", "france", "uk", "united kingdom",
    "italy", "spain", "canada", "australia", "japan", "china", "brazil",
    "netherlands", "belgium", "switzerland", "austria", "sweden", "norway",
    "denmark", "finland", "poland", "czech republic", "hungary", "israel",
    "south korea", "korea", "taiwan", "india", "russia", "mexico", "argentina",
}


# =============================================================================
# FEASIBILITY DETECTOR CLASS
# =============================================================================


class FeasibilityDetector:
    """
    Extracts clinical trial feasibility information from documents.

    Covers:
    - Eligibility criteria (inclusion/exclusion)
    - Epidemiology (prevalence, incidence, demographics)
    - Patient journey phases
    - Study endpoints
    - Site/country information
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.run_id = str(self.config.get("run_id") or generate_run_id("FEAS"))
        self.pipeline_version = (
            self.config.get("pipeline_version") or get_git_revision_hash()
        )
        self.doc_fingerprint_default = (
            self.config.get("doc_fingerprint") or "unknown-doc-fingerprint"
        )

        # Context window for evidence extraction
        self.context_window = int(self.config.get("context_window", 300))

        # Feature flags
        self.enable_eligibility = self.config.get("enable_eligibility", True)
        self.enable_epidemiology = self.config.get("enable_epidemiology", True)
        self.enable_patient_journey = self.config.get("enable_patient_journey", True)
        self.enable_endpoints = self.config.get("enable_endpoints", True)
        self.enable_sites = self.config.get("enable_sites", True)

        # Compile patterns
        self._compile_patterns()

        # Stats for summary
        self._extraction_stats: Dict[str, int] = {}

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for efficiency."""
        self.inclusion_re = [re.compile(p, re.IGNORECASE) for p in INCLUSION_MARKERS]
        self.exclusion_re = [re.compile(p, re.IGNORECASE) for p in EXCLUSION_MARKERS]

        self.prevalence_re = [re.compile(p, re.IGNORECASE) for p in PREVALENCE_PATTERNS]
        self.incidence_re = [re.compile(p, re.IGNORECASE) for p in INCIDENCE_PATTERNS]
        self.demographics_re = [
            re.compile(p, re.IGNORECASE) for p in DEMOGRAPHICS_PATTERNS
        ]

        self.endpoint_re = {
            etype: [re.compile(p, re.IGNORECASE) for p in patterns]
            for etype, patterns in ENDPOINT_PATTERNS.items()
        }

        self.criterion_category_re = {
            cat: re.compile(pattern, re.IGNORECASE)
            for cat, pattern in CRITERION_CATEGORIES.items()
        }

    def extract(self, doc_structure: DocumentGraph) -> List[FeasibilityCandidate]:
        """
        Extract feasibility information from document.

        Args:
            doc_structure: Parsed document graph

        Returns:
            List of FeasibilityCandidate objects
        """
        doc = doc_structure
        candidates: List[FeasibilityCandidate] = []
        seen: Set[str] = set()

        # Get document fingerprint
        doc_fingerprint = getattr(doc, "fingerprint", self.doc_fingerprint_default)
        doc_id = getattr(doc, "doc_id", "unknown")

        # Track current section for context
        current_section = "unknown"

        for block in doc.iter_linear_blocks(skip_header_footer=True):
            text = (block.text or "").strip()
            if not text:
                continue

            page_num = getattr(block, "page_number", None)

            # Update section tracking
            section = self._detect_section(text)
            if section:
                current_section = section

            # Extract different types of feasibility info
            if self.enable_eligibility:
                candidates.extend(
                    self._extract_eligibility(
                        text, doc_id, doc_fingerprint, page_num, current_section, seen
                    )
                )

            if self.enable_epidemiology:
                candidates.extend(
                    self._extract_epidemiology(
                        text, doc_id, doc_fingerprint, page_num, current_section, seen
                    )
                )

            if self.enable_patient_journey:
                candidates.extend(
                    self._extract_patient_journey(
                        text, doc_id, doc_fingerprint, page_num, current_section, seen
                    )
                )

            if self.enable_endpoints:
                candidates.extend(
                    self._extract_endpoints(
                        text, doc_id, doc_fingerprint, page_num, current_section, seen
                    )
                )

            if self.enable_sites:
                candidates.extend(
                    self._extract_sites(
                        text, doc_id, doc_fingerprint, page_num, current_section, seen
                    )
                )

        self._update_stats(candidates)
        return candidates

    def _detect_section(self, text: str) -> Optional[str]:
        """Detect if text is a section header."""
        text_lower = text.lower().strip()

        # Short text more likely to be a header
        if len(text) > 100:
            return None

        for section_name, patterns in SECTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return section_name

        return None

    def _extract_eligibility(
        self,
        text: str,
        doc_id: str,
        doc_fingerprint: str,
        page_num: Optional[int],
        section: str,
        seen: Set[str],
    ) -> List[FeasibilityCandidate]:
        """Extract eligibility criteria from text."""
        candidates = []

        # Check for inclusion criteria markers
        is_inclusion = any(p.search(text) for p in self.inclusion_re)
        is_exclusion = any(p.search(text) for p in self.exclusion_re)

        # If we're in eligibility section, look for bullet points
        if section == "eligibility" or is_inclusion or is_exclusion:
            # Extract bullet point items
            criteria = self._extract_bullet_items(text)

            for criterion_text in criteria:
                if len(criterion_text) < 10:
                    continue

                # Determine type based on context
                criterion_type = CriterionType.EXCLUSION if is_exclusion else CriterionType.INCLUSION

                # Deduplicate
                dedup_key = f"elig:{criterion_text[:50].lower()}"
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                # Detect category
                category = self._detect_criterion_category(criterion_text)

                field_type = (
                    FeasibilityFieldType.ELIGIBILITY_EXCLUSION
                    if criterion_type == CriterionType.EXCLUSION
                    else FeasibilityFieldType.ELIGIBILITY_INCLUSION
                )

                candidate = self._make_candidate(
                    doc_id=doc_id,
                    doc_fingerprint=doc_fingerprint,
                    matched_text=criterion_text,
                    context_text=text[:self.context_window],
                    field_type=field_type,
                    page_num=page_num,
                    section=section,
                    confidence=0.7 if section == "eligibility" else 0.5,
                )
                candidate.eligibility_criterion = EligibilityCriterion(
                    criterion_type=criterion_type,
                    text=criterion_text,
                    category=category,
                )
                candidates.append(candidate)

        return candidates

    def _extract_bullet_items(self, text: str) -> List[str]:
        """Extract bullet point or numbered list items."""
        items = []

        # Split by common bullet patterns
        patterns = [
            r"(?:^|\n)\s*[•●○▪▸]\s*(.+?)(?=\n\s*[•●○▪▸]|\n\n|$)",  # Bullet points
            r"(?:^|\n)\s*[-–—]\s*(.+?)(?=\n\s*[-–—]|\n\n|$)",  # Dashes
            r"(?:^|\n)\s*\d+[.)]\s*(.+?)(?=\n\s*\d+[.)]|\n\n|$)",  # Numbered
            r"(?:^|\n)\s*[a-z][.)]\s*(.+?)(?=\n\s*[a-z][.)]|\n\n|$)",  # Lettered
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
            items.extend([m.strip() for m in matches if m.strip()])

        # If no bullet items found, check if this is a single criterion sentence
        if not items and len(text) < 500:
            # Look for criterion-like sentences
            sentences = re.split(r'[.;](?=\s|$)', text)
            for sent in sentences:
                sent = sent.strip()
                if 20 < len(sent) < 300:
                    # Check if it looks like a criterion
                    if any(
                        p.search(sent)
                        for p in self.criterion_category_re.values()
                    ):
                        items.append(sent)

        return items

    def _detect_criterion_category(self, text: str) -> Optional[str]:
        """Detect the category of an eligibility criterion."""
        text_lower = text.lower()
        for category, pattern in self.criterion_category_re.items():
            if pattern.search(text_lower):
                return category
        return None

    def _extract_epidemiology(
        self,
        text: str,
        doc_id: str,
        doc_fingerprint: str,
        page_num: Optional[int],
        section: str,
        seen: Set[str],
    ) -> List[FeasibilityCandidate]:
        """Extract epidemiology statistics from text."""
        candidates = []

        # Prevalence patterns
        for pattern in self.prevalence_re:
            for match in pattern.finditer(text):
                value = match.group(1) if match.lastindex else match.group(0)
                dedup_key = f"prev:{value.lower()}"
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                candidate = self._make_candidate(
                    doc_id=doc_id,
                    doc_fingerprint=doc_fingerprint,
                    matched_text=match.group(0),
                    context_text=self._get_context(text, match.start(), match.end()),
                    field_type=FeasibilityFieldType.EPIDEMIOLOGY_PREVALENCE,
                    page_num=page_num,
                    section=section,
                    confidence=0.8,
                )
                candidate.epidemiology_data = EpidemiologyData(
                    data_type="prevalence",
                    value=value.strip(),
                )
                candidates.append(candidate)

        # Incidence patterns
        for pattern in self.incidence_re:
            for match in pattern.finditer(text):
                value = match.group(1) if match.lastindex else match.group(0)
                dedup_key = f"inc:{value.lower()}"
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                candidate = self._make_candidate(
                    doc_id=doc_id,
                    doc_fingerprint=doc_fingerprint,
                    matched_text=match.group(0),
                    context_text=self._get_context(text, match.start(), match.end()),
                    field_type=FeasibilityFieldType.EPIDEMIOLOGY_INCIDENCE,
                    page_num=page_num,
                    section=section,
                    confidence=0.8,
                )
                candidate.epidemiology_data = EpidemiologyData(
                    data_type="incidence",
                    value=value.strip(),
                )
                candidates.append(candidate)

        # Demographics patterns
        for pattern in self.demographics_re:
            for match in pattern.finditer(text):
                value = match.group(1) if match.lastindex else match.group(0)
                dedup_key = f"demo:{value.lower()}"
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                candidate = self._make_candidate(
                    doc_id=doc_id,
                    doc_fingerprint=doc_fingerprint,
                    matched_text=match.group(0),
                    context_text=self._get_context(text, match.start(), match.end()),
                    field_type=FeasibilityFieldType.EPIDEMIOLOGY_DEMOGRAPHICS,
                    page_num=page_num,
                    section=section,
                    confidence=0.7,
                )
                candidate.epidemiology_data = EpidemiologyData(
                    data_type="demographics",
                    value=value.strip(),
                )
                candidates.append(candidate)

        return candidates

    def _extract_patient_journey(
        self,
        text: str,
        doc_id: str,
        doc_fingerprint: str,
        page_num: Optional[int],
        section: str,
        seen: Set[str],
    ) -> List[FeasibilityCandidate]:
        """Extract patient journey phase information."""
        candidates = []
        text_lower = text.lower()

        for phase_type, keywords in JOURNEY_PHASE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    dedup_key = f"journey:{phase_type.value}"
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)

                    # Try to extract duration
                    duration_match = DURATION_PATTERN.search(text)
                    duration = None
                    if duration_match:
                        duration = f"{duration_match.group(1)} {duration_match.group(2)}"

                    candidate = self._make_candidate(
                        doc_id=doc_id,
                        doc_fingerprint=doc_fingerprint,
                        matched_text=keyword,
                        context_text=text[:self.context_window],
                        field_type=FeasibilityFieldType.PATIENT_JOURNEY_PHASE,
                        page_num=page_num,
                        section=section,
                        confidence=0.6,
                    )
                    candidate.patient_journey_phase = PatientJourneyPhase(
                        phase_type=phase_type,
                        description=text[:200],
                        duration=duration,
                    )
                    candidates.append(candidate)
                    break  # Only one match per phase type per block

        return candidates

    def _extract_endpoints(
        self,
        text: str,
        doc_id: str,
        doc_fingerprint: str,
        page_num: Optional[int],
        section: str,
        seen: Set[str],
    ) -> List[FeasibilityCandidate]:
        """Extract study endpoint information."""
        candidates = []

        for endpoint_type, patterns in self.endpoint_re.items():
            for pattern in patterns:
                match = pattern.search(text)
                if match:
                    # Try to extract the actual endpoint text after the marker
                    endpoint_text = text[match.end():].strip()
                    # Take first sentence or up to 200 chars
                    endpoint_text = re.split(r'[.;]', endpoint_text)[0][:200].strip()

                    if len(endpoint_text) < 10:
                        endpoint_text = text[match.start():match.start() + 200]

                    dedup_key = f"endpoint:{endpoint_type.value}:{endpoint_text[:30].lower()}"
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)

                    candidate = self._make_candidate(
                        doc_id=doc_id,
                        doc_fingerprint=doc_fingerprint,
                        matched_text=endpoint_text,
                        context_text=text[:self.context_window],
                        field_type=FeasibilityFieldType.STUDY_ENDPOINT,
                        page_num=page_num,
                        section=section,
                        confidence=0.7,
                    )
                    candidate.study_endpoint = StudyEndpoint(
                        endpoint_type=endpoint_type,
                        name=endpoint_text,
                    )
                    candidates.append(candidate)

        return candidates

    def _extract_sites(
        self,
        text: str,
        doc_id: str,
        doc_fingerprint: str,
        page_num: Optional[int],
        section: str,
        seen: Set[str],
    ) -> List[FeasibilityCandidate]:
        """Extract site/country information."""
        candidates = []
        text_lower = text.lower()

        # Check for country mentions
        found_countries = []
        for country in COUNTRIES:
            if country in text_lower:
                found_countries.append(country)

        if found_countries:
            dedup_key = f"sites:{','.join(sorted(found_countries)[:3])}"
            if dedup_key not in seen:
                seen.add(dedup_key)

                # Try to extract site count
                site_count_match = re.search(
                    r"(\d+)\s*(?:sites?|centers?|countries)", text_lower
                )
                site_count = int(site_count_match.group(1)) if site_count_match else None

                for country in found_countries[:5]:  # Limit to top 5
                    candidate = self._make_candidate(
                        doc_id=doc_id,
                        doc_fingerprint=doc_fingerprint,
                        matched_text=country,
                        context_text=text[:self.context_window],
                        field_type=FeasibilityFieldType.STUDY_SITE,
                        page_num=page_num,
                        section=section,
                        confidence=0.6,
                    )
                    candidate.study_site = StudySite(
                        country=country.title(),
                        site_count=site_count,
                    )
                    candidates.append(candidate)

        return candidates

    def _make_candidate(
        self,
        doc_id: str,
        doc_fingerprint: str,
        matched_text: str,
        context_text: str,
        field_type: FeasibilityFieldType,
        page_num: Optional[int],
        section: str,
        confidence: float,
    ) -> FeasibilityCandidate:
        """Create a FeasibilityCandidate with provenance."""
        provenance = FeasibilityProvenanceMetadata(
            pipeline_version=self.pipeline_version,
            run_id=self.run_id,
            doc_fingerprint=doc_fingerprint,
            generator_name=FeasibilityGeneratorType.PATTERN_MATCH,
        )

        return FeasibilityCandidate(
            doc_id=doc_id,
            field_type=field_type,
            generator_type=FeasibilityGeneratorType.PATTERN_MATCH,
            matched_text=matched_text,
            context_text=context_text,
            page_number=page_num,
            section_name=section,
            confidence=confidence,
            provenance=provenance,
        )

    def _get_context(self, text: str, start: int, end: int) -> str:
        """Extract context window around a match."""
        ctx_start = max(0, start - self.context_window // 2)
        ctx_end = min(len(text), end + self.context_window // 2)
        return text[ctx_start:ctx_end]

    def _update_stats(self, candidates: List[FeasibilityCandidate]) -> None:
        """Update extraction statistics."""
        for c in candidates:
            field_type = c.field_type.value
            self._extraction_stats[field_type] = (
                self._extraction_stats.get(field_type, 0) + 1
            )

    def get_stats(self) -> Dict[str, int]:
        """Return extraction statistics."""
        return self._extraction_stats.copy()

    def print_summary(self) -> None:
        """Print extraction summary."""
        if not self._extraction_stats:
            print("\nFeasibility extraction: No items found")
            return

        total = sum(self._extraction_stats.values())
        print(f"\nFeasibility extraction: {total} items found")
        print("─" * 50)
        for field_type, count in sorted(self._extraction_stats.items()):
            print(f"  {field_type:<40} {count:>5}")

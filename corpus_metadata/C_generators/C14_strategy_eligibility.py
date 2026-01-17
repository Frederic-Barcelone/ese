# corpus_metadata/C_generators/C14_strategy_eligibility.py
"""
Computable Eligibility Extractor - Extract structured eligibility criteria.

Targets:
- Age range
- Diagnosis requirements with confirmation method
- Lab thresholds (analyte, operator, value, unit, timing)
- Prior/concomitant medication requirements
- Comorbidity exclusions

Output is designed to be translatable to cohort queries (OMOP, i2b2).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from A_core.A04_feasibility_models import (
    AgeRange,
    ComorbidityExclusion,
    ComparisonOperator,
    ComputableEligibility,
    DiagnosisConfirmationType,
    DiagnosisRequirement,
    EvidenceSnippet,
    LabThreshold,
    MedicationRequirement,
)
from B_parsing.B02_doc_graph import DocumentGraph


class EligibilityExtractor:
    """
    Extract structured eligibility criteria from clinical trial documents.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Age patterns
        self.age_patterns = [
            # "aged 18 to 65 years" or "age 18-65"
            r"(?:aged?|ages?)\s*(?:of\s*)?(\d+)\s*(?:to|–|-|through)\s*(\d+)\s*(?:years?)?",
            # "18 years or older" or "≥18 years"
            r"(?:≥|>=|at\s+least)\s*(\d+)\s*(?:years?\s+(?:of\s+age|old)?)?",
            # "between 18 and 65 years"
            r"between\s+(\d+)\s+and\s+(\d+)\s+years?",
            # "adults (18-65)"
            r"adults?\s*\((\d+)\s*[-–]\s*(\d+)\)",
        ]

        # Lab threshold patterns
        self.lab_patterns = [
            # "serum C3 <77 mg/dL"
            (r"(?:serum\s+)?([A-Za-z0-9]+)\s*(<=?|>=?|≤|≥|<|>)\s*([\d.]+)\s*(\w+(?:/\w+)?)", "simple"),
            # "eGFR ≥30 mL/min/1.73m²"
            (r"([eE]?GFR)\s*(<=?|>=?|≤|≥|<|>)\s*([\d.]+)\s*(mL/min(?:/1\.73\s*m²)?)", "egfr"),
            # "UPCR ≥1.0 g/g"
            (r"([A-Z]{2,}(?:-[A-Z])?)\s*(<=?|>=?|≤|≥|<|>)\s*([\d.]+)\s*(\w+(?:/\w+)?)", "ratio"),
            # "hemoglobin of at least 10 g/dL"
            (r"([A-Za-z]+)\s+(?:of\s+)?(?:at\s+least|≥|>=)\s*([\d.]+)\s*(\w+(?:/\w+)?)", "at_least"),
            # "platelet count >100,000/μL"
            (r"([A-Za-z\s]+(?:count)?)\s*(<=?|>=?|≤|≥|<|>)\s*([\d,]+)\s*(/?\w+)", "count"),
        ]

        # Common lab analytes (for normalization)
        self.common_analytes = {
            "c3": "C3 (complement)",
            "egfr": "eGFR",
            "gfr": "GFR",
            "upcr": "UPCR (urine protein-creatinine ratio)",
            "hemoglobin": "Hemoglobin",
            "hgb": "Hemoglobin",
            "hb": "Hemoglobin",
            "platelet": "Platelet count",
            "wbc": "WBC (white blood cell count)",
            "creatinine": "Creatinine",
            "bilirubin": "Bilirubin",
            "alt": "ALT",
            "ast": "AST",
            "albumin": "Albumin",
            "proteinuria": "Proteinuria",
        }

        # Medication patterns
        self.medication_patterns = [
            # "stable dose of X for at least Y weeks"
            (r"stable\s+(?:dose\s+of\s+)?(.+?)\s+for\s+(?:at\s+least\s+)?(\d+)\s+(weeks?|months?|days?)", "stable"),
            # "prior treatment with X"
            (r"prior\s+(?:treatment|therapy)\s+with\s+(.+?)(?:\s+for|\.|,|;|$)", "prior"),
            # "X prohibited" or "not receiving X"
            (r"(.+?)\s+(?:is\s+)?prohibited", "prohibited"),
            (r"not\s+(?:currently\s+)?receiving\s+(.+?)(?:\.|,|;|$)", "prohibited"),
            # "washout of X weeks"
            (r"(\d+)\s*(?:week|month|day)s?\s+washout\s+(?:of|from|for)\s+(.+?)(?:\.|,|;|$)", "washout"),
        ]

        # Diagnosis confirmation patterns
        self.diagnosis_patterns = [
            # "biopsy-confirmed X"
            (r"biopsy[- ](?:confirmed|proven|documented)\s+(.+?)(?:\s+within|\.|,|;|$)", DiagnosisConfirmationType.BIOPSY),
            # "genetically confirmed X"
            (r"(?:genetic(?:ally)?|molecular(?:ly)?)[- ](?:confirmed|proven|documented)\s+(.+?)(?:\s+within|\.|,|;|$)", DiagnosisConfirmationType.GENETIC_TEST),
            # "histologically confirmed"
            (r"histologic(?:ally)?[- ](?:confirmed|proven)\s+(.+?)(?:\s+within|\.|,|;|$)", DiagnosisConfirmationType.BIOPSY),
            # "imaging-confirmed"
            (r"(?:imaging|radiologic(?:ally)?)[- ](?:confirmed|proven)\s+(.+?)(?:\s+within|\.|,|;|$)", DiagnosisConfirmationType.IMAGING),
            # "clinical diagnosis of X"
            (r"clinical\s+diagnosis\s+of\s+(.+?)(?:\s+within|\.|,|;|$)", DiagnosisConfirmationType.CLINICAL),
        ]

        # Comorbidity exclusion patterns
        self.exclusion_patterns = [
            # "history of X"
            r"(?:known\s+)?history\s+of\s+(.+?)(?:\s+within|\.|,|;|$)",
            # "presence of X"
            r"presence\s+of\s+(.+?)(?:\.|,|;|$)",
            # "diagnosis of X"
            r"(?:current\s+)?diagnosis\s+of\s+(.+?)(?:\.|,|;|$)",
            # "X excluded"
            r"(.+?)\s+(?:is\s+)?excluded",
        ]

    def extract(self, doc: DocumentGraph) -> ComputableEligibility:
        """Extract computable eligibility criteria from document."""
        result = ComputableEligibility()

        # Find eligibility section text
        eligibility_text = self._find_eligibility_sections(doc)

        if eligibility_text:
            inclusion_text, exclusion_text = eligibility_text

            result.inclusion_criteria_text = inclusion_text
            result.exclusion_criteria_text = exclusion_text

            # Extract from inclusion criteria
            if inclusion_text:
                result.age = self._extract_age(inclusion_text)
                result.diagnosis_requirements = self._extract_diagnoses(inclusion_text, doc)
                result.lab_thresholds = self._extract_lab_thresholds(inclusion_text, doc)
                result.medication_requirements.extend(
                    self._extract_medications(inclusion_text, doc, "inclusion")
                )

            # Extract from exclusion criteria
            if exclusion_text:
                result.comorbidity_exclusions = self._extract_exclusions(exclusion_text, doc)
                result.medication_requirements.extend(
                    self._extract_medications(exclusion_text, doc, "exclusion")
                )

        # Also extract from full document (may find data outside explicit sections)
        for block in doc.iter_linear_blocks(skip_header_footer=True):
            text = block.text
            if not text:
                continue

            # Age (if not found)
            if not result.age:
                age = self._extract_age(text)
                if age and (age.min_age or age.max_age):
                    age.evidence.append(EvidenceSnippet(
                        text=text[:200],
                        page=block.page_num,
                    ))
                    result.age = age

            # Lab thresholds
            for lab in self._extract_lab_thresholds(text, doc, block.page_num):
                if not self._lab_already_exists(lab, result.lab_thresholds):
                    result.lab_thresholds.append(lab)

            # Diagnoses
            for diag in self._extract_diagnoses(text, doc, block.page_num):
                if not self._diagnosis_already_exists(diag, result.diagnosis_requirements):
                    result.diagnosis_requirements.append(diag)

        return result

    def _find_eligibility_sections(
        self,
        doc: DocumentGraph
    ) -> Optional[Tuple[str, str]]:
        """Find and extract inclusion/exclusion criteria sections."""
        inclusion_text = []
        exclusion_text = []

        in_inclusion = False
        in_exclusion = False

        for block in doc.iter_linear_blocks(skip_header_footer=True):
            text = (block.text or "").strip()
            text_lower = text.lower()

            # Check for section headers
            if any(kw in text_lower for kw in ["inclusion criteria", "key inclusion", "eligibility criteria"]):
                in_inclusion = True
                in_exclusion = False
                continue
            elif any(kw in text_lower for kw in ["exclusion criteria", "key exclusion"]):
                in_inclusion = False
                in_exclusion = True
                continue
            elif in_inclusion or in_exclusion:
                # Check for section end
                if self._is_new_section(text_lower):
                    in_inclusion = False
                    in_exclusion = False
                    continue

            if in_inclusion:
                inclusion_text.append(text)
            elif in_exclusion:
                exclusion_text.append(text)

        inc = " ".join(inclusion_text) if inclusion_text else None
        exc = " ".join(exclusion_text) if exclusion_text else None

        return (inc, exc) if inc or exc else None

    def _is_new_section(self, text_lower: str) -> bool:
        """Check if text indicates a new section (not eligibility)."""
        section_indicators = [
            "statistical analysis", "endpoints", "outcome", "study design",
            "treatment", "intervention", "methods", "results", "discussion",
            "references", "acknowledgment"
        ]
        return any(ind in text_lower for ind in section_indicators)

    def _extract_age(self, text: str) -> Optional[AgeRange]:
        """Extract age range from text."""
        for pattern in self.age_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) >= 2:
                    return AgeRange(
                        min_age=int(groups[0]),
                        max_age=int(groups[1]) if groups[1] else None,
                        evidence=[EvidenceSnippet(text=match.group(0), page=1)]
                    )
                elif len(groups) == 1:
                    return AgeRange(
                        min_age=int(groups[0]),
                        evidence=[EvidenceSnippet(text=match.group(0), page=1)]
                    )
        return None

    def _extract_lab_thresholds(
        self,
        text: str,
        doc: DocumentGraph,
        page_num: int = 1
    ) -> List[LabThreshold]:
        """Extract laboratory thresholds from text."""
        results = []

        for pattern, pattern_type in self.lab_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    groups = match.groups()

                    if pattern_type == "at_least":
                        analyte = groups[0].strip()
                        operator = ComparisonOperator.GE
                        value = float(groups[1].replace(",", ""))
                        unit = groups[2] if len(groups) > 2 else ""
                    elif pattern_type == "count":
                        analyte = groups[0].strip()
                        operator = self._parse_operator(groups[1])
                        value = float(groups[2].replace(",", ""))
                        unit = groups[3] if len(groups) > 3 else ""
                    else:
                        analyte = groups[0].strip()
                        operator = self._parse_operator(groups[1])
                        value = float(groups[2].replace(",", ""))
                        unit = groups[3] if len(groups) > 3 else ""

                    # Normalize analyte name
                    analyte_normalized = self._normalize_analyte(analyte)

                    results.append(LabThreshold(
                        analyte=analyte_normalized,
                        operator=operator,
                        value=value,
                        unit=unit,
                        evidence=[EvidenceSnippet(text=match.group(0), page=page_num)]
                    ))
                except (ValueError, IndexError):
                    continue

        return results

    def _parse_operator(self, op_str: str) -> ComparisonOperator:
        """Parse comparison operator string."""
        op_map = {
            "<": ComparisonOperator.LT,
            "<=": ComparisonOperator.LE,
            "≤": ComparisonOperator.LE,
            ">": ComparisonOperator.GT,
            ">=": ComparisonOperator.GE,
            "≥": ComparisonOperator.GE,
            "=": ComparisonOperator.EQ,
        }
        return op_map.get(op_str.strip(), ComparisonOperator.EQ)

    def _normalize_analyte(self, analyte: str) -> str:
        """Normalize analyte name."""
        analyte_lower = analyte.lower().strip()
        return self.common_analytes.get(analyte_lower, analyte)

    def _extract_diagnoses(
        self,
        text: str,
        doc: DocumentGraph,
        page_num: int = 1
    ) -> List[DiagnosisRequirement]:
        """Extract diagnosis requirements from text."""
        results = []

        for pattern, confirmation_type in self.diagnosis_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                condition = match.group(1).strip()
                # Clean up condition text
                condition = re.sub(r"\s+", " ", condition)
                condition = condition.strip(" .,;:")

                if condition and len(condition) > 3:
                    # Check for timing
                    timing = None
                    timing_match = re.search(
                        r"within\s+(\d+)\s+(months?|years?|weeks?|days?)",
                        text[match.end():match.end()+50],
                        re.IGNORECASE
                    )
                    if timing_match:
                        timing = f"within {timing_match.group(1)} {timing_match.group(2)}"

                    results.append(DiagnosisRequirement(
                        condition=condition,
                        confirmation_method=confirmation_type,
                        timing=timing,
                        evidence=[EvidenceSnippet(text=match.group(0), page=page_num)]
                    ))

        return results

    def _extract_medications(
        self,
        text: str,
        doc: DocumentGraph,
        context: str = "inclusion"
    ) -> List[MedicationRequirement]:
        """Extract medication requirements from text."""
        results = []

        for pattern, pattern_type in self.medication_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    groups = match.groups()

                    if pattern_type == "stable":
                        medication = groups[0].strip()
                        duration = f"{groups[1]} {groups[2]}"
                        results.append(MedicationRequirement(
                            medication=medication,
                            requirement_type="required",
                            duration=duration,
                            stable_dose=True,
                            evidence=[EvidenceSnippet(text=match.group(0), page=1)]
                        ))
                    elif pattern_type == "prior":
                        medication = groups[0].strip()
                        results.append(MedicationRequirement(
                            medication=medication,
                            requirement_type="required" if context == "inclusion" else "prohibited",
                            evidence=[EvidenceSnippet(text=match.group(0), page=1)]
                        ))
                    elif pattern_type == "prohibited":
                        medication = groups[0].strip()
                        results.append(MedicationRequirement(
                            medication=medication,
                            requirement_type="prohibited",
                            evidence=[EvidenceSnippet(text=match.group(0), page=1)]
                        ))
                    elif pattern_type == "washout":
                        duration = groups[0]
                        medication = groups[1].strip()
                        results.append(MedicationRequirement(
                            medication=medication,
                            requirement_type="washout",
                            washout_period=f"{duration} weeks",
                            evidence=[EvidenceSnippet(text=match.group(0), page=1)]
                        ))
                except (ValueError, IndexError):
                    continue

        return results

    def _extract_exclusions(
        self,
        text: str,
        doc: DocumentGraph
    ) -> List[ComorbidityExclusion]:
        """Extract comorbidity exclusions from text."""
        results = []

        for pattern in self.exclusion_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                condition = match.group(1).strip()
                # Clean up
                condition = re.sub(r"\s+", " ", condition)
                condition = condition.strip(" .,;:")

                # Skip if too short or looks like noise
                if condition and len(condition) > 5 and not condition.startswith("the "):
                    # Check for timing
                    timing = None
                    timing_match = re.search(
                        r"within\s+(\d+)\s+(months?|years?|weeks?|days?)",
                        text[match.end():match.end()+50],
                        re.IGNORECASE
                    )
                    if timing_match:
                        timing = f"within {timing_match.group(1)} {timing_match.group(2)}"

                    results.append(ComorbidityExclusion(
                        condition=condition,
                        timing=timing,
                        evidence=[EvidenceSnippet(text=match.group(0), page=1)]
                    ))

        return results

    def _lab_already_exists(
        self,
        new_lab: LabThreshold,
        existing: List[LabThreshold]
    ) -> bool:
        """Check if a lab threshold already exists."""
        for lab in existing:
            if (lab.analyte.lower() == new_lab.analyte.lower() and
                lab.operator == new_lab.operator and
                lab.value == new_lab.value):
                return True
        return False

    def _diagnosis_already_exists(
        self,
        new_diag: DiagnosisRequirement,
        existing: List[DiagnosisRequirement]
    ) -> bool:
        """Check if a diagnosis requirement already exists."""
        for diag in existing:
            if diag.condition.lower() == new_diag.condition.lower():
                return True
        return False


def extract_eligibility(doc: DocumentGraph) -> ComputableEligibility:
    """Convenience function for eligibility extraction."""
    extractor = EligibilityExtractor()
    return extractor.extract(doc)

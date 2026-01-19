# corpus_metadata/E_normalization/E09_drug_combination_parser.py
"""
Drug combination decomposition parser.

Parses drug combination strings into individual components:
- "ACE-I + SGLT2i" → [ACE-I, SGLT2i]
- "metformin/sitagliptin" → [metformin, sitagliptin]
- "triple therapy (ACE-I, ARB, diuretic)" → [ACE-I, ARB, diuretic]

Also handles:
- Dose/frequency extraction
- Drug class recognition
- Background therapy vs investigational drug distinction
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DrugComponent:
    """Single drug component from a combination."""

    name: str
    drug_class: Optional[str] = None
    dose: Optional[str] = None
    dose_value: Optional[float] = None
    dose_unit: Optional[str] = None
    frequency: Optional[str] = None
    route: Optional[str] = None
    is_class_reference: bool = False  # True if this is a drug class, not specific drug


@dataclass
class ParsedDrugCombination:
    """Result of parsing a drug combination string."""

    original_text: str
    components: List[DrugComponent] = field(default_factory=list)
    combination_type: str = "unknown"  # "fixed_dose", "concomitant", "background", "regimen"
    is_background_therapy: bool = False


class DrugCombinationParser:
    """
    Parser for drug combination strings.

    Decomposes multi-drug expressions into individual components
    with structured dose/frequency information.
    """

    # Drug class abbreviations and their expansions
    DRUG_CLASSES = {
        # Cardiovascular
        "ace-i": "ACE inhibitor",
        "acei": "ACE inhibitor",
        "arb": "angiotensin receptor blocker",
        "arni": "angiotensin receptor-neprilysin inhibitor",
        "bb": "beta blocker",
        "ccb": "calcium channel blocker",
        "mra": "mineralocorticoid receptor antagonist",
        "sglt2i": "SGLT2 inhibitor",
        "sglt-2i": "SGLT2 inhibitor",
        "glp-1ra": "GLP-1 receptor agonist",
        "dpp-4i": "DPP-4 inhibitor",
        # Immunosuppressants
        "cni": "calcineurin inhibitor",
        "mmf": "mycophenolate mofetil",
        "aza": "azathioprine",
        # Steroids
        "cs": "corticosteroid",
        "gcs": "glucocorticoid",
        # Anticoagulants
        "doac": "direct oral anticoagulant",
        "noac": "novel oral anticoagulant",
        "lmwh": "low molecular weight heparin",
        # Biologics
        "mab": "monoclonal antibody",
        "tnfi": "TNF inhibitor",
    }

    # Combination separators
    SEPARATORS = [
        r"\s*\+\s*",  # "drug1 + drug2"
        r"\s*/\s*",  # "drug1/drug2"
        r"\s*\band\b\s*",  # "drug1 and drug2"
        r"\s*,\s*(?:and\s+)?",  # "drug1, drug2" or "drug1, and drug2"
        r"\s*;\s*",  # "drug1; drug2"
        r"\s*\bwith\b\s*",  # "drug1 with drug2"
        r"\s*\bplus\b\s*",  # "drug1 plus drug2"
    ]

    # Dose patterns
    DOSE_PATTERNS = [
        # "100 mg" or "100mg"
        r"(?P<value>[\d.]+)\s*(?P<unit>mg|g|mcg|µg|ug|ml|mL|IU|units?)",
        # "100-200 mg" (range)
        r"(?P<min>[\d.]+)\s*[-–]\s*(?P<max>[\d.]+)\s*(?P<unit>mg|g|mcg|µg|ug|ml|mL|IU|units?)",
    ]

    # Frequency patterns
    FREQUENCY_PATTERNS = [
        r"\b(?P<freq>once|twice|three\s+times?)\s+(?:a\s+)?(?P<period>daily|day|weekly|week)",
        r"\b(?P<freq>qd|bid|tid|qid|qhs|prn)\b",
        r"\b(?P<freq>\d+)\s*(?:times?\s+)?(?:per|/)\s*(?P<period>day|week|month)",
        r"\bevery\s+(?P<interval>\d+)\s*(?P<period>hours?|days?|weeks?)",
    ]

    # Route patterns
    ROUTE_PATTERNS = [
        r"\b(?P<route>oral(?:ly)?|iv|intravenous(?:ly)?|im|intramuscular(?:ly)?|"
        r"sc|subcutaneous(?:ly)?|topical(?:ly)?|inhaled?|po)\b",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self._separator_re = re.compile("|".join(self.SEPARATORS), re.IGNORECASE)
        self._dose_patterns = [re.compile(p, re.IGNORECASE) for p in self.DOSE_PATTERNS]
        self._freq_patterns = [re.compile(p, re.IGNORECASE) for p in self.FREQUENCY_PATTERNS]
        self._route_patterns = [re.compile(p, re.IGNORECASE) for p in self.ROUTE_PATTERNS]

    def parse(self, text: str) -> ParsedDrugCombination:
        """
        Parse a drug combination string.

        Args:
            text: Drug combination text (e.g., "ACE-I + SGLT2i 10mg daily")

        Returns:
            ParsedDrugCombination with decomposed components
        """
        result = ParsedDrugCombination(original_text=text)

        # Detect combination type
        result.combination_type = self._detect_combination_type(text)
        result.is_background_therapy = self._is_background_therapy(text)

        # Split into components
        parts = self._split_combination(text)

        for part in parts:
            component = self._parse_component(part)
            if component:
                result.components.append(component)

        return result

    def _detect_combination_type(self, text: str) -> str:
        """Detect the type of drug combination."""
        text_lower = text.lower()

        if "fixed" in text_lower or "combination" in text_lower:
            return "fixed_dose"
        elif "background" in text_lower or "stable" in text_lower:
            return "background"
        elif "concomitant" in text_lower:
            return "concomitant"
        elif "regimen" in text_lower or "protocol" in text_lower:
            return "regimen"
        elif "+" in text or "/" in text:
            return "combination"
        else:
            return "single"

    def _is_background_therapy(self, text: str) -> bool:
        """Check if this represents background/standard therapy."""
        indicators = [
            "background",
            "standard of care",
            "soc",
            "stable",
            "baseline",
            "permitted",
            "allowed",
            "concomitant",
        ]
        text_lower = text.lower()
        return any(ind in text_lower for ind in indicators)

    def _split_combination(self, text: str) -> List[str]:
        """Split combination text into individual drug parts."""
        # Handle parenthetical lists: "triple therapy (A, B, C)"
        paren_match = re.search(r"\(([^)]+)\)", text)
        if paren_match:
            inner = paren_match.group(1)
            return self._split_combination(inner)

        # Split on separators
        parts = self._separator_re.split(text)

        # Clean up parts
        cleaned = []
        for part in parts:
            part = part.strip()
            if part and len(part) > 1:  # Skip single characters
                cleaned.append(part)

        return cleaned if cleaned else [text]

    def _parse_component(self, text: str) -> Optional[DrugComponent]:
        """Parse a single drug component."""
        if not text or len(text.strip()) < 2:
            return None

        text = text.strip()

        # Extract dose
        dose, dose_value, dose_unit = self._extract_dose(text)

        # Extract frequency
        frequency = self._extract_frequency(text)

        # Extract route
        route = self._extract_route(text)

        # Clean drug name (remove dose/freq/route)
        name = self._extract_drug_name(text)
        if not name:
            return None

        # Check if this is a drug class reference
        is_class = self._is_drug_class(name)
        drug_class = self._get_drug_class(name)

        return DrugComponent(
            name=name,
            drug_class=drug_class,
            dose=dose,
            dose_value=dose_value,
            dose_unit=dose_unit,
            frequency=frequency,
            route=route,
            is_class_reference=is_class,
        )

    def _extract_dose(self, text: str) -> Tuple[Optional[str], Optional[float], Optional[str]]:
        """Extract dose information from text."""
        for pattern in self._dose_patterns:
            match = pattern.search(text)
            if match:
                groups = match.groupdict()

                if "value" in groups:
                    try:
                        value = float(groups["value"])
                        unit = groups.get("unit", "")
                        return f"{value} {unit}".strip(), value, unit
                    except ValueError:
                        pass

                elif "min" in groups and "max" in groups:
                    try:
                        min_val = float(groups["min"])
                        max_val = float(groups["max"])
                        unit = groups.get("unit", "")
                        return f"{min_val}-{max_val} {unit}".strip(), (min_val + max_val) / 2, unit
                    except ValueError:
                        pass

        return None, None, None

    def _extract_frequency(self, text: str) -> Optional[str]:
        """Extract dosing frequency from text."""
        for pattern in self._freq_patterns:
            match = pattern.search(text)
            if match:
                return match.group(0)

        # Standard abbreviation mapping
        freq_map = {
            "qd": "once daily",
            "bid": "twice daily",
            "tid": "three times daily",
            "qid": "four times daily",
            "qhs": "at bedtime",
            "prn": "as needed",
        }

        text_lower = text.lower()
        for abbr, full in freq_map.items():
            if abbr in text_lower:
                return full

        return None

    def _extract_route(self, text: str) -> Optional[str]:
        """Extract administration route from text."""
        for pattern in self._route_patterns:
            match = pattern.search(text)
            if match:
                route = match.group("route").lower()
                # Normalize
                route_map = {
                    "po": "oral",
                    "iv": "intravenous",
                    "im": "intramuscular",
                    "sc": "subcutaneous",
                }
                return route_map.get(route, route)

        return None

    def _extract_drug_name(self, text: str) -> str:
        """Extract drug name by removing dose/frequency/route info."""
        name = text

        # Remove dose patterns
        for pattern in self._dose_patterns:
            name = pattern.sub("", name)

        # Remove frequency patterns
        for pattern in self._freq_patterns:
            name = pattern.sub("", name)

        # Remove route patterns
        for pattern in self._route_patterns:
            name = pattern.sub("", name)

        # Clean up
        name = re.sub(r"\s+", " ", name).strip()
        name = re.sub(r"^[,;/+\s]+|[,;/+\s]+$", "", name)

        return name

    def _is_drug_class(self, name: str) -> bool:
        """Check if name refers to a drug class rather than specific drug."""
        name_lower = name.lower().replace("-", "").replace(" ", "")

        # Check against known class abbreviations
        for abbr in self.DRUG_CLASSES:
            if abbr.replace("-", "") == name_lower:
                return True

        # Check for class indicators
        class_indicators = ["inhibitor", "blocker", "agonist", "antagonist", "class"]
        return any(ind in name.lower() for ind in class_indicators)

    def _get_drug_class(self, name: str) -> Optional[str]:
        """Get the drug class for a name if it's a class reference."""
        name_lower = name.lower().replace("-", "").replace(" ", "")

        for abbr, full_name in self.DRUG_CLASSES.items():
            if abbr.replace("-", "") == name_lower:
                return full_name

        return None


# Convenience function
def parse_drug_combination(text: str) -> ParsedDrugCombination:
    """
    Parse a drug combination string into components.

    Args:
        text: Drug combination text

    Returns:
        ParsedDrugCombination with decomposed components
    """
    parser = DrugCombinationParser()
    return parser.parse(text)


def decompose_drug_regimen(text: str) -> List[Dict[str, Any]]:
    """
    Decompose a drug regimen into a list of drug dictionaries.

    Args:
        text: Drug regimen text

    Returns:
        List of drug dictionaries with name, dose, frequency, etc.
    """
    result = parse_drug_combination(text)

    return [
        {
            "name": c.name,
            "drug_class": c.drug_class,
            "dose": c.dose,
            "dose_value": c.dose_value,
            "dose_unit": c.dose_unit,
            "frequency": c.frequency,
            "route": c.route,
            "is_class_reference": c.is_class_reference,
        }
        for c in result.components
    ]

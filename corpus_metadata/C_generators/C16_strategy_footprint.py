# corpus_metadata/C_generators/C16_strategy_footprint.py
"""
Recruitment Footprint Extractor - Extract geographic and temporal recruitment data.

Targets:
- Number of countries
- List of countries
- Number of sites
- Site types (academic, community, specialty)
- Enrollment dates and duration
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from A_core.A04_feasibility_models import (
    EvidenceSnippet,
    RecruitmentFootprint,
    SiteInfo,
)
from B_parsing.B02_doc_graph import DocumentGraph


# Common country names for extraction
COUNTRIES = [
    "Argentina", "Australia", "Austria", "Belgium", "Brazil", "Bulgaria",
    "Canada", "Chile", "China", "Colombia", "Croatia", "Czech Republic",
    "Czechia", "Denmark", "Estonia", "Finland", "France", "Germany", "Greece",
    "Hong Kong", "Hungary", "India", "Indonesia", "Ireland", "Israel", "Italy",
    "Japan", "Korea", "South Korea", "Latvia", "Lithuania", "Malaysia", "Mexico",
    "Netherlands", "New Zealand", "Norway", "Peru", "Philippines", "Poland",
    "Portugal", "Romania", "Russia", "Russian Federation", "Singapore",
    "Slovakia", "Slovenia", "South Africa", "Spain", "Sweden", "Switzerland",
    "Taiwan", "Thailand", "Turkey", "Ukraine", "United Kingdom", "UK",
    "United States", "USA", "US", "Vietnam"
]

# Build regex pattern
COUNTRY_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(c) for c in COUNTRIES) + r")\b",
    re.IGNORECASE
)


class RecruitmentFootprintExtractor:
    """
    Extract recruitment footprint data from clinical trial documents.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Site/country count patterns
        self.site_count_patterns = [
            # "35 sites in 18 countries"
            r"(\d+)\s+(?:study\s+)?sites?\s+(?:in|across)\s+(\d+)\s+countr(?:y|ies)",
            # "18 countries and 35 sites"
            r"(\d+)\s+countr(?:y|ies)\s+(?:and|with)\s+(\d+)\s+sites?",
            # "conducted at 35 sites"
            r"(?:conducted|performed)\s+(?:at|in)\s+(\d+)\s+(?:study\s+)?sites?",
            # "35 centers in 18 countries"
            r"(\d+)\s+(?:study\s+)?(?:centers?|institutions?)\s+(?:in|across)\s+(\d+)\s+countr(?:y|ies)",
            # "from 35 sites"
            r"(?:from|at)\s+(\d+)\s+(?:clinical\s+)?(?:sites?|centers?)",
        ]

        # Country count only patterns
        self.country_count_patterns = [
            r"(?:in|across)\s+(\d+)\s+countr(?:y|ies)",
            r"(\d+)\s+countr(?:y|ies)\s+(?:worldwide|globally|internationally)",
        ]

        # Site count only patterns
        self.site_only_patterns = [
            r"(\d+)\s+(?:clinical\s+)?(?:study\s+)?sites?",
            r"(\d+)\s+(?:investigational\s+)?(?:centers?|institutions?)",
            r"(?:total\s+of\s+)?(\d+)\s+sites?",
        ]

        # Enrollment date patterns
        self.date_patterns = [
            # "from January 2020 to December 2021"
            r"(?:from\s+)?([A-Z][a-z]+)\s+(\d{4})\s+(?:to|through|until)\s+([A-Z][a-z]+)\s+(\d{4})",
            # "between 2020 and 2021"
            r"between\s+(\d{4})\s+and\s+(\d{4})",
            # "enrollment began in January 2020"
            r"enrollment\s+(?:began|started|commenced)\s+(?:in\s+)?([A-Z][a-z]+)\s+(\d{4})",
            # "completed enrollment in December 2021"
            r"(?:completed|finished)\s+enrollment\s+(?:in\s+)?([A-Z][a-z]+)\s+(\d{4})",
        ]

        # Month mapping
        self.months = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12
        }

    def extract(self, doc: DocumentGraph) -> RecruitmentFootprint:
        """Extract recruitment footprint from document."""
        result = RecruitmentFootprint()

        # Process document blocks
        for block in doc.iter_linear_blocks(skip_header_footer=True):
            text = block.text
            if not text:
                continue

            page_num = block.page_num

            # Extract site/country counts
            if not result.num_sites or not result.num_countries:
                counts = self._extract_site_country_counts(text, page_num)
                if counts:
                    sites, countries, evidence = counts
                    if sites and not result.num_sites:
                        result.num_sites = sites
                    if countries and not result.num_countries:
                        result.num_countries = countries
                    if evidence:
                        result.evidence.extend(evidence)

            # Extract country names
            countries = self._extract_country_names(text)
            for country in countries:
                if country not in result.countries:
                    result.countries.append(country)

            # Extract dates
            if not result.enrollment_start:
                dates = self._extract_enrollment_dates(text, page_num)
                if dates:
                    start, end = dates
                    if start:
                        result.enrollment_start = start
                    if end:
                        result.enrollment_end = end

        # Extract from tables
        self._extract_from_tables(doc, result)

        # Compute enrollment duration
        if result.enrollment_start and result.enrollment_end:
            delta = result.enrollment_end - result.enrollment_start
            result.enrollment_duration_months = round(delta.days / 30.44, 1)

        # Update country count from list if not found
        if result.countries and not result.num_countries:
            result.num_countries = len(result.countries)

        return result

    def _extract_site_country_counts(
        self,
        text: str,
        page_num: int
    ) -> Optional[Tuple[Optional[int], Optional[int], List[EvidenceSnippet]]]:
        """Extract site and country counts from text."""
        # Try combined patterns first
        for pattern in self.site_count_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) >= 2:
                    # Pattern has both sites and countries
                    # Determine which is which based on context
                    num1 = int(groups[0])
                    num2 = int(groups[1])

                    # Usually sites > countries, but pattern determines order
                    if "sites" in match.group(0).lower()[:30]:
                        sites = num1
                        countries = num2
                    else:
                        countries = num1
                        sites = num2

                    return (
                        sites, countries,
                        [EvidenceSnippet(text=match.group(0), page=page_num)]
                    )
                elif len(groups) == 1:
                    # Only sites
                    return (
                        int(groups[0]), None,
                        [EvidenceSnippet(text=match.group(0), page=page_num)]
                    )

        # Try country count only
        for pattern in self.country_count_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return (
                    None, int(match.group(1)),
                    [EvidenceSnippet(text=match.group(0), page=page_num)]
                )

        # Try site count only
        for pattern in self.site_only_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return (
                    int(match.group(1)), None,
                    [EvidenceSnippet(text=match.group(0), page=page_num)]
                )

        return None

    def _extract_country_names(self, text: str) -> List[str]:
        """Extract country names from text."""
        countries = []

        for match in COUNTRY_PATTERN.finditer(text):
            country = match.group(1)
            # Normalize some common variations
            country_normalized = self._normalize_country(country)
            if country_normalized and country_normalized not in countries:
                countries.append(country_normalized)

        return countries

    def _normalize_country(self, country: str) -> str:
        """Normalize country name."""
        normalizations = {
            "usa": "United States",
            "us": "United States",
            "uk": "United Kingdom",
            "czechia": "Czech Republic",
            "south korea": "Korea",
            "russian federation": "Russia",
        }
        country_lower = country.lower()
        return normalizations.get(country_lower, country.title())

    def _extract_enrollment_dates(
        self,
        text: str,
        page_num: int
    ) -> Optional[Tuple[Optional[date], Optional[date]]]:
        """Extract enrollment start and end dates."""
        for pattern in self.date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                try:
                    if len(groups) == 4:
                        # Full date range: "January 2020 to December 2021"
                        start_month = self.months.get(groups[0].lower())
                        start_year = int(groups[1])
                        end_month = self.months.get(groups[2].lower())
                        end_year = int(groups[3])

                        if start_month and end_month:
                            start = date(start_year, start_month, 1)
                            end = date(end_year, end_month, 28)  # Approximate
                            return (start, end)

                    elif len(groups) == 2:
                        # Either year range or single date
                        if groups[0].isdigit() and groups[1].isdigit():
                            # Year range: "between 2020 and 2021"
                            start = date(int(groups[0]), 1, 1)
                            end = date(int(groups[1]), 12, 31)
                            return (start, end)
                        else:
                            # Single date: "January 2020"
                            month = self.months.get(groups[0].lower())
                            year = int(groups[1])
                            if month:
                                return (date(year, month, 1), None)
                except (ValueError, TypeError):
                    continue

        return None

    def _extract_from_tables(
        self,
        doc: DocumentGraph,
        result: RecruitmentFootprint
    ) -> None:
        """Extract footprint data from tables."""
        for table in doc.iter_tables():
            # Check if this looks like a demographics/sites table
            if not self._is_demographics_table(table):
                continue

            for row in table.logical_rows:
                if not row:
                    continue

                row_text = " ".join(str(cell) for cell in row).lower()

                # Extract site count from table
                if "site" in row_text or "center" in row_text:
                    numbers = re.findall(r"\d+", row_text)
                    if numbers and not result.num_sites:
                        result.num_sites = int(numbers[0])
                        result.evidence.append(EvidenceSnippet(
                            text=" ".join(str(cell) for cell in row),
                            page=table.page_num
                        ))

                # Extract country count from table
                if "countr" in row_text:
                    numbers = re.findall(r"\d+", row_text)
                    if numbers and not result.num_countries:
                        result.num_countries = int(numbers[0])

    def _is_demographics_table(self, table) -> bool:
        """Check if table contains demographic/site information."""
        keywords = [
            "site", "center", "country", "region", "location",
            "demographic", "baseline", "population"
        ]

        if table.headers:
            header_text = " ".join(str(h).lower() for h in table.headers)
            if any(kw in header_text for kw in keywords):
                return True

        # Check first column
        for row in table.logical_rows[:5]:
            if row:
                first_cell = str(row[0]).lower() if row[0] else ""
                if any(kw in first_cell for kw in keywords):
                    return True

        return False


def extract_recruitment_footprint(doc: DocumentGraph) -> RecruitmentFootprint:
    """Convenience function for footprint extraction."""
    extractor = RecruitmentFootprintExtractor()
    return extractor.extract(doc)

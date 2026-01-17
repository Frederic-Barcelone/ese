# corpus_metadata/C_generators/C13_strategy_screening.py
"""
Screening Yield Extractor - Extract CONSORT-style enrollment data.

Targets:
- Screened / randomized / enrolled / completed counts
- Screen failure reasons with counts
- Run-in period completers/failures

Sources:
- CONSORT flow diagrams (as tables or figures)
- Methods sections
- Results sections (Patient Disposition)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from A_core.A04_feasibility_models import (
    EvidenceSnippet,
    ScreenFailReason,
    ScreeningYield,
)
from B_parsing.B02_doc_graph import DocumentGraph, TableType


class ScreeningYieldExtractor:
    """
    Extract screening and enrollment data from clinical trial documents.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Patterns for enrollment numbers
        self.screened_patterns = [
            r"(\d+)\s*(?:patients?|subjects?|participants?|individuals?)\s*(?:were\s*)?screened",
            r"screened\s*[:\-=]?\s*(\d+)",
            r"(\d+)\s*screened",
            r"assessed\s+for\s+eligibility\s*[:\-=]?\s*(\d+)",
            r"(\d+)\s*assessed\s+for\s+eligibility",
        ]

        self.randomized_patterns = [
            r"(\d+)\s*(?:patients?|subjects?|participants?)\s*(?:were\s*)?randomized",
            r"randomized\s*[:\-=]?\s*(\d+)",
            r"(\d+)\s*randomized",
            r"(\d+)\s*(?:were\s*)?randomly\s+assigned",
            r"randomly\s+assigned\s*[:\-=]?\s*(\d+)",
        ]

        self.enrolled_patterns = [
            r"(\d+)\s*(?:patients?|subjects?|participants?)\s*(?:were\s*)?enrolled",
            r"enrolled\s*[:\-=]?\s*(\d+)",
            r"(\d+)\s*enrolled",
        ]

        self.completed_patterns = [
            r"(\d+)\s*(?:patients?|subjects?|participants?)\s*completed",
            r"completed\s*[:\-=]?\s*(\d+)",
            r"(\d+)\s*completed\s+(?:the\s+)?(?:study|trial|treatment)",
        ]

        self.discontinued_patterns = [
            r"(\d+)\s*(?:patients?|subjects?|participants?)\s*discontinued",
            r"discontinued\s*[:\-=]?\s*(\d+)",
            r"(\d+)\s*(?:withdrew|withdrawn)",
        ]

        self.screen_fail_patterns = [
            r"(\d+)\s*(?:patients?|subjects?|participants?)\s*(?:were\s*)?(?:excluded|screen.?failed?|not\s+eligible)",
            r"screen\s*fail(?:ure|ed)?\s*[:\-=]?\s*(\d+)",
            r"(\d+)\s*screen\s*fail(?:ure|ed)?",
            r"(\d+)\s*did\s+not\s+meet\s+(?:eligibility\s+)?criteria",
            r"excluded\s*[:\-=]?\s*(\d+)",
        ]

        # Screen failure reason patterns
        self.reason_patterns = [
            # "did not meet X criteria (n=Y)" or "X (n=Y)"
            (r"did\s+not\s+meet\s+(.+?)\s+(?:criteria\s+)?\(n\s*=\s*(\d+)\)", "criteria_not_met"),
            (r"(.+?)\s+(?:threshold|criteria)\s+not\s+met\s*\(n\s*=\s*(\d+)\)", "threshold_not_met"),
            (r"(.+?)\s*[:\-]\s*(\d+)\s*(?:patients?|subjects?)?", "general"),
            # "Y patients had X" or "Y excluded due to X"
            (r"(\d+)\s*(?:patients?|subjects?)?\s*(?:had|with|due\s+to)\s+(.+?)(?:\.|,|;|$)", "had_condition"),
            (r"(\d+)\s*excluded\s+(?:due\s+to|for|because\s+of)\s+(.+?)(?:\.|,|;|$)", "excluded_for"),
        ]

    def extract(self, doc: DocumentGraph) -> ScreeningYield:
        """Extract screening yield data from document."""
        result = ScreeningYield()

        # Extract from tables first (most reliable)
        table_data = self._extract_from_tables(doc)
        if table_data:
            result = self._merge_data(result, table_data)

        # Extract from text
        text_data = self._extract_from_text(doc)
        if text_data:
            result = self._merge_data(result, text_data)

        # Compute derived values
        if result.screened and result.randomized:
            result.screening_yield_pct = round(
                result.randomized / result.screened * 100, 1
            )

        if result.screened and result.randomized:
            if not result.screen_failed:
                result.screen_failed = result.screened - result.randomized

        return result

    def _extract_from_tables(self, doc: DocumentGraph) -> Optional[ScreeningYield]:
        """Extract from CONSORT/disposition tables."""
        result = ScreeningYield()

        for table in doc.iter_tables():
            # Check if this looks like a disposition/CONSORT table
            if not self._is_disposition_table(table):
                continue

            # Extract data from table rows
            for row_idx, row in enumerate(table.logical_rows):
                if not row:
                    continue

                row_text = " ".join(str(cell) for cell in row).lower()

                # Try to extract counts from this row
                numbers = re.findall(r"\d+", row_text)
                if not numbers:
                    continue

                # Match row content to field
                if any(kw in row_text for kw in ["screened", "assessed for eligibility"]):
                    result.screened = int(numbers[0])
                    result.evidence.append(EvidenceSnippet(
                        text=" ".join(str(cell) for cell in row),
                        page=table.page_num,
                    ))
                elif any(kw in row_text for kw in ["randomized", "randomly assigned"]):
                    result.randomized = int(numbers[0])
                    result.evidence.append(EvidenceSnippet(
                        text=" ".join(str(cell) for cell in row),
                        page=table.page_num,
                    ))
                elif "enrolled" in row_text:
                    result.enrolled = int(numbers[0])
                elif "completed" in row_text:
                    result.completed = int(numbers[0])
                elif any(kw in row_text for kw in ["discontinued", "withdrew", "withdrawal"]):
                    result.discontinued = int(numbers[0])
                elif any(kw in row_text for kw in ["screen fail", "excluded", "not eligible", "did not meet"]):
                    result.screen_failed = int(numbers[0])

                    # Try to extract reason
                    reason = self._extract_reason_from_row(row_text)
                    if reason:
                        result.screen_fail_reasons.append(ScreenFailReason(
                            reason=reason,
                            count=int(numbers[0]),
                            evidence=[EvidenceSnippet(
                                text=" ".join(str(cell) for cell in row),
                                page=table.page_num,
                            )]
                        ))

        return result if result.screened or result.randomized else None

    def _is_disposition_table(self, table) -> bool:
        """Check if table is a patient disposition/CONSORT table."""
        # Check headers and first column for keywords
        keywords = [
            "screened", "enrolled", "randomized", "disposition",
            "consort", "flow", "patient", "subject", "participant",
            "eligible", "excluded", "completed", "discontinued"
        ]

        # Check table headers
        if table.headers:
            header_text = " ".join(str(h).lower() for h in table.headers)
            if any(kw in header_text for kw in keywords):
                return True

        # Check first few rows
        for row in table.logical_rows[:5]:
            if row:
                row_text = " ".join(str(cell).lower() for cell in row)
                if any(kw in row_text for kw in keywords):
                    return True

        return False

    def _extract_reason_from_row(self, row_text: str) -> Optional[str]:
        """Extract screen failure reason from table row."""
        # Common patterns
        patterns = [
            r"did\s+not\s+meet\s+(.+?)\s+(?:criteria|threshold)",
            r"(.+?)\s+(?:criteria|threshold)\s+not\s+met",
            r"excluded\s+(?:due\s+to|for)\s+(.+)",
            r"ineligible\s+(?:due\s+to|for)\s+(.+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, row_text, re.IGNORECASE)
            if match:
                reason = match.group(1).strip()
                # Clean up
                reason = re.sub(r"\s*\(.*?\)\s*", "", reason)
                reason = reason.strip(" .,;:")
                if reason and len(reason) > 3:
                    return reason

        return None

    def _extract_from_text(self, doc: DocumentGraph) -> Optional[ScreeningYield]:
        """Extract from running text."""
        result = ScreeningYield()

        # Target sections
        target_sections = ["methods", "results", "patient", "disposition", "enrollment"]

        for block in doc.iter_linear_blocks(skip_header_footer=True):
            text = block.text
            if not text:
                continue

            # Screened
            if not result.screened:
                for pattern in self.screened_patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        result.screened = int(match.group(1))
                        result.evidence.append(EvidenceSnippet(
                            text=text[:200],
                            page=block.page_num,
                        ))
                        break

            # Randomized
            if not result.randomized:
                for pattern in self.randomized_patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        result.randomized = int(match.group(1))
                        result.evidence.append(EvidenceSnippet(
                            text=text[:200],
                            page=block.page_num,
                        ))
                        break

            # Enrolled
            if not result.enrolled:
                for pattern in self.enrolled_patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        result.enrolled = int(match.group(1))
                        break

            # Completed
            if not result.completed:
                for pattern in self.completed_patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        result.completed = int(match.group(1))
                        break

            # Screen failures with reasons
            self._extract_screen_fail_reasons(text, block.page_num, result)

        return result if result.screened or result.randomized else None

    def _extract_screen_fail_reasons(
        self,
        text: str,
        page_num: int,
        result: ScreeningYield
    ) -> None:
        """Extract screen failure reasons from text."""
        # Look for common patterns
        # Pattern: "X patients did not meet Y criteria"
        pattern1 = r"(\d+)\s*(?:patients?|subjects?)?\s*did\s+not\s+meet\s+(.+?)\s+(?:criteria|threshold|requirement)"
        for match in re.finditer(pattern1, text, re.IGNORECASE):
            count = int(match.group(1))
            reason = match.group(2).strip()
            if reason and len(reason) > 2:
                result.screen_fail_reasons.append(ScreenFailReason(
                    reason=f"Did not meet {reason}",
                    count=count,
                    evidence=[EvidenceSnippet(text=match.group(0), page=page_num)]
                ))

        # Pattern: "excluded due to X (n=Y)" or "(Y patients)"
        pattern2 = r"excluded\s+(?:due\s+to|for|because\s+of)\s+(.+?)\s*(?:\(n\s*=\s*(\d+)\)|\((\d+)\s*patients?\))"
        for match in re.finditer(pattern2, text, re.IGNORECASE):
            reason = match.group(1).strip()
            count = int(match.group(2) or match.group(3))
            if reason and len(reason) > 2:
                result.screen_fail_reasons.append(ScreenFailReason(
                    reason=reason,
                    count=count,
                    evidence=[EvidenceSnippet(text=match.group(0), page=page_num)]
                ))

        # Pattern: "serum X threshold (n=Y)"
        pattern3 = r"((?:serum\s+)?[A-Za-z0-9]+\s+(?:threshold|level|value))\s*\(n\s*=\s*(\d+)\)"
        for match in re.finditer(pattern3, text, re.IGNORECASE):
            reason = match.group(1).strip()
            count = int(match.group(2))
            result.screen_fail_reasons.append(ScreenFailReason(
                reason=reason,
                count=count,
                category="lab_threshold",
                evidence=[EvidenceSnippet(text=match.group(0), page=page_num)]
            ))

    def _merge_data(
        self,
        existing: ScreeningYield,
        new: ScreeningYield
    ) -> ScreeningYield:
        """Merge two ScreeningYield objects, preferring non-None values."""
        return ScreeningYield(
            screened=new.screened or existing.screened,
            screen_failed=new.screen_failed or existing.screen_failed,
            randomized=new.randomized or existing.randomized,
            enrolled=new.enrolled or existing.enrolled,
            completed=new.completed or existing.completed,
            discontinued=new.discontinued or existing.discontinued,
            run_in_completers=new.run_in_completers or existing.run_in_completers,
            run_in_failures=new.run_in_failures or existing.run_in_failures,
            screen_fail_reasons=existing.screen_fail_reasons + new.screen_fail_reasons,
            evidence=existing.evidence + new.evidence,
        )


def extract_screening_yield(doc: DocumentGraph) -> ScreeningYield:
    """Convenience function for screening yield extraction."""
    extractor = ScreeningYieldExtractor()
    return extractor.extract(doc)

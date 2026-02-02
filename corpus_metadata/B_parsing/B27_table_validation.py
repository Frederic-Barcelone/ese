# corpus_metadata/B_parsing/B27_table_validation.py
"""
Table validation and false positive filtering.

Provides:
- Prose text detection (multi-column article text misclassified as tables)
- Definition/glossary table salvaging
- Table structure validation

Extracted from B03_table_extractor.py to reduce file size.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

# Minimum confidence threshold for table detection (0.0-1.0)
MIN_TABLE_CONFIDENCE = 0.5

# Minimum requirements for a valid table
MIN_TABLE_ROWS = 2  # At least header + 1 data row
MIN_TABLE_COLS = 2  # At least 2 columns


def is_valid_table(
    rows: List[List[str]],
    html: Optional[str],
) -> Tuple[bool, Optional[str]]:
    """
    Validate that the detected element is actually a table, not misclassified text.

    Returns (False, reason) for false positives like:
    - Multi-column article text
    - Single-column lists
    - Elements with no proper table structure
    - Tables that are mostly prose text

    Returns (True, None) for valid tables.
    Returns (True, "definition_table_salvaged ...") for tables salvaged as definition tables.

    Args:
        rows: List of rows, where each row is a list of cell strings
        html: Optional HTML source of the table

    Returns:
        Tuple of (is_valid, reason_or_salvage_info)
    """
    # Must have minimum rows
    if len(rows) < MIN_TABLE_ROWS:
        return False, f"too few rows ({len(rows)} < {MIN_TABLE_ROWS})"

    # Must have minimum columns
    if not rows or len(rows[0]) < MIN_TABLE_COLS:
        col_count = len(rows[0]) if rows else 0
        return False, f"too few columns ({col_count} < {MIN_TABLE_COLS})"

    # Check that HTML actually contains table structure
    if html:
        # Must have actual table tags (not just text wrapped in table tags)
        if "<tr" not in html.lower() or "<td" not in html.lower():
            return False, "missing HTML table tags (<tr>/<td>)"

        # Count actual table cells vs text length ratio
        # A real table has structured short cells, not long paragraphs
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        cells = soup.find_all(["td", "th"])

        if not cells:
            return False, "no table cells found in HTML"

        # Calculate average cell text length
        cell_texts = [cell.get_text(strip=True) for cell in cells]
        if cell_texts:
            avg_cell_length = sum(len(t) for t in cell_texts) / len(cell_texts)

            # Determine column count for adaptive thresholds
            # 2-column tables (definition/glossary) get more lenient thresholds
            num_cols = len(rows[0]) if rows else 0
            is_two_col = num_cols == 2

            # Adaptive thresholds based on table structure
            # 2-column tables often have long definitions, so we allow longer cells
            max_avg_cell_length = 200 if is_two_col else 150
            max_long_cell_pct = 0.50 if is_two_col else 0.30
            max_prose_pct = 0.25 if is_two_col else 0.15

            # If average cell length is very long, likely not a table
            # Real table cells are typically short (numbers, names, codes)
            # But definition tables can have longer explanatory text
            if avg_cell_length > max_avg_cell_length:
                # Before rejecting, check if it's a definition table
                is_def_table, salvage_reason = is_definition_table_candidate(rows, html)
                if is_def_table:
                    return True, salvage_reason
                return False, f"avg cell length too long ({avg_cell_length:.0f} > {max_avg_cell_length} chars) - likely prose"

            # Check for paragraph-like content (multiple sentences in many cells)
            long_text_cells = sum(1 for t in cell_texts if len(t) > 80)
            long_pct = long_text_cells / len(cell_texts) * 100
            if long_text_cells / len(cell_texts) > max_long_cell_pct:
                # Before rejecting, check if it's a definition table
                is_def_table, salvage_reason = is_definition_table_candidate(rows, html)
                if is_def_table:
                    return True, salvage_reason
                return False, f"too many long cells ({long_pct:.0f}% > 80 chars) - likely paragraph text"

            # Check for sentence-like content (periods followed by capital letters)
            # This indicates prose text, not tabular data
            prose_indicators = 0
            for text in cell_texts:
                # Count cells with multiple sentences (prose indicators)
                if '. ' in text and len(text) > 50:
                    # Check if it's a sentence pattern (period followed by capital)
                    if re.search(r'\. [A-Z]', text):
                        prose_indicators += 1

            # If too many cells look like prose, reject
            prose_pct = prose_indicators / len(cell_texts) * 100 if cell_texts else 0
            if len(cell_texts) > 0 and prose_indicators / len(cell_texts) > max_prose_pct:
                # Before rejecting, check if it's a definition table
                is_def_table, salvage_reason = is_definition_table_candidate(rows, html)
                if is_def_table:
                    return True, salvage_reason
                return False, f"too many prose cells ({prose_pct:.0f}% contain sentences) - likely article text"

            # Check total text length - real tables shouldn't have huge amounts of text
            # Relaxed for 2-column tables which may have lengthy definitions
            total_text = sum(len(t) for t in cell_texts)
            max_total_text = 5000 if is_two_col else 3000
            min_cells_for_text = 10 if is_two_col else 15
            if total_text > max_total_text and len(cell_texts) < min_cells_for_text:
                # Before rejecting, check if it's a definition table
                is_def_table, salvage_reason = is_definition_table_candidate(rows, html)
                if is_def_table:
                    return True, salvage_reason
                return False, f"too much text ({total_text} chars) in few cells ({len(cell_texts)}) - likely prose block"

            # Check for multi-column layout (article text split into columns)
            # If cells have very similar lengths and are long, it's likely prose
            if len(cell_texts) >= 4:
                lengths = [len(t) for t in cell_texts if len(t) > 20]
                if lengths:
                    avg_len = sum(lengths) / len(lengths)
                    # If most cells are similar length and long (>60 chars), likely prose columns
                    similar_long = sum(1 for length in lengths if abs(length - avg_len) < avg_len * 0.3 and length > 60)
                    if similar_long > len(lengths) * 0.6:
                        # SALVAGE CHECK: Before rejecting, check if it's a definition table
                        # Definition tables have 2 columns: short acronyms + long definitions
                        is_def_table, salvage_reason = is_definition_table_candidate(rows, html)
                        if is_def_table:
                            # Salvage this as a definition table
                            return True, salvage_reason
                        similar_pct = similar_long / len(lengths) * 100
                        return False, f"multi-column prose layout detected ({similar_pct:.0f}% cells have similar long text)"

    # Validate column consistency across rows
    col_counts = [len(row) for row in rows]
    if col_counts:
        # Allow some variation (for merged cells) but not wild inconsistency
        min_cols = min(col_counts)
        max_cols = max(col_counts)
        if max_cols > 0 and min_cols / max_cols < 0.5:  # >50% column count variation
            return False, f"inconsistent column counts ({min_cols}-{max_cols} cols across rows)"

    return True, None


def is_definition_table_candidate(
    rows: List[List[str]],
    html: str,
) -> Tuple[bool, Optional[str]]:
    """
    Check if a table is a definition/glossary table that should be salvaged.

    Definition tables have the signature:
    - 2 columns
    - Column 1: Short text (acronyms/abbreviations) with median <= 10 chars
    - Column 2: Long text (definitions) with median >= 40 chars
    - High ratio of acronym-like patterns in column 1

    This allows us to salvage tables that would otherwise be rejected as
    "multi-column prose" because their definitions are long.

    Args:
        rows: List of rows, where each row is a list of cell strings
        html: HTML source of the table

    Returns:
        Tuple of (is_definition_table, salvage_reason)
    """
    if not rows:
        return False, None

    # Must have exactly 2 columns (term + definition pattern)
    num_cols = len(rows[0]) if rows else 0
    if num_cols != 2:
        return False, None

    # Extract column 1 (terms) and column 2 (definitions) from all rows
    col1_texts = []
    col2_texts = []
    for row in rows:
        if len(row) >= 2:
            col1_texts.append(row[0].strip())
            col2_texts.append(row[1].strip())

    if not col1_texts or not col2_texts:
        return False, None

    # Calculate median lengths for each column
    col1_lengths = sorted(len(t) for t in col1_texts if t)
    col2_lengths = sorted(len(t) for t in col2_texts if t)

    if not col1_lengths or not col2_lengths:
        return False, None

    col1_median = col1_lengths[len(col1_lengths) // 2]
    col2_median = col2_lengths[len(col2_lengths) // 2]

    # Definition table signature: short col1 (<=15 chars), longer col2 (>=30 chars)
    # The ratio between them should be significant (col2 at least 2x col1)
    if col1_median > 15:
        return False, None
    if col2_median < 30:
        return False, None
    if col2_median < col1_median * 2:
        return False, None

    # Check for acronym-like patterns in column 1
    # Acronyms: ALL CAPS, mixed case abbreviations, or short words
    acronym_pattern = re.compile(r'^[A-Z][A-Z0-9\-/+]{0,10}$|^[A-Z][a-z]{0,3}[A-Z]')
    acronym_count = sum(1 for t in col1_texts if t and acronym_pattern.match(t))
    acronym_ratio = acronym_count / len(col1_texts) if col1_texts else 0

    # At least 30% should look like acronyms/abbreviations
    if acronym_ratio < 0.3:
        # Also check for short terms (even if not strict acronym pattern)
        short_count = sum(1 for t in col1_texts if t and len(t) <= 12)
        short_ratio = short_count / len(col1_texts) if col1_texts else 0
        if short_ratio < 0.6:
            return False, None

    reason = (
        f"definition_table_salvaged: col1_median={col1_median}, "
        f"col2_median={col2_median}, acronym_ratio={acronym_ratio:.1%}"
    )
    return True, reason


__all__ = [
    "MIN_TABLE_CONFIDENCE",
    "MIN_TABLE_ROWS",
    "MIN_TABLE_COLS",
    "is_valid_table",
    "is_definition_table_candidate",
]

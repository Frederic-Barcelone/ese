# corpus_metadata/B_parsing/B21_filename_generator.py
"""
Layout-Aware Filename Generator.

Generates filenames that encode:
- Document name
- Visual type (figure/table)
- Page number
- Layout pattern (full, 2col, 2col-fullbot, fulltop-2col)
- Position (L, R, F)
- Index (for multiple visuals in same position)

Format: {doc}_{type}_p{page}_{layout}_{position}_{index}.png
"""
from __future__ import annotations

import re

from B_parsing.B20_zone_expander import ExpandedVisual


def sanitize_name(name: str) -> str:
    """
    Sanitize document name for use in filename.

    Removes/replaces special characters, spaces, and extensions.

    Args:
        name: Raw document name

    Returns:
        Sanitized name safe for filenames
    """
    # Remove file extension
    name = re.sub(r'\.(pdf|PDF)$', '', name)

    # Replace spaces and special chars with underscores (keep alphanumeric, underscore, dash)
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)

    # Collapse multiple underscores
    name = re.sub(r'_+', '_', name)

    # Remove leading/trailing underscores
    name = name.strip('_')

    # Limit length
    if len(name) > 50:
        name = name[:50]

    return name or "document"


def generate_visual_filename(
    doc_name: str,
    page_num: int,
    visual: ExpandedVisual,
    index: int,
    extension: str = "png",
) -> str:
    """
    Generate layout-aware filename for a visual.

    Format: {doc}_{type}_p{page}_{layout}_{position}_{index}.{ext}

    Args:
        doc_name: Document name (will be sanitized)
        page_num: Page number (1-indexed)
        visual: Expanded visual with layout/position codes
        index: Index for multiple visuals (1-indexed)
        extension: File extension (default: png)

    Returns:
        Generated filename

    Examples:
        article_figure_p3_2col_L_1.png
        paper_table_p5_2col-fullbot_F_1.png
    """
    safe_name = sanitize_name(doc_name)
    visual_type = visual.zone.visual_type.lower()
    layout_code = visual.layout_code
    position_code = visual.position_code

    return f"{safe_name}_{visual_type}_p{page_num}_{layout_code}_{position_code}_{index}.{extension}"


__all__ = [
    # Functions
    "sanitize_name",
    "generate_visual_filename",
]

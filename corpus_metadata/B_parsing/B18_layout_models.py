# corpus_metadata/B_parsing/B18_layout_models.py
"""Backward-compat re-exports. Canonical home: A_core.A24_layout_models."""
from A_core.A24_layout_models import (  # noqa: F401
    LayoutPattern,
    VisualPosition,
    VisualZone,
    PageLayout,
)

__all__ = ["LayoutPattern", "VisualPosition", "VisualZone", "PageLayout"]

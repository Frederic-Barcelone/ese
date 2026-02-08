# corpus_metadata/B_parsing/B02_doc_graph.py
"""Backward-compat re-exports. Canonical home: A_core.A23_doc_graph_models."""
from A_core.A23_doc_graph_models import (  # noqa: F401
    ContentRole,
    TableType,
    ImageType,
    TextBlock,
    ImageBlock,
    TableCell,
    Table,
    Page,
    DocumentGraph,
)

__all__ = [
    "ContentRole",
    "TableType",
    "ImageType",
    "TextBlock",
    "ImageBlock",
    "TableCell",
    "Table",
    "Page",
    "DocumentGraph",
]

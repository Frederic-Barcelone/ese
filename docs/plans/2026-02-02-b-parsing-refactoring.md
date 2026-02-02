# B_parsing Folder Refactoring Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Consolidate B_parsing files to follow sequential B## naming convention, eliminate submodule naming (B01a, B03b, etc.), fix naming conflicts, and add missing `__all__` exports.

**Architecture:** Consolidate related submodules into their parent files where they fit, or rename to new sequential numbers. The B17 conflict (two files with same prefix) must be resolved first. Rendering functionality in B03b and B14 should be deduplicated.

**Tech Stack:** Python, PyMuPDF (fitz), Docling, Anthropic API

---

## Current Issues Summary

| Issue | Files Affected | Resolution |
|-------|----------------|------------|
| B17 naming conflict | B17_vlm_detector.py, B17_document_resolver.py | Rename vlm_detector → B23 |
| Submodule naming | B01a-d, B03a-c, B04a-b | Consolidate or rename to sequential |
| Duplicate rendering | B03b, B14 | Keep B14, update B03 imports |
| Missing `__all__` | B18, B19, B20, B21 | Add exports |

## New File Structure (Proposed)

| Current | New | Contents |
|---------|-----|----------|
| B01_pdf_to_docgraph.py | B01_pdf_to_docgraph.py | Main PDF parser |
| B01a_text_helpers.py | B23_text_helpers.py | Text utilities (widely used) |
| B01b_native_figure_extraction.py | B24_native_figure_extraction.py | Native PDF figure extraction |
| B01c_legacy_ordering.py | B25_legacy_ordering.py | Legacy block ordering |
| B01d_repetition_inference.py | B26_repetition_inference.py | Header/footer repetition |
| B03_table_extractor.py | B03_table_extractor.py | Main table extractor |
| B03a_table_validation.py | B27_table_validation.py | Table validation |
| B03b_table_rendering.py | REMOVE | Duplicate of B14 |
| B03c_docling_backend.py | B28_docling_backend.py | Docling backend |
| B04_column_ordering.py | B04_column_ordering.py | Main column ordering |
| B04a_column_detection.py | B29_column_detection.py | Column detection |
| B04b_xy_cut_ordering.py | B30_xy_cut_ordering.py | XY-Cut ordering |
| B17_vlm_detector.py | B31_vlm_detector.py | VLM visual detection |
| B17_document_resolver.py | B17_document_resolver.py | Keep (used by pipeline) |

---

### Task 1: Fix B17 Naming Conflict

**Files:**
- Rename: `B_parsing/B17_vlm_detector.py` → `B_parsing/B31_vlm_detector.py`

**Step 1: Check for imports of B17_vlm_detector**

Run: `grep -r "B17_vlm" corpus_metadata/`
Expected: No imports (file is unused currently)

**Step 2: Rename the file**

```bash
git mv corpus_metadata/B_parsing/B17_vlm_detector.py corpus_metadata/B_parsing/B31_vlm_detector.py
```

**Step 3: Update module docstring header comment**

Change line 1 from:
```python
# corpus_metadata/B_parsing/B17_vlm_detector.py
```
to:
```python
# corpus_metadata/B_parsing/B31_vlm_detector.py
```

**Step 4: Commit**

```bash
git add corpus_metadata/B_parsing/
git commit -m "$(cat <<'EOF'
refactor(B_parsing): rename B17_vlm_detector to B31 to fix naming conflict

B17_document_resolver.py is actively used by the visual pipeline.
B17_vlm_detector.py was a naming conflict - now B31_vlm_detector.py.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Rename B01 Submodules to Sequential Numbers

**Files:**
- Rename: `B01a_text_helpers.py` → `B23_text_helpers.py`
- Rename: `B01b_native_figure_extraction.py` → `B24_native_figure_extraction.py`
- Rename: `B01c_legacy_ordering.py` → `B25_legacy_ordering.py`
- Rename: `B01d_repetition_inference.py` → `B26_repetition_inference.py`
- Modify: `B01_pdf_to_docgraph.py` (update imports)
- Modify: `B01b→B24` internal import

**Step 1: Rename files**

```bash
git mv corpus_metadata/B_parsing/B01a_text_helpers.py corpus_metadata/B_parsing/B23_text_helpers.py
git mv corpus_metadata/B_parsing/B01b_native_figure_extraction.py corpus_metadata/B_parsing/B24_native_figure_extraction.py
git mv corpus_metadata/B_parsing/B01c_legacy_ordering.py corpus_metadata/B_parsing/B25_legacy_ordering.py
git mv corpus_metadata/B_parsing/B01d_repetition_inference.py corpus_metadata/B_parsing/B26_repetition_inference.py
```

**Step 2: Update imports in B01_pdf_to_docgraph.py**

Change:
```python
from B_parsing.B01a_text_helpers import (
```
to:
```python
from B_parsing.B23_text_helpers import (
```

Change:
```python
from B_parsing.B01b_native_figure_extraction import (
```
to:
```python
from B_parsing.B24_native_figure_extraction import (
```

Change:
```python
from B_parsing.B01c_legacy_ordering import (
```
to:
```python
from B_parsing.B25_legacy_ordering import (
```

Change:
```python
from B_parsing.B01d_repetition_inference import (
```
to:
```python
from B_parsing.B26_repetition_inference import (
```

**Step 3: Update imports in B24_native_figure_extraction.py**

Change:
```python
from B_parsing.B01a_text_helpers import PERCENTAGE_PATTERN
```
to:
```python
from B_parsing.B23_text_helpers import PERCENTAGE_PATTERN
```

**Step 4: Update imports in B26_repetition_inference.py**

Change:
```python
from B_parsing.B01a_text_helpers import (
```
to:
```python
from B_parsing.B23_text_helpers import (
```

**Step 5: Update module docstring headers in all renamed files**

**Step 6: Run tests to verify**

Run: `cd corpus_metadata && python -m pytest tests/ -v -x`
Expected: PASS

**Step 7: Commit**

```bash
git add corpus_metadata/B_parsing/
git commit -m "$(cat <<'EOF'
refactor(B_parsing): rename B01a-d submodules to B23-B26

Follow sequential naming convention instead of letter suffixes:
- B01a_text_helpers.py → B23_text_helpers.py
- B01b_native_figure_extraction.py → B24_native_figure_extraction.py
- B01c_legacy_ordering.py → B25_legacy_ordering.py
- B01d_repetition_inference.py → B26_repetition_inference.py

Updated all internal imports.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Deduplicate B03b_table_rendering (Merge to B14)

**Files:**
- Modify: `B03_table_extractor.py` (update imports to use B14)
- Modify: `J_export/J01_export_handlers.py` (update imports to use B14)
- Delete: `B03b_table_rendering.py`

**Step 1: Compare B03b and B14 functions**

B03b provides: `find_table_bbox_pymupdf`, `render_table_as_image`, `render_full_page`, `render_multipage_table`
B14 provides: `render_visual`, `render_visual_region`, `render_full_page`, `render_multipage_visual`

The functions overlap in purpose. B14 is the newer, cleaner implementation.
B03b's `find_table_bbox_pymupdf` is unique - must be added to B14.

**Step 2: Add find_table_bbox_pymupdf to B14_visual_renderer.py**

Add the following function to B14 (before `render_visual_region`):

```python
def find_table_bbox_pymupdf(
    file_path: str,
    page_num: int,
    hint_bbox: Optional[Tuple[float, float, float, float]] = None,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Use PyMuPDF's table detection to find actual table boundaries.

    Args:
        file_path: Path to PDF
        page_num: 1-indexed page number
        hint_bbox: Optional hint bbox to find the nearest table

    Returns:
        Bbox tuple (x0, y0, x1, y1) or None if no table found
    """
    try:
        doc = fitz.open(file_path)
        if page_num < 1 or page_num > len(doc):
            doc.close()
            return None

        page = doc[page_num - 1]
        page_width = page.rect.width
        page_height = page.rect.height

        # Use PyMuPDF's table finder
        tables = page.find_tables()

        if not tables or len(tables.tables) == 0:
            doc.close()
            return None

        # If we have a hint bbox, find the closest table
        if hint_bbox:
            # Check if hint_bbox is in a different coordinate space
            hint_out_of_bounds = (
                hint_bbox[2] > page_width * 1.5 or
                hint_bbox[3] > page_height * 1.5
            )

            if hint_out_of_bounds:
                scale_x = page_width / max(hint_bbox[2], page_width)
                scale_y = page_height / max(hint_bbox[3], page_height)
                scale = min(scale_x, scale_y)
                scaled_hint = (
                    hint_bbox[0] * scale,
                    hint_bbox[1] * scale,
                    hint_bbox[2] * scale,
                    hint_bbox[3] * scale,
                )
                hint_center_x = (scaled_hint[0] + scaled_hint[2]) / 2
                hint_center_y = (scaled_hint[1] + scaled_hint[3]) / 2
            else:
                hint_center_x = (hint_bbox[0] + hint_bbox[2]) / 2
                hint_center_y = (hint_bbox[1] + hint_bbox[3]) / 2

            best_table = None
            best_distance = float('inf')

            for table in tables.tables:
                table_bbox = table.bbox
                table_center_x = (table_bbox[0] + table_bbox[2]) / 2
                table_center_y = (table_bbox[1] + table_bbox[3]) / 2

                distance = ((hint_center_x - table_center_x) ** 2 +
                           (hint_center_y - table_center_y) ** 2) ** 0.5

                if distance < best_distance:
                    best_distance = distance
                    best_table = table

            if best_table:
                doc.close()
                return tuple(best_table.bbox)

        # Otherwise return the largest table
        largest_table = max(
            tables.tables,
            key=lambda t: (t.bbox[2] - t.bbox[0]) * (t.bbox[3] - t.bbox[1])
        )
        doc.close()
        return tuple(largest_table.bbox)

    except Exception as e:
        logger.warning("PyMuPDF table detection failed: %s", e)
        return None
```

**Step 3: Add to B14's __all__**

Add `"find_table_bbox_pymupdf"` to the `__all__` list.

**Step 4: Update B03_table_extractor.py imports**

Change:
```python
from B_parsing.B03b_table_rendering import (
    render_table_as_image,
    render_multipage_table,
)
```
to:
```python
from B_parsing.B14_visual_renderer import (
    render_visual_region,
    render_multipage_visual,
    find_table_bbox_pymupdf,
)
```

Note: The function signatures differ slightly. B03 uses `render_table_as_image` with specific parameters. We need to create a thin wrapper or update call sites.

**Step 5: Update J01_export_handlers.py imports**

Change:
```python
from B_parsing.B03b_table_rendering import find_table_bbox_pymupdf
```
to:
```python
from B_parsing.B14_visual_renderer import find_table_bbox_pymupdf
```

**Step 6: Delete B03b_table_rendering.py**

```bash
git rm corpus_metadata/B_parsing/B03b_table_rendering.py
```

**Step 7: Run tests to verify**

Run: `cd corpus_metadata && python -m pytest tests/ -v -x`
Expected: PASS

**Step 8: Commit**

```bash
git add corpus_metadata/
git commit -m "$(cat <<'EOF'
refactor(B_parsing): deduplicate table rendering, remove B03b

Consolidated rendering functions into B14_visual_renderer.py:
- Added find_table_bbox_pymupdf from B03b
- Removed B03b_table_rendering.py (duplicate functionality)
- Updated imports in B03_table_extractor.py and J01_export_handlers.py

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Rename B03 Submodules to Sequential Numbers

**Files:**
- Rename: `B03a_table_validation.py` → `B27_table_validation.py`
- Rename: `B03c_docling_backend.py` → `B28_docling_backend.py`
- Modify: `B03_table_extractor.py` (update imports)
- Modify: `H_pipeline/H01_component_factory.py` (update imports)
- Modify: `tests/test_parsing/test_docling_backend.py` (update imports)

**Step 1: Rename files**

```bash
git mv corpus_metadata/B_parsing/B03a_table_validation.py corpus_metadata/B_parsing/B27_table_validation.py
git mv corpus_metadata/B_parsing/B03c_docling_backend.py corpus_metadata/B_parsing/B28_docling_backend.py
```

**Step 2: Update imports in B03_table_extractor.py**

Change:
```python
from B_parsing.B03c_docling_backend import DoclingTableExtractor
```
to:
```python
from B_parsing.B28_docling_backend import DoclingTableExtractor
```

Change:
```python
from B_parsing.B03a_table_validation import MIN_TABLE_COLS
```
to:
```python
from B_parsing.B27_table_validation import MIN_TABLE_COLS
```

**Step 3: Update imports in H01_component_factory.py**

Change:
```python
from B_parsing.B03c_docling_backend import DOCLING_AVAILABLE
```
to:
```python
from B_parsing.B28_docling_backend import DOCLING_AVAILABLE
```

**Step 4: Update imports in test_docling_backend.py**

Change all occurrences of:
```python
from B_parsing.B03c_docling_backend import
```
to:
```python
from B_parsing.B28_docling_backend import
```

**Step 5: Update module docstring headers**

**Step 6: Run tests**

Run: `cd corpus_metadata && python -m pytest tests/ -v -x`
Expected: PASS

**Step 7: Commit**

```bash
git add corpus_metadata/
git commit -m "$(cat <<'EOF'
refactor(B_parsing): rename B03a,c submodules to B27-B28

Follow sequential naming convention:
- B03a_table_validation.py → B27_table_validation.py
- B03c_docling_backend.py → B28_docling_backend.py

Updated all imports in B03, H01, and tests.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Rename B04 Submodules to Sequential Numbers

**Files:**
- Rename: `B04a_column_detection.py` → `B29_column_detection.py`
- Rename: `B04b_xy_cut_ordering.py` → `B30_xy_cut_ordering.py`
- Modify: `B04_column_ordering.py` (update imports)
- Modify: `B30_xy_cut_ordering.py` (update internal import)

**Step 1: Rename files**

```bash
git mv corpus_metadata/B_parsing/B04a_column_detection.py corpus_metadata/B_parsing/B29_column_detection.py
git mv corpus_metadata/B_parsing/B04b_xy_cut_ordering.py corpus_metadata/B_parsing/B30_xy_cut_ordering.py
```

**Step 2: Update imports in B04_column_ordering.py**

Change:
```python
from B_parsing.B04a_column_detection import (
```
to:
```python
from B_parsing.B29_column_detection import (
```

Change:
```python
from B_parsing.B04b_xy_cut_ordering import (
```
to:
```python
from B_parsing.B30_xy_cut_ordering import (
```

**Step 3: Update imports in B30_xy_cut_ordering.py**

Change TYPE_CHECKING import:
```python
from B_parsing.B04a_column_detection import PageStats
```
to:
```python
from B_parsing.B29_column_detection import PageStats
```

Change runtime import:
```python
from B_parsing.B04a_column_detection import detect_l_shaped_regions
```
to:
```python
from B_parsing.B29_column_detection import detect_l_shaped_regions
```

**Step 4: Update module docstring headers**

**Step 5: Run tests**

Run: `cd corpus_metadata && python -m pytest tests/ -v -x`
Expected: PASS

**Step 6: Commit**

```bash
git add corpus_metadata/B_parsing/
git commit -m "$(cat <<'EOF'
refactor(B_parsing): rename B04a,b submodules to B29-B30

Follow sequential naming convention:
- B04a_column_detection.py → B29_column_detection.py
- B04b_xy_cut_ordering.py → B30_xy_cut_ordering.py

Updated all imports.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Add Missing __all__ Exports

**Files:**
- Modify: `B18_layout_models.py`
- Modify: `B19_layout_analyzer.py`
- Modify: `B20_zone_expander.py`
- Modify: `B21_filename_generator.py`

**Step 1: Read each file to identify public exports**

**Step 2: Add __all__ to B18_layout_models.py**

```python
__all__ = [
    # Add all public classes and functions
]
```

**Step 3: Add __all__ to B19_layout_analyzer.py**

```python
__all__ = [
    # Add all public classes and functions
]
```

**Step 4: Add __all__ to B20_zone_expander.py**

```python
__all__ = [
    # Add all public classes and functions
]
```

**Step 5: Add __all__ to B21_filename_generator.py**

```python
__all__ = [
    # Add all public classes and functions
]
```

**Step 6: Run tests**

Run: `cd corpus_metadata && python -m pytest tests/ -v -x`
Expected: PASS

**Step 7: Commit**

```bash
git add corpus_metadata/B_parsing/
git commit -m "$(cat <<'EOF'
refactor(B_parsing): add missing __all__ exports to B18-B21

Added explicit __all__ exports for better API clarity:
- B18_layout_models.py
- B19_layout_analyzer.py
- B20_zone_expander.py
- B21_filename_generator.py

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Update __init__.py and Final Verification

**Files:**
- Modify: `B_parsing/__init__.py`

**Step 1: Update __init__.py docstring**

Update to reflect new file numbers:
```python
"""
B_parsing: PDF parsing and document structure extraction.

Provides:
- PDF to DocumentGraph conversion (B01)
- Document graph data structures (B02)
- Table extraction and validation (B03, B27, B28)
- Column layout detection and reading order (B04, B29, B30)
- Section and negation detection (B05, B07)
- Confidence scoring (B06)
- Eligibility parsing (B08)
- Native figure extraction (B09-B11, B24)
- Visual pipeline (B12-B17)
- Layout analysis (B18-B21)
- DocLayout detection (B22)
- Text utilities (B23)
- Legacy ordering (B25)
- Repetition inference (B26)
- VLM detection (B31)
"""
```

**Step 2: Run full test suite**

Run: `cd corpus_metadata && python -m pytest tests/ -v`
Expected: All tests PASS

**Step 3: Run type checking**

Run: `cd corpus_metadata && mypy .`
Expected: No errors

**Step 4: Run linting**

Run: `cd corpus_metadata && ruff check .`
Expected: No errors

**Step 5: Commit**

```bash
git add corpus_metadata/B_parsing/__init__.py
git commit -m "$(cat <<'EOF'
docs(B_parsing): update __init__.py with new file structure

Updated module docstring to reflect renamed files after refactoring.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Final File Structure

After all tasks complete:

```
B_parsing/
├── __init__.py
├── B01_pdf_to_docgraph.py      # Main PDF parser
├── B02_doc_graph.py            # DocumentGraph model
├── B03_table_extractor.py      # Table extraction
├── B04_column_ordering.py      # Column ordering main
├── B05_section_detector.py     # Section detection
├── B06_confidence.py           # Confidence scoring
├── B07_negation.py             # Negation detection
├── B08_eligibility_parser.py   # Eligibility parsing
├── B09_pdf_native_figures.py   # Native PDF figures
├── B10_caption_detector.py     # Caption detection
├── B11_extraction_resolver.py  # Extraction resolver
├── B12_visual_pipeline.py      # Visual pipeline
├── B13_visual_detector.py      # Visual detection
├── B14_visual_renderer.py      # Visual rendering (consolidated)
├── B15_caption_extractor.py    # Caption extraction
├── B16_triage.py               # Triage logic
├── B17_document_resolver.py    # Document resolution
├── B18_layout_models.py        # Layout models
├── B19_layout_analyzer.py      # Layout analysis
├── B20_zone_expander.py        # Zone expansion
├── B21_filename_generator.py   # Filename generation
├── B22_doclayout_detector.py   # DocLayout YOLO detection
├── B23_text_helpers.py         # Text utilities (was B01a)
├── B24_native_figure_extraction.py  # Native figures (was B01b)
├── B25_legacy_ordering.py      # Legacy ordering (was B01c)
├── B26_repetition_inference.py # Repetition (was B01d)
├── B27_table_validation.py     # Table validation (was B03a)
├── B28_docling_backend.py      # Docling backend (was B03c)
├── B29_column_detection.py     # Column detection (was B04a)
├── B30_xy_cut_ordering.py      # XY-Cut ordering (was B04b)
└── B31_vlm_detector.py         # VLM detection (was B17_vlm)
```

Removed files:
- `B03b_table_rendering.py` (merged into B14)

---

**Plan complete and saved to `docs/plans/2026-02-02-b-parsing-refactoring.md`. Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**

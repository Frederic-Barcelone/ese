# corpus_metadata/tests/Z22_visual_models.py
"""
Tests for visual extraction data models.
"""
import pytest

from A_core.A13_visual_models import (
    CaptionCandidate,
    CaptionProvenance,
    CaptionSearchZones,
    ExtractedVisual,
    PageLocation,
    ReferenceSource,
    TableCellStructure,
    TableComplexitySignals,
    TableExtractionMode,
    TableStructure,
    TextMention,
    TriageDecision,
    TriageResult,
    VisualCandidate,
    VisualReference,
    VisualRelationships,
    VisualType,
    VLMClassificationResult,
    VLMEnrichmentResult,
    VLMTableValidation,
)


# -------------------------
# PageLocation Tests
# -------------------------


class TestPageLocation:
    def test_valid_page_location(self):
        loc = PageLocation(page_num=1, bbox_pts=(10.0, 20.0, 100.0, 200.0))
        assert loc.page_num == 1
        assert loc.bbox_pts == (10.0, 20.0, 100.0, 200.0)

    def test_invalid_negative_coordinates(self):
        with pytest.raises(ValueError, match="cannot be negative"):
            PageLocation(page_num=1, bbox_pts=(-10.0, 20.0, 100.0, 200.0))

    def test_invalid_bbox_geometry(self):
        # x1 < x0
        with pytest.raises(ValueError, match="Invalid bbox geometry"):
            PageLocation(page_num=1, bbox_pts=(100.0, 20.0, 10.0, 200.0))

        # y1 < y0
        with pytest.raises(ValueError, match="Invalid bbox geometry"):
            PageLocation(page_num=1, bbox_pts=(10.0, 200.0, 100.0, 20.0))

    def test_invalid_page_num(self):
        with pytest.raises(ValueError):
            PageLocation(page_num=0, bbox_pts=(10.0, 20.0, 100.0, 200.0))

    def test_frozen_model(self):
        loc = PageLocation(page_num=1, bbox_pts=(10.0, 20.0, 100.0, 200.0))
        with pytest.raises(Exception):  # ValidationError for frozen model
            loc.page_num = 2


# -------------------------
# VisualReference Tests
# -------------------------


class TestVisualReference:
    def test_simple_reference(self):
        ref = VisualReference(
            raw_string="Table 1",
            type_label="Table",
            numbers=[1],
            source=ReferenceSource.CAPTION,
        )
        assert ref.type_label == "Table"
        assert ref.numbers == [1]
        assert ref.is_range is False
        assert ref.suffix is None

    def test_range_reference(self):
        ref = VisualReference(
            raw_string="Figure 2-4",
            type_label="Figure",
            numbers=[2, 3, 4],
            is_range=True,
            source=ReferenceSource.BODY_TEXT,
        )
        assert ref.numbers == [2, 3, 4]
        assert ref.is_range is True

    def test_reference_with_suffix(self):
        ref = VisualReference(
            raw_string="Figure 1A",
            type_label="Figure",
            numbers=[1],
            suffix="A",
            source=ReferenceSource.VLM,
        )
        assert ref.suffix == "A"

    def test_empty_numbers_invalid(self):
        with pytest.raises(ValueError, match="at least one number"):
            VisualReference(
                raw_string="Table",
                type_label="Table",
                numbers=[],
                source=ReferenceSource.CAPTION,
            )

    def test_empty_type_label_invalid(self):
        with pytest.raises(ValueError, match="non-empty"):
            VisualReference(
                raw_string="1",
                type_label="",
                numbers=[1],
                source=ReferenceSource.CAPTION,
            )


# -------------------------
# TextMention Tests
# -------------------------


class TestTextMention:
    def test_valid_mention(self):
        ref = VisualReference(
            raw_string="Figure 2",
            type_label="Figure",
            numbers=[2],
            source=ReferenceSource.BODY_TEXT,
        )
        mention = TextMention(
            text="see Figure 2",
            page_num=5,
            char_offset=120,
            reference=ref,
        )
        assert mention.text == "see Figure 2"
        assert mention.page_num == 5
        assert mention.reference.numbers == [2]

    def test_mention_without_reference(self):
        mention = TextMention(
            text="see the figure",
            page_num=3,
            char_offset=50,
        )
        assert mention.reference is None


# -------------------------
# VisualRelationships Tests
# -------------------------


class TestVisualRelationships:
    def test_empty_relationships(self):
        rel = VisualRelationships()
        assert rel.text_mentions == []
        assert rel.section_context is None
        assert rel.continued_from is None
        assert rel.continues_to is None

    def test_full_relationships(self):
        mention = TextMention(
            text="see Table 1",
            page_num=2,
            char_offset=100,
        )
        rel = VisualRelationships(
            text_mentions=[mention],
            section_context="Results",
            continued_from="visual-001",
            continues_to="visual-003",
        )
        assert len(rel.text_mentions) == 1
        assert rel.section_context == "Results"


# -------------------------
# TableStructure Tests
# -------------------------


class TestTableStructure:
    def test_simple_table(self):
        table = TableStructure(
            headers=[["Col A", "Col B"]],
            rows=[["val1", "val2"], ["val3", "val4"]],
        )
        assert len(table.headers) == 1
        assert len(table.rows) == 2
        assert table.token_coverage == 1.0

    def test_multi_level_headers(self):
        table = TableStructure(
            headers=[
                ["Group A", "Group A", "Group B", "Group B"],
                ["Sub 1", "Sub 2", "Sub 1", "Sub 2"],
            ],
            rows=[["a", "b", "c", "d"]],
        )
        assert len(table.headers) == 2

    def test_table_with_cells(self):
        cells = [
            TableCellStructure(text="Header", row_index=0, col_index=0, is_header=True),
            TableCellStructure(text="Data", row_index=1, col_index=0),
        ]
        table = TableStructure(cells=cells)
        assert len(table.cells) == 2
        assert table.cells[0].is_header is True

    def test_low_confidence(self):
        table = TableStructure(
            headers=[["A"]],
            rows=[["1"]],
            token_coverage=0.5,
            structure_confidence=0.7,
        )
        assert table.token_coverage == 0.5
        assert table.structure_confidence == 0.7


# -------------------------
# TableComplexitySignals Tests
# -------------------------


class TestTableComplexitySignals:
    def test_simple_table_signals(self):
        signals = TableComplexitySignals(
            column_count=3,
            row_count=10,
        )
        assert signals.merged_cell_count == 0
        assert signals.header_depth == 1
        assert signals.spans_multiple_pages is False

    def test_complex_table_signals(self):
        signals = TableComplexitySignals(
            merged_cell_count=8,
            header_depth=3,
            token_coverage_ratio=0.65,
            column_count=12,
            row_count=50,
            spans_multiple_pages=True,
            has_continuation_marker=True,
        )
        assert signals.merged_cell_count == 8
        assert signals.header_depth == 3
        assert signals.spans_multiple_pages is True


# -------------------------
# TriageResult Tests
# -------------------------


class TestTriageResult:
    def test_skip_decision(self):
        result = TriageResult(
            decision=TriageDecision.SKIP,
            reason="tiny_area",
            confidence=0.95,
        )
        assert result.decision == TriageDecision.SKIP

    def test_vlm_required_decision(self):
        result = TriageResult(
            decision=TriageDecision.VLM_REQUIRED,
            reason="has_caption",
            confidence=0.90,
        )
        assert result.decision == TriageDecision.VLM_REQUIRED


# -------------------------
# CaptionCandidate Tests
# -------------------------


class TestCaptionCandidate:
    def test_valid_caption_candidate(self):
        ref = VisualReference(
            raw_string="Table 1",
            type_label="Table",
            numbers=[1],
            source=ReferenceSource.CAPTION,
        )
        caption = CaptionCandidate(
            text="Table 1. Patient demographics.",
            bbox_pts=(50.0, 400.0, 300.0, 420.0),
            provenance=CaptionProvenance.PDF_TEXT,
            position="below",
            distance_pts=10.0,
            confidence=0.95,
            parsed_reference=ref,
        )
        assert caption.position == "below"
        assert caption.provenance == CaptionProvenance.PDF_TEXT

    def test_caption_from_ocr(self):
        caption = CaptionCandidate(
            text="Figure 2 - Results",
            bbox_pts=(50.0, 100.0, 300.0, 120.0),
            provenance=CaptionProvenance.OCR,
            position="above",
            distance_pts=5.0,
            confidence=0.80,
        )
        assert caption.provenance == CaptionProvenance.OCR


# -------------------------
# CaptionSearchZones Tests
# -------------------------


class TestCaptionSearchZones:
    def test_default_zones(self):
        zones = CaptionSearchZones()
        assert zones.above == 72.0
        assert zones.below == 72.0
        assert zones.left == 36.0
        assert zones.right == 36.0

    def test_custom_zones(self):
        zones = CaptionSearchZones(above=100.0, below=150.0, left=50.0, right=50.0)
        assert zones.above == 100.0
        assert zones.below == 150.0


# -------------------------
# VisualCandidate Tests
# -------------------------


class TestVisualCandidate:
    def test_valid_candidate(self):
        candidate = VisualCandidate(
            source="docling",
            page_num=1,
            bbox_pts=(100.0, 200.0, 400.0, 500.0),
            page_width_pts=612.0,
            page_height_pts=792.0,
            area_ratio=0.15,
            has_nearby_caption=True,
        )
        assert candidate.source == "docling"
        assert candidate.has_nearby_caption is True

    def test_area_property(self):
        candidate = VisualCandidate(
            source="native_raster",
            page_num=1,
            bbox_pts=(0.0, 0.0, 100.0, 100.0),
            page_width_pts=612.0,
            page_height_pts=792.0,
            area_ratio=0.02,
        )
        assert candidate.area == 10000.0  # 100 * 100
        assert candidate.page_area == 612.0 * 792.0

    def test_candidate_with_caption(self):
        caption = CaptionCandidate(
            text="Table 1",
            bbox_pts=(100.0, 510.0, 400.0, 530.0),
            provenance=CaptionProvenance.PDF_TEXT,
            position="below",
            distance_pts=10.0,
            confidence=0.9,
        )
        candidate = VisualCandidate(
            source="docling",
            docling_type="table",
            page_num=1,
            bbox_pts=(100.0, 200.0, 400.0, 500.0),
            page_width_pts=612.0,
            page_height_pts=792.0,
            area_ratio=0.15,
            has_nearby_caption=True,
            caption_candidate=caption,
        )
        assert candidate.caption_candidate is not None
        assert candidate.caption_candidate.text == "Table 1"


# -------------------------
# ExtractedVisual Tests
# -------------------------


class TestExtractedVisual:
    def test_single_page_figure(self):
        visual = ExtractedVisual(
            visual_type=VisualType.FIGURE,
            confidence=0.95,
            page_range=[1],
            bbox_pts_per_page=[PageLocation(page_num=1, bbox_pts=(100.0, 200.0, 400.0, 500.0))],
            image_base64="iVBORw0KGgo=",  # minimal valid base64
            extraction_method="docling+vlm",
            source_file="test.pdf",
        )
        assert visual.is_figure is True
        assert visual.is_table is False
        assert visual.is_multipage is False
        assert visual.primary_page == 1

    def test_multipage_table(self):
        visual = ExtractedVisual(
            visual_type=VisualType.TABLE,
            confidence=0.90,
            page_range=[3, 4],
            bbox_pts_per_page=[
                PageLocation(page_num=3, bbox_pts=(50.0, 100.0, 550.0, 700.0)),
                PageLocation(page_num=4, bbox_pts=(50.0, 50.0, 550.0, 400.0)),
            ],
            image_base64="iVBORw0KGgo=",
            extraction_method="docling+vlm",
            source_file="protocol.pdf",
            docling_table=TableStructure(headers=[["A", "B"]], rows=[["1", "2"]]),
            table_extraction_mode=TableExtractionMode.ACCURATE,
        )
        assert visual.is_table is True
        assert visual.is_multipage is True
        assert len(visual.page_range) == 2
        assert visual.docling_table is not None

    def test_full_visual_with_relationships(self):
        ref = VisualReference(
            raw_string="Figure 1",
            type_label="Figure",
            numbers=[1],
            source=ReferenceSource.CAPTION,
        )
        mention = TextMention(
            text="As shown in Figure 1",
            page_num=5,
            char_offset=200,
            reference=ref,
        )
        relationships = VisualRelationships(
            text_mentions=[mention],
            section_context="Results",
        )
        visual = ExtractedVisual(
            visual_type=VisualType.FIGURE,
            confidence=0.98,
            page_range=[2],
            bbox_pts_per_page=[PageLocation(page_num=2, bbox_pts=(100.0, 100.0, 500.0, 400.0))],
            caption_text="Figure 1. Kaplan-Meier survival curves.",
            caption_provenance=CaptionProvenance.PDF_TEXT,
            reference=ref,
            image_base64="iVBORw0KGgo=",
            extraction_method="docling+vlm",
            source_file="paper.pdf",
            relationships=relationships,
            triage_decision=TriageDecision.VLM_REQUIRED,
            triage_reason="has_caption",
        )
        assert visual.caption_text is not None
        assert visual.reference.numbers == [1]
        assert len(visual.relationships.text_mentions) == 1

    def test_page_consistency_validation(self):
        # Mismatch between page_range and bbox_pts_per_page length
        with pytest.raises(ValueError, match="page_range length"):
            ExtractedVisual(
                visual_type=VisualType.FIGURE,
                confidence=0.9,
                page_range=[1, 2],
                bbox_pts_per_page=[PageLocation(page_num=1, bbox_pts=(0.0, 0.0, 100.0, 100.0))],
                image_base64="iVBORw0KGgo=",
                extraction_method="docling+vlm",
                source_file="test.pdf",
            )

    def test_page_num_consistency_validation(self):
        # Page numbers don't match
        with pytest.raises(ValueError, match="must match"):
            ExtractedVisual(
                visual_type=VisualType.FIGURE,
                confidence=0.9,
                page_range=[1],
                bbox_pts_per_page=[PageLocation(page_num=2, bbox_pts=(0.0, 0.0, 100.0, 100.0))],
                image_base64="iVBORw0KGgo=",
                extraction_method="docling+vlm",
                source_file="test.pdf",
            )

    def test_table_fields_on_non_table_invalid(self):
        # docling_table on a FIGURE should fail
        with pytest.raises(ValueError, match="should only be set for TABLE"):
            ExtractedVisual(
                visual_type=VisualType.FIGURE,
                confidence=0.9,
                page_range=[1],
                bbox_pts_per_page=[PageLocation(page_num=1, bbox_pts=(0.0, 0.0, 100.0, 100.0))],
                image_base64="iVBORw0KGgo=",
                extraction_method="docling+vlm",
                source_file="test.pdf",
                docling_table=TableStructure(headers=[["A"]], rows=[["1"]]),
            )


# -------------------------
# VLM Result Models Tests
# -------------------------


class TestVLMClassificationResult:
    def test_valid_classification(self):
        result = VLMClassificationResult(
            visual_type=VisualType.TABLE,
            confidence=0.92,
            reasoning="Grid structure with headers",
        )
        assert result.visual_type == VisualType.TABLE
        assert result.confidence == 0.92


class TestVLMTableValidation:
    def test_valid_table(self):
        result = VLMTableValidation(
            is_misparsed=False,
            confidence=0.95,
        )
        assert result.is_misparsed is False

    def test_misparsed_table(self):
        corrected = TableStructure(headers=[["A", "B"]], rows=[["1", "2"]])
        result = VLMTableValidation(
            is_misparsed=True,
            confidence=0.88,
            issues=["Merged cells not detected", "Header row missing"],
            corrected_structure=corrected,
        )
        assert result.is_misparsed is True
        assert len(result.issues) == 2
        assert result.corrected_structure is not None


class TestVLMEnrichmentResult:
    def test_full_enrichment(self):
        classification = VLMClassificationResult(
            visual_type=VisualType.TABLE,
            confidence=0.95,
        )
        ref = VisualReference(
            raw_string="Table 2",
            type_label="Table",
            numbers=[2],
            source=ReferenceSource.VLM,
        )
        validation = VLMTableValidation(
            is_misparsed=False,
            confidence=0.90,
        )
        result = VLMEnrichmentResult(
            classification=classification,
            parsed_reference=ref,
            extracted_caption="Table 2. Demographics.",
            table_validation=validation,
        )
        assert result.classification.visual_type == VisualType.TABLE
        assert result.parsed_reference.numbers == [2]
        assert result.is_continuation is False

    def test_continuation_detection(self):
        classification = VLMClassificationResult(
            visual_type=VisualType.TABLE,
            confidence=0.90,
        )
        result = VLMEnrichmentResult(
            classification=classification,
            is_continuation=True,
            continuation_of_reference="Table 1",
        )
        assert result.is_continuation is True
        assert result.continuation_of_reference == "Table 1"

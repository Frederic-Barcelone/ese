# corpus_metadata/K_tests/K50_test_g_h_i_j_imports.py
"""
Tests for G_config, H_pipeline, I_extraction, and J_export module imports.

Ensures all modules can be imported and have proper exports.
"""

from __future__ import annotations

import importlib
from enum import Enum

import pytest


# G_config modules
G_CONFIG_MODULES = [
    "G01_config_keys",
]

# H_pipeline modules
H_PIPELINE_MODULES = [
    "H01_component_factory",
    "H02_abbreviation_pipeline",
    "H03_visual_integration",
    "H04_merge_resolver",
]

# I_extraction modules
I_EXTRACTION_MODULES = [
    "I01_entity_processors",
    "I02_feasibility_processor",
]

# J_export modules
J_EXPORT_MODULES = [
    "J01_export_handlers",
    "J01a_entity_exporters",
    "J01b_metadata_exporters",
    "J02_visual_export",
]


class TestGConfigImports:
    """Tests that all G_config modules can be imported."""

    @pytest.mark.parametrize("module_name", G_CONFIG_MODULES)
    def test_module_imports(self, module_name):
        try:
            module = importlib.import_module(f"G_config.{module_name}")
            assert module is not None
        except ImportError as e:
            pytest.fail(f"Failed to import G_config.{module_name}: {e}")

    @pytest.mark.parametrize("module_name", G_CONFIG_MODULES)
    def test_module_has_all(self, module_name):
        try:
            module = importlib.import_module(f"G_config.{module_name}")
            if hasattr(module, "__all__"):
                assert isinstance(module.__all__, (list, tuple))
        except ImportError as e:
            pytest.fail(f"Module {module_name} not importable: {e}")


class TestHPipelineImports:
    """Tests that all H_pipeline modules can be imported."""

    @pytest.mark.parametrize("module_name", H_PIPELINE_MODULES)
    def test_module_imports(self, module_name):
        try:
            module = importlib.import_module(f"H_pipeline.{module_name}")
            assert module is not None
        except ImportError as e:
            pytest.fail(f"Failed to import H_pipeline.{module_name}: {e}")

    @pytest.mark.parametrize("module_name", H_PIPELINE_MODULES)
    def test_module_has_all(self, module_name):
        try:
            module = importlib.import_module(f"H_pipeline.{module_name}")
            if hasattr(module, "__all__"):
                assert isinstance(module.__all__, (list, tuple))
        except ImportError as e:
            pytest.fail(f"Module {module_name} not importable: {e}")


class TestIExtractionImports:
    """Tests that all I_extraction modules can be imported."""

    @pytest.mark.parametrize("module_name", I_EXTRACTION_MODULES)
    def test_module_imports(self, module_name):
        try:
            module = importlib.import_module(f"I_extraction.{module_name}")
            assert module is not None
        except ImportError as e:
            pytest.fail(f"Failed to import I_extraction.{module_name}: {e}")

    @pytest.mark.parametrize("module_name", I_EXTRACTION_MODULES)
    def test_module_has_all(self, module_name):
        try:
            module = importlib.import_module(f"I_extraction.{module_name}")
            if hasattr(module, "__all__"):
                assert isinstance(module.__all__, (list, tuple))
        except ImportError as e:
            pytest.fail(f"Module {module_name} not importable: {e}")


class TestJExportImports:
    """Tests that all J_export modules can be imported."""

    @pytest.mark.parametrize("module_name", J_EXPORT_MODULES)
    def test_module_imports(self, module_name):
        try:
            module = importlib.import_module(f"J_export.{module_name}")
            assert module is not None
        except ImportError as e:
            pytest.fail(f"Failed to import J_export.{module_name}: {e}")

    @pytest.mark.parametrize("module_name", J_EXPORT_MODULES)
    def test_module_has_all(self, module_name):
        try:
            module = importlib.import_module(f"J_export.{module_name}")
            if hasattr(module, "__all__"):
                assert isinstance(module.__all__, (list, tuple))
        except ImportError as e:
            pytest.fail(f"Module {module_name} not importable: {e}")


class TestGConfigExports:
    """Tests for specific G_config module exports."""

    def test_config_keys_exports(self):
        from G_config.G01_config_keys import (
            ConfigKey,
            get_config,
            get_nested_config,
        )
        assert issubclass(ConfigKey, Enum)
        assert callable(get_config)
        assert callable(get_nested_config)


class TestHPipelineExports:
    """Tests for specific H_pipeline module exports."""

    def test_component_factory_exports(self):
        from H_pipeline.H01_component_factory import ComponentFactory
        assert hasattr(ComponentFactory, "create_parser")
        assert hasattr(ComponentFactory, "create_generators")

    def test_merge_resolver_exports(self):
        from H_pipeline.H04_merge_resolver import (
            MergeConfig,
            MergeResolver,
            get_merge_resolver,
        )
        assert hasattr(MergeConfig, "dedupe_key_fields")
        assert hasattr(MergeResolver, "merge")
        assert callable(get_merge_resolver)

    def test_visual_integration_exports(self):
        from H_pipeline.H03_visual_integration import (
            VisualPipelineIntegration,
            create_visual_integration,
        )
        assert hasattr(VisualPipelineIntegration, "extract")
        assert callable(create_visual_integration)


class TestIExtractionExports:
    """Tests for specific I_extraction module exports."""

    def test_entity_processors_exports(self):
        from I_extraction.I01_entity_processors import EntityProcessor
        assert hasattr(EntityProcessor, "create_entity_from_candidate")
        assert hasattr(EntityProcessor, "process_diseases")
        assert hasattr(EntityProcessor, "process_drugs")
        assert hasattr(EntityProcessor, "process_genes")

    def test_feasibility_processor_exports(self):
        from I_extraction.I02_feasibility_processor import FeasibilityProcessor
        assert hasattr(FeasibilityProcessor, "process")


class TestJExportExports:
    """Tests for specific J_export module exports."""

    def test_export_handlers_exports(self):
        from J_export.J01_export_handlers import ExportManager
        assert hasattr(ExportManager, "get_output_dir")

    def test_entity_exporters_exports(self):
        from J_export.J01a_entity_exporters import (
            export_disease_results,
            export_gene_results,
            export_drug_results,
        )
        assert callable(export_disease_results)
        assert callable(export_gene_results)
        assert callable(export_drug_results)

    def test_visual_export_exports(self):
        from J_export.J02_visual_export import (
            visual_to_dict,
            export_visuals_to_json,
        )
        assert callable(visual_to_dict)
        assert callable(export_visuals_to_json)


# ---------------------------------------------------------------------------
# Behavioral tests
# ---------------------------------------------------------------------------


class TestConfigKeysBehavioral:
    """Behavioral tests for G01_config_keys helper functions."""

    def test_get_config_returns_value_for_known_key(self):
        """get_config returns the stored value when the key is present."""
        from G_config.G01_config_keys import ConfigKey, get_config

        config = {"confidence_threshold": 0.85, "timeout_seconds": 60}
        result = get_config(config, ConfigKey.CONFIDENCE_THRESHOLD)

        assert result == 0.85
        assert isinstance(result, float)

    def test_get_config_returns_default_for_missing_key(self):
        """get_config falls back to the enum default when the key is absent."""
        from G_config.G01_config_keys import ConfigKey, get_config

        empty_config: dict = {}
        result = get_config(empty_config, ConfigKey.CONTEXT_WINDOW)

        assert result == ConfigKey.CONTEXT_WINDOW.default
        assert result == 300

    def test_get_config_returns_explicit_default_over_enum_default(self):
        """An explicit default parameter overrides the enum's built-in default."""
        from G_config.G01_config_keys import ConfigKey, get_config

        empty_config: dict = {}
        result = get_config(empty_config, ConfigKey.TIMEOUT_SECONDS, default=120)

        assert result == 120

    def test_get_nested_config_traverses_sections(self):
        """get_nested_config walks through nested dicts using enum keys."""
        from G_config.G01_config_keys import ConfigKey, CacheConfig, get_nested_config

        config = {"cache": {"ttl_hours": 48, "enabled": False}}
        result = get_nested_config(config, ConfigKey.CACHE, CacheConfig.TTL_HOURS)

        assert result == 48

    def test_get_nested_config_returns_default_on_missing_section(self):
        """get_nested_config returns the leaf default when a section is absent."""
        from G_config.G01_config_keys import ConfigKey, CacheConfig, get_nested_config

        config: dict = {}
        result = get_nested_config(config, ConfigKey.CACHE, CacheConfig.TTL_DAYS)

        assert result == CacheConfig.TTL_DAYS.default
        assert result == 30


class TestMergeResolverBehavioral:
    """Behavioral tests for H04_merge_resolver merge logic."""

    def _make_extraction(
        self,
        value: str,
        strategy_id: str,
        evidence_text: str = "",
        from_table: bool = False,
    ):
        """Helper to build a RawExtraction with pharma-relevant data."""
        from A_core.A02_interfaces import RawExtraction
        from A_core.A14_extraction_result import EntityType

        return RawExtraction(
            doc_id="rare_disease_trial_001",
            entity_type=EntityType.DISEASE,
            field_name="disease",
            value=value,
            page_num=1,
            strategy_id=strategy_id,
            evidence_text=evidence_text,
        )

    def test_merge_picks_higher_priority_strategy(self):
        """When two strategies extract the same entity, the higher-priority one wins."""
        from H_pipeline.H04_merge_resolver import MergeConfig, MergeResolver

        config = MergeConfig(
            strategy_priority={
                "disease_lexicon_specialized": 10,
                "disease_lexicon_general": 1,
            },
        )
        resolver = MergeResolver(config)

        specialized = self._make_extraction(
            value="IgA nephropathy",
            strategy_id="disease_lexicon_specialized",
            evidence_text="Patients with biopsy-proven IgA nephropathy were enrolled.",
        )
        general = self._make_extraction(
            value="IgA nephropathy",
            strategy_id="disease_lexicon_general",
            evidence_text="IgA nephropathy diagnosis confirmed.",
        )

        merged = resolver.merge([general, specialized])

        assert len(merged) == 1
        assert merged[0].strategy_id == "disease_lexicon_specialized"
        assert merged[0].value == "IgA nephropathy"

    def test_merge_no_conflict_returns_all_entities(self):
        """Non-duplicate extractions are all preserved after merge."""
        from H_pipeline.H04_merge_resolver import MergeResolver

        resolver = MergeResolver()

        disease_a = self._make_extraction(
            value="Duchenne muscular dystrophy",
            strategy_id="disease_lexicon_orphanet",
            evidence_text="DMD affects approximately 1 in 3,500 male births.",
        )
        disease_b = self._make_extraction(
            value="spinal muscular atrophy",
            strategy_id="disease_lexicon_general",
            evidence_text="SMA is caused by mutations in the SMN1 gene.",
        )

        merged = resolver.merge([disease_a, disease_b])

        assert len(merged) == 2
        merged_values = {r.value for r in merged}
        assert "Duchenne muscular dystrophy" in merged_values
        assert "spinal muscular atrophy" in merged_values

    def test_merge_empty_input_returns_empty_list(self):
        """Merging an empty list returns an empty list without error."""
        from H_pipeline.H04_merge_resolver import MergeResolver

        resolver = MergeResolver()
        merged = resolver.merge([])

        assert merged == []

    def test_get_merge_resolver_returns_singleton(self):
        """get_merge_resolver returns a usable singleton with default config."""
        from H_pipeline.H04_merge_resolver import get_merge_resolver, MergeResolver

        resolver = get_merge_resolver()

        assert isinstance(resolver, MergeResolver)
        assert resolver.config.strategy_priority.get("disease_lexicon_specialized") == 10


class TestExportManagerBehavioral:
    """Behavioral tests for J01_export_handlers ExportManager."""

    def test_get_output_dir_uses_pdf_stem(self, tmp_path):
        """get_output_dir creates a folder named after the PDF stem."""
        from J_export.J01_export_handlers import ExportManager
        from pathlib import Path

        manager = ExportManager(
            run_id="run_20260215_001",
            pipeline_version="v0.8",
        )

        pdf_path = tmp_path / "rare_disease_trial.pdf"
        pdf_path.touch()

        out_dir = manager.get_output_dir(pdf_path)

        assert isinstance(out_dir, Path)
        assert out_dir.name == "rare_disease_trial"
        assert out_dir.exists()
        assert out_dir.parent == tmp_path

    def test_get_output_dir_respects_override(self, tmp_path):
        """When output_dir_override is set, get_output_dir uses it instead of PDF parent."""
        from J_export.J01_export_handlers import ExportManager

        custom_dir = tmp_path / "custom_output"
        manager = ExportManager(
            run_id="run_20260215_002",
            pipeline_version="v0.8",
            output_dir=custom_dir,
        )

        pdf_path = tmp_path / "enzyme_replacement_therapy.pdf"
        pdf_path.touch()

        out_dir = manager.get_output_dir(pdf_path)

        assert out_dir == custom_dir
        assert out_dir.exists()

    def test_get_output_dir_creates_nested_directories(self, tmp_path):
        """get_output_dir creates parent directories if they do not exist."""
        from J_export.J01_export_handlers import ExportManager

        deep_dir = tmp_path / "level1" / "level2"
        manager = ExportManager(
            run_id="run_20260215_003",
            pipeline_version="v0.8",
            output_dir=deep_dir,
        )

        pdf_path = tmp_path / "gene_therapy_study.pdf"
        pdf_path.touch()

        out_dir = manager.get_output_dir(pdf_path)

        assert out_dir == deep_dir
        assert out_dir.exists()


class TestVisualExportBehavioral:
    """Behavioral tests for J02_visual_export visual_to_dict."""

    def _make_visual(self, **overrides):
        """Helper to create a minimal ExtractedVisual for a clinical figure."""
        from A_core.A13_visual_models import ExtractedVisual, VisualType, PageLocation

        defaults = dict(
            visual_type=VisualType.FIGURE,
            confidence=0.92,
            page_range=[3],
            bbox_pts_per_page=[PageLocation(page_num=3, bbox_pts=(72.0, 150.0, 540.0, 450.0))],
            image_base64="iVBORw0KGgo=",
            extraction_method="docling+vlm",
            source_file="patient_flow_consort.pdf",
            caption_text="Figure 1. CONSORT flow diagram for Phase III randomized trial.",
            vlm_title="CONSORT Patient Flow Diagram",
            vlm_description="Flow diagram showing screening, randomization, and follow-up.",
        )
        defaults.update(overrides)
        return ExtractedVisual(**defaults)  # type: ignore[arg-type]

    def test_visual_to_dict_contains_expected_keys(self):
        """visual_to_dict returns a dict with visual_id, visual_type, confidence, and caption."""
        from J_export.J02_visual_export import visual_to_dict

        visual = self._make_visual()
        result = visual_to_dict(visual, include_image=False)

        assert isinstance(result, dict)
        assert result["visual_id"] == visual.visual_id
        assert result["visual_type"] == "figure"
        assert result["confidence"] == 0.92
        assert result["caption"] == "Figure 1. CONSORT flow diagram for Phase III randomized trial."
        assert result["title"] == "CONSORT Patient Flow Diagram"
        assert result["description"] == "Flow diagram showing screening, randomization, and follow-up."

    def test_visual_to_dict_includes_location_data(self):
        """visual_to_dict includes page location and bbox in the locations list."""
        from J_export.J02_visual_export import visual_to_dict

        visual = self._make_visual()
        result = visual_to_dict(visual, include_image=False)

        assert "locations" in result
        assert len(result["locations"]) == 1
        loc = result["locations"][0]
        assert loc["page_num"] == 3
        assert loc["bbox_pts"] == [72.0, 150.0, 540.0, 450.0]

    def test_visual_to_dict_with_image_file_omits_base64(self):
        """When image_file is provided, base64 data is omitted from the dict."""
        from J_export.J02_visual_export import visual_to_dict

        visual = self._make_visual()
        result = visual_to_dict(
            visual, include_image=True, image_file="consort_figure_page3_1.png",
        )

        assert result["image_file"] == "consort_figure_page3_1.png"
        assert "image_base64" not in result

    def test_visual_to_dict_table_includes_table_data(self):
        """A TABLE-type visual includes a table_data section in the output dict."""
        from A_core.A13_visual_models import VisualType, TableStructure
        from J_export.J02_visual_export import visual_to_dict

        table_structure = TableStructure(
            headers=[["Drug", "Dose", "Route", "Frequency"]],
            rows=[
                ["Nusinersen", "12 mg", "Intrathecal", "Every 4 months"],
                ["Risdiplam", "5 mg", "Oral", "Daily"],
            ],
            structure_confidence=0.95,
        )
        visual = self._make_visual(
            visual_type=VisualType.TABLE,
            caption_text="Table 2. Approved treatments for spinal muscular atrophy.",
            vlm_title="SMA Treatment Comparison",
            docling_table=table_structure,
        )
        result = visual_to_dict(visual, include_image=False)

        assert result["visual_type"] == "table"
        assert "table_data" in result
        assert result["table_data"]["structure"]["headers"] == [["Drug", "Dose", "Route", "Frequency"]]
        assert len(result["table_data"]["structure"]["rows"]) == 2
        assert result["table_data"]["structure"]["confidence"] == 0.95

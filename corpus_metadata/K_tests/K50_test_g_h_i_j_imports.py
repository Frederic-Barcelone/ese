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

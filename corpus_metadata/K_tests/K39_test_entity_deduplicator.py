# corpus_metadata/K_tests/K39_test_entity_deduplicator.py
"""
Tests for E_normalization.E17_entity_deduplicator module.

Tests entity deduplication for diseases, drugs, and genes.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from E_normalization.E17_entity_deduplicator import EntityDeduplicator
from A_core.A01_domain_models import ValidationStatus


def make_mock_disease(
    name: str,
    mondo_id: str = None,
    orpha_code: str = None,
    umls_cui: str = None,
    mesh_id: str = None,
    confidence: float = 0.9,
    status: ValidationStatus = ValidationStatus.VALIDATED,
    page: int = 1,
):
    """Create a mock disease entity."""
    disease = MagicMock()
    disease.id = f"disease_{name}_{page}"
    disease.preferred_label = name
    disease.mondo_id = mondo_id
    disease.orpha_code = orpha_code
    disease.umls_cui = umls_cui
    disease.mesh_id = mesh_id
    disease.confidence_score = confidence
    disease.status = status
    disease.identifiers = []
    if mondo_id:
        disease.identifiers.append({"source": "MONDO", "id": mondo_id})
    if orpha_code:
        disease.identifiers.append({"source": "ORPHA", "id": orpha_code})
    disease.primary_evidence = MagicMock()
    disease.primary_evidence.location = MagicMock()
    disease.primary_evidence.location.page_num = page
    disease.supporting_evidence = []
    disease.validation_flags = []
    disease.model_copy = lambda update: make_updated_mock(disease, update)
    return disease


def make_mock_drug(
    name: str,
    rxcui: str = None,
    drugbank_id: str = None,
    compound_id: str = None,
    confidence: float = 0.9,
    is_investigational: bool = False,
    status: ValidationStatus = ValidationStatus.VALIDATED,
    page: int = 1,
):
    """Create a mock drug entity."""
    drug = MagicMock()
    drug.id = f"drug_{name}_{page}"
    drug.preferred_name = name
    drug.rxcui = rxcui
    drug.drugbank_id = drugbank_id
    drug.compound_id = compound_id
    drug.confidence_score = confidence
    drug.is_investigational = is_investigational
    drug.status = status
    drug.identifiers = []
    if rxcui:
        drug.identifiers.append({"source": "RxNorm", "id": rxcui})
    drug.primary_evidence = MagicMock()
    drug.primary_evidence.location = MagicMock()
    drug.primary_evidence.location.page_num = page
    drug.supporting_evidence = []
    drug.validation_flags = []
    drug.model_copy = lambda update: make_updated_mock(drug, update)
    return drug


def make_mock_gene(
    symbol: str,
    hgnc_id: str = None,
    entrez_id: str = None,
    confidence: float = 0.9,
    is_alias: bool = False,
    associated_diseases: list = None,
    status: ValidationStatus = ValidationStatus.VALIDATED,
    page: int = 1,
):
    """Create a mock gene entity."""
    gene = MagicMock()
    gene.id = f"gene_{symbol}_{page}"
    gene.hgnc_symbol = symbol
    gene.hgnc_id = hgnc_id
    gene.entrez_id = entrez_id
    gene.confidence_score = confidence
    gene.is_alias = is_alias
    gene.associated_diseases = associated_diseases or []
    gene.status = status
    gene.identifiers = []
    if hgnc_id:
        gene.identifiers.append({"source": "HGNC", "id": hgnc_id})
    gene.primary_evidence = MagicMock()
    gene.primary_evidence.location = MagicMock()
    gene.primary_evidence.location.page_num = page
    gene.supporting_evidence = []
    gene.validation_flags = []
    gene.model_copy = lambda update: make_updated_mock(gene, update)
    return gene


def make_updated_mock(original, update):
    """Create updated mock with new values."""
    new_mock = MagicMock()
    for attr in dir(original):
        if not attr.startswith("_"):
            setattr(new_mock, attr, getattr(original, attr))
    for key, value in update.items():
        setattr(new_mock, key, value)
    return new_mock


class TestEntityDeduplicator:
    """Tests for EntityDeduplicator class."""

    @pytest.fixture
    def deduplicator(self):
        return EntityDeduplicator()

    def test_empty_list(self, deduplicator):
        assert deduplicator.deduplicate_diseases([]) == []
        assert deduplicator.deduplicate_drugs([]) == []
        assert deduplicator.deduplicate_genes([]) == []


class TestDiseaseDeduplication:
    """Tests for disease deduplication."""

    @pytest.fixture
    def deduplicator(self):
        return EntityDeduplicator()

    def test_dedup_by_mondo_id(self, deduplicator):
        diseases = [
            make_mock_disease("PAH", mondo_id="MONDO:0005149", page=1),
            make_mock_disease("Pulmonary Arterial Hypertension", mondo_id="MONDO:0005149", page=5),
        ]
        result = deduplicator.deduplicate_diseases(diseases)

        assert len(result) == 1
        assert result[0].mention_count == 2
        assert set(result[0].pages_mentioned) == {1, 5}

    def test_dedup_by_orpha_code(self, deduplicator):
        diseases = [
            make_mock_disease("Disease A", orpha_code="ORPHA:123", page=1),
            make_mock_disease("Disease A Variant", orpha_code="ORPHA:123", page=2),
        ]
        result = deduplicator.deduplicate_diseases(diseases)

        assert len(result) == 1

    def test_dedup_by_text_fallback(self, deduplicator):
        """When no IDs, dedup by normalized text."""
        diseases = [
            make_mock_disease("Pulmonary Arterial Hypertension", page=1),
            make_mock_disease("pulmonary arterial hypertension", page=2),
        ]
        result = deduplicator.deduplicate_diseases(diseases)

        assert len(result) == 1

    def test_different_diseases_not_merged(self, deduplicator):
        diseases = [
            make_mock_disease("Disease A", mondo_id="MONDO:0001"),
            make_mock_disease("Disease B", mondo_id="MONDO:0002"),
        ]
        result = deduplicator.deduplicate_diseases(diseases)

        assert len(result) == 2

    def test_non_validated_excluded(self, deduplicator):
        diseases = [
            make_mock_disease("Validated", status=ValidationStatus.VALIDATED),
            make_mock_disease("Rejected", status=ValidationStatus.REJECTED),
        ]
        result = deduplicator.deduplicate_diseases(diseases)

        # Validated stays, rejected goes to separate list
        validated = [d for d in result if d.status == ValidationStatus.VALIDATED]
        assert len(validated) == 1


class TestDrugDeduplication:
    """Tests for drug deduplication."""

    @pytest.fixture
    def deduplicator(self):
        return EntityDeduplicator()

    def test_dedup_by_rxcui(self, deduplicator):
        drugs = [
            make_mock_drug("aspirin", rxcui="1191", page=1),
            make_mock_drug("Aspirin", rxcui="1191", page=3),
        ]
        result = deduplicator.deduplicate_drugs(drugs)

        assert len(result) == 1
        assert result[0].mention_count == 2

    def test_dedup_by_drugbank_id(self, deduplicator):
        drugs = [
            make_mock_drug("Drug A", drugbank_id="DB00001", page=1),
            make_mock_drug("Drug A Brand", drugbank_id="DB00001", page=2),
        ]
        result = deduplicator.deduplicate_drugs(drugs)

        assert len(result) == 1

    def test_dedup_by_compound_id(self, deduplicator):
        drugs = [
            make_mock_drug("LNP023", compound_id="LNP023", page=1),
            make_mock_drug("iptacopan", compound_id="lnp023", page=2),  # case insensitive
        ]
        result = deduplicator.deduplicate_drugs(drugs)

        assert len(result) == 1

    def test_investigational_drug_priority(self, deduplicator):
        drugs = [
            make_mock_drug("Drug", is_investigational=False, confidence=0.9),
            make_mock_drug("Drug", is_investigational=True, confidence=0.85),
        ]
        result = deduplicator.deduplicate_drugs(drugs)

        # Investigational should have higher priority in scoring
        assert len(result) == 1


class TestGeneDeduplication:
    """Tests for gene deduplication."""

    @pytest.fixture
    def deduplicator(self):
        return EntityDeduplicator()

    def test_dedup_by_hgnc_id(self, deduplicator):
        genes = [
            make_mock_gene("BRCA1", hgnc_id="HGNC:1100", page=1),
            make_mock_gene("BRCA1", hgnc_id="HGNC:1100", page=5),
        ]
        result = deduplicator.deduplicate_genes(genes)

        assert len(result) == 1
        assert result[0].mention_count == 2

    def test_dedup_by_entrez_id(self, deduplicator):
        genes = [
            make_mock_gene("TP53", entrez_id="7157", page=1),
            make_mock_gene("p53", entrez_id="7157", page=2),
        ]
        result = deduplicator.deduplicate_genes(genes)

        assert len(result) == 1

    def test_dedup_by_symbol_fallback(self, deduplicator):
        genes = [
            make_mock_gene("EGFR", page=1),
            make_mock_gene("egfr", page=2),  # case normalization
        ]
        result = deduplicator.deduplicate_genes(genes)

        assert len(result) == 1

    def test_alias_deprioritized(self, deduplicator):
        genes = [
            make_mock_gene("BRCA1", is_alias=True, confidence=0.95),
            make_mock_gene("BRCA1", is_alias=False, confidence=0.8),
        ]
        result = deduplicator.deduplicate_genes(genes)

        # Non-alias should be preferred
        assert len(result) == 1
        assert not result[0].is_alias

    def test_associated_diseases_boost(self, deduplicator):
        genes = [
            make_mock_gene("BRCA1", associated_diseases=[]),
            make_mock_gene("BRCA1", associated_diseases=["Breast Cancer", "Ovarian Cancer"]),
        ]
        result = deduplicator.deduplicate_genes(genes)

        # Gene with more associated diseases should be preferred
        assert len(result) == 1
        assert len(result[0].associated_diseases) == 2

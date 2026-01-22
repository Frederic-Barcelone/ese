# corpus_metadata/tests/conftest.py
"""
Pytest configuration and fixtures for corpus_metadata tests.

Provides:
- Common fixtures for testing enrichers, generators, and core modules
- Mock responses for external APIs (PubTator, ClinicalTrials.gov, DOI)
- Sample data factories for disease, drug, and other entity types
- Test configuration helpers

Usage:
    # In test files, fixtures are automatically available:
    def test_disease_enricher(mock_pubtator_client, sample_disease):
        enricher = DiseaseEnricher()
        result = enricher.enrich(sample_disease)
        assert result.mesh_id is not None
"""

import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, List
from unittest.mock import MagicMock, patch

import pytest

# Add corpus_metadata to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Provide a standard test configuration."""
    return {
        "enabled": True,
        "run_id": "test_run_001",
        "pipeline_version": "test_v1.0.0",
        "confidence_threshold": 0.5,
        "cache": {
            "enabled": False,  # Disable caching in tests by default
            "directory": "/tmp/test_cache",
            "ttl_hours": 1,
        },
        "rate_limit_per_second": 100,  # High limit for tests
        "timeout_seconds": 5,
    }


@pytest.fixture
def temp_cache_dir() -> Generator[Path, None, None]:
    """Provide a temporary cache directory that's cleaned up after tests."""
    with tempfile.TemporaryDirectory(prefix="test_cache_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def cache_enabled_config(temp_cache_dir: Path) -> Dict[str, Any]:
    """Configuration with caching enabled in a temp directory."""
    return {
        "enabled": True,
        "cache": {
            "enabled": True,
            "directory": str(temp_cache_dir),
            "ttl_hours": 1,
        },
    }


# =============================================================================
# MOCK API RESPONSES
# =============================================================================

@pytest.fixture
def mock_pubtator_disease_response() -> List[Dict[str, Any]]:
    """Mock PubTator3 autocomplete response for diseases."""
    return [
        {
            "_id": "disease_001",
            "name": "Type 2 Diabetes Mellitus",
            "db": "ncbi_mesh",
            "db_id": "D003924",
            "biotype": "disease",
        },
        {
            "_id": "disease_002",
            "name": "Diabetes Mellitus",
            "db": "ncbi_mesh",
            "db_id": "D003920",
            "biotype": "disease",
        },
        {
            "_id": "disease_003",
            "name": "Diabetes Complications",
            "db": "ncbi_mesh",
            "db_id": "D048909",
            "biotype": "disease",
        },
    ]


@pytest.fixture
def mock_pubtator_chemical_response() -> List[Dict[str, Any]]:
    """Mock PubTator3 autocomplete response for chemicals/drugs."""
    return [
        {
            "_id": "chemical_001",
            "name": "Metformin",
            "db": "ncbi_mesh",
            "db_id": "D008687",
            "biotype": "chemical",
        },
        {
            "_id": "chemical_002",
            "name": "Metformin Hydrochloride",
            "db": "ncbi_mesh",
            "db_id": "C011617",
            "biotype": "chemical",
        },
    ]


@pytest.fixture
def mock_clinicaltrials_response() -> Dict[str, Any]:
    """Mock ClinicalTrials.gov API response."""
    return {
        "studies": [
            {
                "protocolSection": {
                    "identificationModule": {
                        "nctId": "NCT04817618",
                        "officialTitle": "A Phase 3, Randomized, Double-Blind Study",
                        "briefTitle": "Study of Drug X in Patients",
                        "acronym": "TESTRIAL",
                    },
                    "conditionsModule": {
                        "conditions": ["Type 2 Diabetes", "Obesity"],
                    },
                    "armsInterventionsModule": {
                        "interventions": [
                            {"name": "Drug X 100mg"},
                            {"name": "Placebo"},
                        ],
                    },
                    "statusModule": {
                        "overallStatus": "Recruiting",
                    },
                    "designModule": {
                        "phases": ["Phase 3"],
                    },
                },
            }
        ],
    }


# =============================================================================
# MOCK CLIENTS
# =============================================================================

@pytest.fixture
def mock_pubtator_client(
    mock_pubtator_disease_response: List[Dict[str, Any]],
    mock_pubtator_chemical_response: List[Dict[str, Any]],
) -> MagicMock:
    """Mock PubTator3Client with predefined responses."""
    client = MagicMock()

    def autocomplete_side_effect(term: str, entity_type: str = "disease"):
        if entity_type == "disease":
            return mock_pubtator_disease_response
        elif entity_type == "chemical":
            return mock_pubtator_chemical_response
        return []

    client.autocomplete.side_effect = autocomplete_side_effect
    client.search_entity.return_value = []
    return client


@pytest.fixture
def mock_clinicaltrials_client(
    mock_clinicaltrials_response: Dict[str, Any],
) -> MagicMock:
    """Mock ClinicalTrialsGovClient with predefined responses."""
    client = MagicMock()

    def get_trial_info_side_effect(nct_id: str):
        from E_normalization.E06_nct_enricher import NCTTrialInfo

        if nct_id.upper() == "NCT04817618":
            return NCTTrialInfo(
                nct_id="NCT04817618",
                official_title="A Phase 3, Randomized, Double-Blind Study",
                brief_title="Study of Drug X in Patients",
                acronym="TESTRIAL",
                conditions=["Type 2 Diabetes", "Obesity"],
                interventions=["Drug X 100mg", "Placebo"],
                status="Recruiting",
                phase="Phase 3",
            )
        return None

    client.get_trial_info.side_effect = get_trial_info_side_effect
    return client


# =============================================================================
# SAMPLE ENTITY FACTORIES
# =============================================================================

@pytest.fixture
def sample_disease_dict() -> Dict[str, Any]:
    """Sample disease data for testing."""
    return {
        "preferred_label": "Type 2 Diabetes",
        "abbreviation": "T2D",
        "mesh_id": None,
        "identifiers": [],
        "validation_flags": [],
        "confidence": 0.9,
    }


@pytest.fixture
def sample_drug_dict() -> Dict[str, Any]:
    """Sample drug data for testing."""
    return {
        "preferred_name": "Metformin",
        "brand_name": "Glucophage",
        "compound_id": None,
        "mesh_id": None,
        "identifiers": [],
        "validation_flags": [],
        "confidence": 0.85,
    }


@pytest.fixture
def sample_genetic_text() -> str:
    """Sample text with genetic entities for testing."""
    return """
    Patients with GBA c.1226A>G (p.N370S) mutation are eligible.
    Exclusion: rs76763715 carriers with confirmed diagnosis of
    Gaucher disease (ORPHA:355). HPO phenotype: HP:0001744 (splenomegaly).
    """


@pytest.fixture
def sample_clinical_text() -> str:
    """Sample clinical trial text for testing."""
    return """
    This phase 3 study evaluates the efficacy of Drug X 100mg twice daily
    in patients with type 2 diabetes mellitus and inadequate glycemic control.
    Primary endpoint: HbA1c reduction at week 24.
    Adverse events will be monitored throughout the study.
    """


# =============================================================================
# HTTP MOCK FIXTURES
# =============================================================================

@pytest.fixture
def mock_requests_get(
    mock_pubtator_disease_response: List[Dict[str, Any]],
    mock_clinicaltrials_response: Dict[str, Any],
):
    """Mock requests.get for all external API calls."""
    with patch("requests.get") as mock_get:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = mock_pubtator_disease_response

        def get_side_effect(url: str, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()

            if "pubtator" in url:
                resp.json.return_value = mock_pubtator_disease_response
            elif "clinicaltrials.gov" in url:
                resp.json.return_value = mock_clinicaltrials_response
            else:
                resp.json.return_value = {}

            return resp

        mock_get.side_effect = get_side_effect
        yield mock_get


@pytest.fixture
def mock_requests_session(
    mock_pubtator_disease_response: List[Dict[str, Any]],
    mock_clinicaltrials_response: Dict[str, Any],
):
    """Mock requests.Session for connection pooling scenarios."""
    with patch("requests.Session") as mock_session_class:
        session = MagicMock()

        def request_side_effect(method: str, url: str, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()

            if "clinicaltrials.gov" in url:
                resp.json.return_value = mock_clinicaltrials_response
            else:
                resp.json.return_value = {}

            return resp

        session.request.side_effect = request_side_effect
        session.get.side_effect = lambda url, **kwargs: request_side_effect("GET", url, **kwargs)
        mock_session_class.return_value = session
        yield session


# =============================================================================
# UTILITY FIXTURES
# =============================================================================

@pytest.fixture
def assert_no_warnings(caplog: pytest.LogCaptureFixture):
    """Assert that no warnings were logged during a test."""
    yield
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 0, f"Unexpected warnings: {[r.message for r in warnings]}"


@pytest.fixture
def capture_logs(caplog: pytest.LogCaptureFixture):
    """Capture log messages for assertions."""
    caplog.set_level("DEBUG")
    yield caplog


# =============================================================================
# TEST MARKERS
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests requiring external services"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: marks tests requiring GPU"
    )

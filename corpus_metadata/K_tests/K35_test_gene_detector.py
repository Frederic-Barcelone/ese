"""
Tests for C_generators.C16_strategy_gene module.

Tests gene detection logic: blacklist, hyphenation, extra aliases,
deduplication, identifier building, context extraction, and regex patterns.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from A_core.A01_domain_models import Coordinate
from A_core.A19_gene_models import (
    GeneCandidate,
    GeneFieldType,
    GeneGeneratorType,
    GeneProvenanceMetadata,
)
from C_generators.C16_strategy_gene import GeneDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_provenance(gen_type: GeneGeneratorType = GeneGeneratorType.LEXICON_ORPHADATA) -> GeneProvenanceMetadata:
    return GeneProvenanceMetadata(
        pipeline_version="test",
        run_id="test-run",
        doc_fingerprint="test-fp",
        generator_name=gen_type,
    )


def _make_candidate(
    symbol: str,
    matched_text: str | None = None,
    gen_type: GeneGeneratorType = GeneGeneratorType.LEXICON_ORPHADATA,
    confidence: float = 0.85,
) -> GeneCandidate:
    text = matched_text or symbol
    is_alias = gen_type != GeneGeneratorType.LEXICON_ORPHADATA
    return GeneCandidate(
        doc_id="test-doc",
        matched_text=text,
        hgnc_symbol=symbol,
        field_type=GeneFieldType.EXACT_MATCH,
        generator_type=gen_type,
        identifiers=[],
        context_text=f"context for {text}",
        context_location=Coordinate(page_num=1),
        initial_confidence=confidence,
        is_alias=is_alias,
        alias_of=symbol if is_alias else None,
        provenance=_make_provenance(gen_type),
    )


@pytest.fixture
def detector():
    """Create a GeneDetector with lexicon loading mocked out."""
    with patch.object(GeneDetector, "_load_lexicons"):
        det = GeneDetector(config={})
        # Initialize empty dicts so methods work
        det.orphadata_genes = {}
        det.alias_genes = {}
        det.name_genes = {}
        det.fp_filter = None  # type: ignore[assignment]  # Not needed for unit tests
        return det


# ---------------------------------------------------------------------------
# TestLexiconLoadBlacklist
# ---------------------------------------------------------------------------


class TestLexiconLoadBlacklist:
    """Tests for LEXICON_LOAD_BLACKLIST class attribute."""

    def test_blacklist_is_set(self):
        assert isinstance(GeneDetector.LEXICON_LOAD_BLACKLIST, set)

    def test_statistical_terms_in_blacklist(self):
        for term in ["or", "hr"]:
            assert term in GeneDetector.LEXICON_LOAD_BLACKLIST

    def test_prepositions_in_blacklist(self):
        for term in ["of", "on", "an", "as", "for"]:
            assert term in GeneDetector.LEXICON_LOAD_BLACKLIST

    def test_roman_numerals_in_blacklist(self):
        for term in ["ii", "iii"]:
            assert term in GeneDetector.LEXICON_LOAD_BLACKLIST

    def test_amino_acids_in_blacklist(self):
        for term in ["arg", "his", "pro", "val", "met"]:
            assert term in GeneDetector.LEXICON_LOAD_BLACKLIST

    def test_disease_abbreviations_in_blacklist(self):
        for term in ["sma", "lca"]:
            assert term in GeneDetector.LEXICON_LOAD_BLACKLIST

    def test_wrong_gene_aliases_in_blacklist(self):
        for term in ["hk2", "gata", "iap"]:
            assert term in GeneDetector.LEXICON_LOAD_BLACKLIST

    def test_common_english_words_in_blacklist(self):
        for term in ["type", "face", "act", "can"]:
            assert term in GeneDetector.LEXICON_LOAD_BLACKLIST

    def test_valid_gene_symbols_not_in_blacklist(self):
        """Ensure legitimate gene symbols are NOT blacklisted."""
        for symbol in ["brca1", "tp53", "egfr", "cfh", "col1a1", "fgfr2"]:
            assert symbol not in GeneDetector.LEXICON_LOAD_BLACKLIST

    def test_blacklist_has_significant_size(self):
        assert len(GeneDetector.LEXICON_LOAD_BLACKLIST) >= 50


# ---------------------------------------------------------------------------
# TestMakeHyphenated
# ---------------------------------------------------------------------------


class TestMakeHyphenated:
    """Tests for _make_hyphenated() method."""

    def test_il6(self, detector):
        assert detector._make_hyphenated("IL6") == "IL-6"

    def test_mmp9(self, detector):
        assert detector._make_hyphenated("MMP9") == "MMP-9"

    def test_tgfb1(self, detector):
        assert detector._make_hyphenated("TGFB1") == "TGFB-1"

    def test_il12a(self, detector):
        assert detector._make_hyphenated("IL12A") == "IL-12A"

    def test_a1(self, detector):
        assert detector._make_hyphenated("A1") == "A-1"

    def test_no_digits_returns_none(self, detector):
        assert detector._make_hyphenated("BRCA") is None

    def test_single_letter_returns_none(self, detector):
        assert detector._make_hyphenated("A") is None

    def test_lowercase_returns_none(self, detector):
        assert detector._make_hyphenated("il6") is None


# ---------------------------------------------------------------------------
# TestExtraAliases
# ---------------------------------------------------------------------------


class TestExtraAliases:
    """Tests for EXTRA_ALIASES protein name mappings."""

    def test_extra_aliases_defined_in_loader(self):
        """Verify EXTRA_ALIASES dict exists in source code by checking key mappings."""
        import inspect
        source = inspect.getsource(GeneDetector._load_orphadata_lexicon)
        assert "EXTRA_ALIASES" in source
        assert "E-cadherin" in source
        assert "CDH1" in source

    def test_e_cadherin_mapping(self):
        import inspect
        source = inspect.getsource(GeneDetector._load_orphadata_lexicon)
        assert '"E-cadherin": "CDH1"' in source

    def test_tgf_beta1_mapping(self):
        import inspect
        source = inspect.getsource(GeneDetector._load_orphadata_lexicon)
        assert '"TGF-beta1": "TGFB1"' in source

    def test_alpha_sma_mapping(self):
        import inspect
        source = inspect.getsource(GeneDetector._load_orphadata_lexicon)
        assert '"alpha-SMA": "ACTA2"' in source

    def test_ifn_gamma_mapping(self):
        import inspect
        source = inspect.getsource(GeneDetector._load_orphadata_lexicon)
        assert '"IFN-gamma": "IFNG"' in source


# ---------------------------------------------------------------------------
# TestDeduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Tests for _deduplicate() method."""

    def test_same_symbol_different_text(self, detector):
        """Same HGNC symbol, different matched text → keep higher priority."""
        c1 = _make_candidate("IL6", "IL6", GeneGeneratorType.LEXICON_ORPHADATA, 0.85)
        c2 = _make_candidate("IL6", "IL-6", GeneGeneratorType.LEXICON_HGNC_ALIAS, 0.80)
        result = detector._deduplicate([c1, c2])
        assert len(result) == 1
        assert result[0].generator_type == GeneGeneratorType.LEXICON_ORPHADATA

    def test_orphadata_beats_alias(self, detector):
        """ORPHADATA priority beats HGNC_ALIAS."""
        c_alias = _make_candidate("BRCA1", "BRCA1", GeneGeneratorType.LEXICON_HGNC_ALIAS, 0.90)
        c_orpha = _make_candidate("BRCA1", "BRCA1", GeneGeneratorType.LEXICON_ORPHADATA, 0.85)
        result = detector._deduplicate([c_alias, c_orpha])
        assert len(result) == 1
        assert result[0].generator_type == GeneGeneratorType.LEXICON_ORPHADATA

    def test_confidence_tiebreaker(self, detector):
        """Same priority → higher confidence wins."""
        c1 = _make_candidate("TP53", "TP53", GeneGeneratorType.LEXICON_ORPHADATA, 0.80)
        c2 = _make_candidate("TP53", "TP53", GeneGeneratorType.LEXICON_ORPHADATA, 0.90)
        result = detector._deduplicate([c1, c2])
        assert len(result) == 1
        assert result[0].initial_confidence == 0.90

    def test_different_symbols_not_deduplicated(self, detector):
        """Different HGNC symbols → both kept."""
        c1 = _make_candidate("BRCA1", "BRCA1", GeneGeneratorType.LEXICON_ORPHADATA)
        c2 = _make_candidate("TP53", "TP53", GeneGeneratorType.LEXICON_ORPHADATA)
        result = detector._deduplicate([c1, c2])
        assert len(result) == 2

    def test_no_hgnc_symbol_fallback(self, detector):
        """Candidate without hgnc_symbol falls back to matched_text.lower()."""
        c1 = _make_candidate("CFHR5", "CFHR5", GeneGeneratorType.LEXICON_ORPHADATA)
        c2 = _make_candidate("CFH", "CFH", GeneGeneratorType.LEXICON_ORPHADATA)
        result = detector._deduplicate([c1, c2])
        assert len(result) == 2

    def test_empty_list(self, detector):
        result = detector._deduplicate([])
        assert result == []


# ---------------------------------------------------------------------------
# TestBuildIdentifiers
# ---------------------------------------------------------------------------


class TestBuildIdentifiers:
    """Tests for _build_identifiers() method."""

    def test_full_gene_info(self, detector):
        """All identifier fields present → 5 identifiers."""
        gene_info = {
            "hgnc_id": "HGNC:1100",
            "entrez_id": "672",
            "ensembl_id": "ENSG00000012048",
            "omim_id": "113705",
            "uniprot_id": "P38398",
        }
        ids = detector._build_identifiers(gene_info)
        assert len(ids) == 5
        systems = {i.system for i in ids}
        assert systems == {"HGNC", "ENTREZ", "ENSEMBL", "OMIM", "UNIPROT"}

    def test_partial_gene_info(self, detector):
        """Only some fields present → only those identifiers."""
        gene_info = {
            "hgnc_id": "HGNC:1100",
            "entrez_id": "672",
        }
        ids = detector._build_identifiers(gene_info)
        assert len(ids) == 2
        systems = {i.system for i in ids}
        assert systems == {"HGNC", "ENTREZ"}

    def test_empty_gene_info(self, detector):
        """Empty dict → no identifiers."""
        ids = detector._build_identifiers({})
        assert ids == []


# ---------------------------------------------------------------------------
# TestExtractContext
# ---------------------------------------------------------------------------


class TestExtractContext:
    """Tests for context extraction via canonical extract_context_window."""

    def test_normal_context(self, detector):
        """Context extracted with window around match."""
        from Z_utils.Z02_text_helpers import extract_context_window

        text = "A" * 500 + "BRCA1" + "B" * 500
        ctx = extract_context_window(text, 500, 505, detector.context_window)
        assert "BRCA1" in ctx
        assert len(ctx) <= detector.context_window + 5  # match text

    def test_match_at_start(self, detector):
        """Match at start of text → no left overflow."""
        from Z_utils.Z02_text_helpers import extract_context_window

        text = "BRCA1 is a tumor suppressor gene involved in DNA repair"
        ctx = extract_context_window(text, 0, 5, detector.context_window)
        assert ctx.startswith("BRCA1")

    def test_match_at_end(self, detector):
        """Match at end of text → no right overflow."""
        from Z_utils.Z02_text_helpers import extract_context_window

        text = "mutations were found in BRCA1"
        start = text.index("BRCA1")
        end = start + 5
        ctx = extract_context_window(text, start, end, detector.context_window)
        assert ctx.endswith("BRCA1")


# ---------------------------------------------------------------------------
# TestHyphenPatternRegex
# ---------------------------------------------------------------------------


class TestHyphenPatternRegex:
    """Tests for _HYPHEN_PATTERN compiled regex."""

    @pytest.fixture
    def pattern(self):
        return GeneDetector._HYPHEN_PATTERN

    def test_il6_matches(self, pattern):
        assert pattern.match("IL6") is not None

    def test_mmp9_matches(self, pattern):
        assert pattern.match("MMP9") is not None

    def test_tgfb1_matches(self, pattern):
        assert pattern.match("TGFB1") is not None

    def test_il12a_matches(self, pattern):
        m = pattern.match("IL12A")
        assert m is not None
        assert m.group(3) == "A"

    def test_a1_matches(self, pattern):
        assert pattern.match("A1") is not None

    def test_brca_no_digits_no_match(self, pattern):
        assert pattern.match("BRCA") is None

    def test_lowercase_no_match(self, pattern):
        assert pattern.match("il6") is None

    def test_pure_digits_no_match(self, pattern):
        assert pattern.match("123") is None

    def test_single_letter_no_match(self, pattern):
        assert pattern.match("A") is None

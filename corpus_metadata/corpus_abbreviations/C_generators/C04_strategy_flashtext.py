# corpus_metadata/corpus_abbreviations/C_generators/C04_strategy_flashtext.py

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from flashtext import KeywordProcessor

from A_core.A02_interfaces import BaseCandidateGenerator

# =============================================================================
# LIGHT NOISE FILTERING (High Recall - Let Validation Layer Judge)
# =============================================================================
# Philosophy: Generators should be EXHAUSTIVE. Only block OBVIOUS noise.
# Claude (D_validation) will handle borderline cases with context awareness.

# Obvious non-abbreviations: single letters, basic English function words
OBVIOUS_NOISE: set = {
    # Single letters (never valid abbreviations alone)
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    # Basic English function words (articles, prepositions, conjunctions)
    "an", "as", "at", "be", "by", "do", "go", "he", "if", "in", "is",
    "it", "me", "my", "no", "of", "on", "or", "so", "to", "up", "us", "we",
    "the", "and", "for", "but", "not", "are", "was", "were", "been",
    "have", "has", "had", "will", "would", "could", "should",
    "this", "that", "these", "those", "with", "from", "into",
    # Citation artifacts
    "et", "al",
}

# Minimum length (allow 2-char if uppercase like CT, MR, IV)
MIN_ABBREV_LENGTH = 2
from A_core.A01_domain_models import (
    Candidate,
    Coordinate,
    FieldType,
    GeneratorType,
    LexiconIdentifier,
    ProvenanceMetadata,
)
from A_core.A03_provenance import generate_run_id, get_git_revision_hash
from B_parsing.B02_doc_graph import DocumentGraph


class LexiconEntry:
    """Compiled lexicon entry with regex pattern and source provenance."""
    __slots__ = ("sf", "lf", "pattern", "source", "lexicon_ids")

    def __init__(
        self,
        sf: str,
        lf: str,
        pattern: re.Pattern,
        source: str,
        lexicon_ids: Optional[List[Dict[str, str]]] = None,
    ):
        self.sf = sf
        self.lf = lf
        self.pattern = pattern
        self.source = source  # Lexicon file name for provenance
        self.lexicon_ids = lexicon_ids or []  # External IDs [{source, id}, ...]


class RegexLexiconGenerator(BaseCandidateGenerator):
    """
    LEXICON MATCHER using:
    - Regex patterns for abbreviation lexicon (handles spacing/case variations)
    - FlashText for disease lexicon (fast exact matching)
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

        # Lexicon paths
        self.abbrev_lexicon_path = Path(self.config.get(
            "abbrev_lexicon_path",
            "/Users/frederictetard/Projects/ese/ouput_datasources/2025_08_abbreviation_general.json"
        ))
        self.disease_lexicon_path = Path(self.config.get(
            "disease_lexicon_path",
            "/Users/frederictetard/Projects/ese/ouput_datasources/2025_08_lexicon_disease.json"
        ))
        self.orphanet_lexicon_path = Path(self.config.get(
            "orphanet_lexicon_path",
            "/Users/frederictetard/Projects/ese/ouput_datasources/2025_08_orphanet_diseases.json"
        ))
        self.rare_disease_acronyms_path = Path(self.config.get(
            "rare_disease_acronyms_path",
            "/Users/frederictetard/Projects/ese/ouput_datasources/2025_08_rare_disease_acronyms.json"
        ))
        self.umls_abbrev_path = Path(self.config.get(
            "umls_abbrev_path",
            "/Users/frederictetard/Projects/ese/ouput_datasources/2025_08_umls_biological_abbreviations_v5.tsv"
        ))
        self.umls_clinical_path = Path(self.config.get(
            "umls_clinical_path",
            "/Users/frederictetard/Projects/ese/ouput_datasources/2025_08_umls_clinical_abbreviations_v5.tsv"
        ))
        self.anca_lexicon_path = Path(self.config.get(
            "anca_lexicon_path",
            "/Users/frederictetard/Projects/ese/ouput_datasources/disease_lexicon_anca.json"
        ))
        self.igan_lexicon_path = Path(self.config.get(
            "igan_lexicon_path",
            "/Users/frederictetard/Projects/ese/ouput_datasources/disease_lexicon_igan.json"
        ))
        self.pah_lexicon_path = Path(self.config.get(
            "pah_lexicon_path",
            "/Users/frederictetard/Projects/ese/ouput_datasources/disease_lexicon_pah.json"
        ))
        
        self.context_window = int(self.config.get("context_window", 80))

        # Abbreviation entries (regex-based)
        self.abbrev_entries: List[LexiconEntry] = []

        # Disease/entity entries (FlashText-based)
        self.entity_kp = KeywordProcessor(case_sensitive=False)
        self.entity_canonical: Dict[str, str] = {}  # matched_term -> canonical_name
        self.entity_source: Dict[str, str] = {}     # matched_term -> source file
        self.entity_ids: Dict[str, List[Dict[str, str]]] = {}  # matched_term -> [{source, id}, ...]

        # Provenance
        self.pipeline_version = str(self.config.get("pipeline_version") or get_git_revision_hash())
        self.run_id = str(self.config.get("run_id") or generate_run_id("ABBR"))
        self.doc_fingerprint_default = str(self.config.get("doc_fingerprint") or "unknown-doc-fingerprint")

        # Load lexicons
        self._load_abbrev_lexicon(self.abbrev_lexicon_path)
        self._load_disease_lexicon(self.disease_lexicon_path)
        self._load_orphanet_lexicon(self.orphanet_lexicon_path)
        self._load_rare_disease_acronyms(self.rare_disease_acronyms_path)
        self._load_umls_tsv(self.umls_abbrev_path)
        self._load_umls_tsv(self.umls_clinical_path)
        self._load_anca_lexicon(self.anca_lexicon_path)
        self._load_igan_lexicon(self.igan_lexicon_path)
        self._load_pah_lexicon(self.pah_lexicon_path)

    @property
    def generator_type(self) -> GeneratorType:
        return GeneratorType.LEXICON_MATCH

    def extract(self, doc: DocumentGraph) -> List[Candidate]:
        out: List[Candidate] = []
        seen: Set[Tuple[str, str]] = set()  # (SF_upper, LF_lower) dedup

        for block in doc.iter_linear_blocks(skip_header_footer=True):
            text = (block.text or "").strip()
            if not text:
                continue

            # 1) Abbreviation matches (regex)
            for entry in self.abbrev_entries:
                # Skip blacklisted/invalid abbreviations
                if not self._is_valid_match(entry.sf):
                    continue

                for match in entry.pattern.finditer(text):
                    start, end = match.start(), match.end()

                    key = (entry.sf.upper(), entry.lf.lower())
                    if key in seen:
                        continue
                    seen.add(key)

                    # Convert dict lexicon_ids to LexiconIdentifier objects
                    lex_ids = None
                    if entry.lexicon_ids:
                        lex_ids = [
                            LexiconIdentifier(source=d["source"], id=d["id"])
                            for d in entry.lexicon_ids
                        ]

                    out.append(self._make_candidate(
                        doc=doc,
                        block=block,
                        short_form=entry.sf,
                        long_form=entry.lf,
                        start=start,
                        end=end,
                        text=text,
                        rule_version="abbrev_regex::v1",
                        lexicon_source=entry.source,
                        lexicon_ids=lex_ids,
                    ))

            # 2) Entity matches (FlashText - diseases, orphanet terms)
            entity_hits = self.entity_kp.extract_keywords(text, span_info=True)
            for matched_term, start, end in entity_hits:
                # Skip blacklisted/invalid terms
                if not self._is_valid_match(matched_term):
                    continue

                canonical = self.entity_canonical.get(matched_term, matched_term)
                source = self.entity_source.get(matched_term, "unknown")

                key = (matched_term.upper(), canonical.lower())
                if key in seen:
                    continue
                seen.add(key)

                # Convert dict lexicon_ids to LexiconIdentifier objects
                entity_lex_ids = None
                raw_ids = self.entity_ids.get(matched_term, [])
                if raw_ids:
                    entity_lex_ids = [
                        LexiconIdentifier(source=d["source"], id=d["id"])
                        for d in raw_ids
                    ]

                out.append(self._make_candidate(
                    doc=doc,
                    block=block,
                    short_form=matched_term,
                    long_form=canonical,
                    start=start,
                    end=end,
                    text=text,
                    rule_version="entity_exact::v1",
                    lexicon_source=source,
                    lexicon_ids=entity_lex_ids,
                ))

        return out

    def _make_candidate(
        self,
        doc: DocumentGraph,
        block,
        short_form: str,
        long_form: str,
        start: int,
        end: int,
        text: str,
        rule_version: str,
        lexicon_source: str,
        lexicon_ids: Optional[List[LexiconIdentifier]] = None,
    ) -> Candidate:
        ctx = self._make_context(text, start, end)

        loc = Coordinate(
            page_num=int(block.page_num),
            block_id=str(block.id),
            bbox=block.bbox,
        )

        prov = ProvenanceMetadata(
            pipeline_version=self.pipeline_version,
            run_id=self.run_id,
            doc_fingerprint=self.doc_fingerprint_default,
            generator_name=self.generator_type,
            rule_version=rule_version,
            lexicon_source=lexicon_source,
            lexicon_ids=lexicon_ids,
        )

        return Candidate(
            doc_id=doc.doc_id,
            field_type=FieldType.DEFINITION_PAIR,
            generator_type=self.generator_type,
            short_form=short_form,
            long_form=long_form,
            context_text=ctx,
            context_location=loc,
            initial_confidence=0.95,
            provenance=prov,
        )

    def _load_abbrev_lexicon(self, path: Path) -> None:
        if not path.exists():
            print(f"Abbreviation lexicon not found: {path}")
            return

        print(f"Loading abbreviation lexicon: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))

        loaded = 0
        failed = 0

        for sf, entry in data.items():
            if not sf or not isinstance(entry, dict):
                continue

            lf = entry.get("canonical_expansion")
            regex_str = entry.get("regex")

            if not lf or not regex_str:
                continue

            try:
                case_insensitive = bool(entry.get("case_insensitive", False))
                flags = re.IGNORECASE if case_insensitive else 0
                pattern = re.compile(regex_str, flags)

                self.abbrev_entries.append(LexiconEntry(
                    sf=sf, lf=lf, pattern=pattern, source=path.name
                ))
                loaded += 1
            except re.error as e:
                failed += 1
                if failed <= 5:
                    print(f"  Bad regex for '{sf}': {e}")

        print(f"Loaded {loaded} abbreviation patterns from {path.name}")
        if failed:
            print(f"  {failed} patterns failed to compile")

    def _load_disease_lexicon(self, path: Path) -> None:
        if not path.exists():
            print(f"Disease lexicon not found: {path}")
            return

        print(f"Loading disease lexicon: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        source = path.name

        loaded = 0

        for entry in data:
            if not isinstance(entry, dict):
                continue

            label = (entry.get("label") or "").strip()
            if not label or len(label) < 3:
                continue

            # Extract external IDs (Orphanet, MONDO, etc.)
            sources_list = entry.get("sources", [])
            lexicon_ids = [
                {"source": s.get("source", ""), "id": s.get("id", "")}
                for s in sources_list if isinstance(s, dict) and s.get("id")
            ]

            self.entity_kp.add_keyword(label, label)
            self.entity_canonical[label] = label
            self.entity_source[label] = source
            self.entity_ids[label] = lexicon_ids
            loaded += 1

        print(f"Loaded {loaded} disease terms from {path.name}")

    def _load_orphanet_lexicon(self, path: Path) -> None:
        if not path.exists():
            print(f"Orphanet lexicon not found: {path}")
            return

        print(f"Loading Orphanet lexicon: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        source = path.name

        loaded = 0

        for entry in data:
            if not isinstance(entry, dict):
                continue

            name = (entry.get("name") or "").strip()
            if not name or len(name) < 3:
                continue

            synonyms = entry.get("synonyms") or []

            # Build lexicon IDs from orphacode
            orphacode = entry.get("orphacode")
            lexicon_ids = []
            if orphacode:
                lexicon_ids.append({"source": "Orphanet", "id": f"ORPHA:{orphacode}"})

            # Register main name
            self.entity_kp.add_keyword(name, name)
            self.entity_canonical[name] = name
            self.entity_source[name] = source
            self.entity_ids[name] = lexicon_ids
            loaded += 1

            # Register synonyms (map to canonical name)
            for syn in synonyms:
                syn = (syn or "").strip()
                if not syn or len(syn) < 2:
                    continue
                self.entity_kp.add_keyword(syn, syn)
                self.entity_canonical[syn] = name  # maps to canonical
                self.entity_source[syn] = source
                self.entity_ids[syn] = lexicon_ids  # Same IDs as canonical
                loaded += 1

        print(f"Loaded {loaded} Orphanet terms from {path.name}")

    def _load_rare_disease_acronyms(self, path: Path) -> None:
        if not path.exists():
            print(f"Rare disease acronyms not found: {path}")
            return

        print(f"Loading rare disease acronyms: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        source = path.name

        loaded = 0

        for acronym, entry in data.items():
            if not acronym or not isinstance(entry, dict):
                continue

            acronym = acronym.strip()
            name = (entry.get("name") or "").strip()

            if not acronym or len(acronym) < 2 or not name:
                continue

            # Build lexicon IDs from orphacode
            orphacode = entry.get("orphacode")
            lexicon_ids = []
            if orphacode:
                lexicon_ids.append({"source": "Orphanet", "id": f"ORPHA:{orphacode}"})

            self.entity_kp.add_keyword(acronym, acronym)
            self.entity_canonical[acronym] = name
            self.entity_source[acronym] = source
            self.entity_ids[acronym] = lexicon_ids
            loaded += 1

        print(f"Loaded {loaded} rare disease acronyms from {path.name}")

    def _load_umls_tsv(self, path: Path) -> None:
        if not path.exists():
            print(f"UMLS file not found: {path}")
            return

        print(f"Loading UMLS: {path}")
        source = path.name

        loaded = 0

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                abbrev = (row.get("Abbreviation") or "").strip()
                expansion = (row.get("Expansion") or "").strip()
                top_source = (row.get("TopSource") or "").strip()

                if not abbrev or len(abbrev) < 2 or not expansion:
                    continue

                # Build lexicon IDs from UMLS source
                lexicon_ids = []
                if top_source:
                    lexicon_ids.append({"source": f"UMLS:{top_source}", "id": abbrev})

                self.entity_kp.add_keyword(abbrev, abbrev)
                self.entity_canonical[abbrev] = expansion
                self.entity_source[abbrev] = source
                self.entity_ids[abbrev] = lexicon_ids
                loaded += 1

        print(f"Loaded {loaded} terms from {path.name}")

    def _extract_identifiers(self, identifiers: Dict) -> List[Dict[str, str]]:
        """Extract lexicon IDs from an identifiers dict."""
        lexicon_ids = []
        if not identifiers or not isinstance(identifiers, dict):
            return lexicon_ids

        # Map of identifier keys to source names
        source_map = {
            "ORPHA": "Orphanet",
            "ICD11": "ICD-11",
            "ICD10": "ICD-10",
            "ICD10CM": "ICD-10-CM",
            "SNOMED_CT": "SNOMED-CT",
            "MESH": "MeSH",
            "UMLS": "UMLS",
            "UMLS_CUI": "UMLS",
            "MONDO": "MONDO",
        }
        for key, source_name in source_map.items():
            if key in identifiers and identifiers[key]:
                id_val = identifiers[key]
                # Format the ID properly
                if key == "ORPHA":
                    id_val = f"ORPHA:{id_val}"
                elif key == "MESH":
                    id_val = f"MESH:{id_val}"
                lexicon_ids.append({"source": source_name, "id": str(id_val)})

        return lexicon_ids

    def _load_anca_lexicon(self, path: Path) -> None:
        if not path.exists():
            print(f"ANCA lexicon not found: {path}")
            return

        print(f"Loading ANCA lexicon: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        source = path.name

        loaded = 0

        # 1) Diseases section
        diseases = data.get("diseases", {})
        for disease_key, entry in diseases.items():
            if not isinstance(entry, dict):
                continue

            preferred = (entry.get("preferred_label") or "").strip()
            abbrev = (entry.get("abbreviation") or "").strip()
            synonyms = entry.get("synonyms") or []
            identifiers = entry.get("identifiers", {})

            if not preferred:
                continue

            # Extract lexicon IDs
            lexicon_ids = self._extract_identifiers(identifiers)

            # Register preferred label
            self.entity_kp.add_keyword(preferred, preferred)
            self.entity_canonical[preferred] = preferred
            self.entity_source[preferred] = source
            self.entity_ids[preferred] = lexicon_ids
            loaded += 1

            # Register abbreviation
            if abbrev and len(abbrev) >= 2:
                self.entity_kp.add_keyword(abbrev, abbrev)
                self.entity_canonical[abbrev] = preferred
                self.entity_source[abbrev] = source
                self.entity_ids[abbrev] = lexicon_ids
                loaded += 1

            # Register synonyms
            for syn in synonyms:
                syn = (syn or "").strip()
                if not syn or len(syn) < 2 or syn == abbrev:
                    continue
                self.entity_kp.add_keyword(syn, syn)
                self.entity_canonical[syn] = preferred
                self.entity_source[syn] = source
                self.entity_ids[syn] = lexicon_ids
                loaded += 1

        # 2) Abbreviation expansions section
        abbrev_expansions = data.get("abbreviation_expansions", {})
        for abbrev, entry in abbrev_expansions.items():
            if not abbrev or not isinstance(entry, dict):
                continue

            preferred = (entry.get("preferred") or "").strip()
            if not preferred or len(abbrev) < 2:
                continue

            self.entity_kp.add_keyword(abbrev, abbrev)
            self.entity_canonical[abbrev] = preferred
            self.entity_source[abbrev] = source
            self.entity_ids[abbrev] = []  # No IDs for abbreviation expansions
            loaded += 1

        # 3) Composite terms section
        composite = data.get("composite_terms", {})
        for term, entry in composite.items():
            if not term or not isinstance(entry, dict):
                continue

            expansion = (entry.get("expansion") or "").strip()
            if not expansion or len(term) < 2:
                continue

            self.entity_kp.add_keyword(term, term)
            self.entity_canonical[term] = expansion
            self.entity_source[term] = source
            self.entity_ids[term] = []  # No IDs for composite terms
            loaded += 1

        print(f"Loaded {loaded} ANCA terms from {path.name}")


    def _load_igan_lexicon(self, path: Path) -> None:
        if not path.exists():
            print(f"Warning: IgAN lexicon not found: {path}")
            return

        print(f"Loading IgAN lexicon: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        source = path.name

        loaded = 0

        # 1) Diseases section
        diseases = data.get("diseases", {})
        for disease_key, entry in diseases.items():
            if not isinstance(entry, dict):
                continue

            preferred = (entry.get("preferred_label") or "").strip()
            abbrev = (entry.get("abbreviation") or "").strip()
            synonyms = entry.get("synonyms") or []
            identifiers = entry.get("identifiers", {})

            if not preferred:
                continue

            # Extract lexicon IDs
            lexicon_ids = self._extract_identifiers(identifiers)

            # Register preferred label
            self.entity_kp.add_keyword(preferred, preferred)
            self.entity_canonical[preferred] = preferred
            self.entity_source[preferred] = source
            self.entity_ids[preferred] = lexicon_ids
            loaded += 1

            # Register abbreviation
            if abbrev and len(abbrev) >= 2:
                self.entity_kp.add_keyword(abbrev, abbrev)
                self.entity_canonical[abbrev] = preferred
                self.entity_source[abbrev] = source
                self.entity_ids[abbrev] = lexicon_ids
                loaded += 1

            # Register synonyms
            for syn in synonyms:
                syn = (syn or "").strip()
                if not syn or len(syn) < 2 or syn == abbrev:
                    continue
                self.entity_kp.add_keyword(syn, syn)
                self.entity_canonical[syn] = preferred
                self.entity_source[syn] = source
                self.entity_ids[syn] = lexicon_ids
                loaded += 1

        # 2) Abbreviation expansions section
        abbrev_expansions = data.get("abbreviation_expansions", {})
        for abbrev, entry in abbrev_expansions.items():
            if not abbrev or not isinstance(entry, dict):
                continue

            preferred = (entry.get("preferred") or "").strip()
            if not preferred or len(abbrev) < 2:
                continue

            self.entity_kp.add_keyword(abbrev, abbrev)
            self.entity_canonical[abbrev] = preferred
            self.entity_source[abbrev] = source
            self.entity_ids[abbrev] = []
            loaded += 1

        # 3) Composite terms section
        composite = data.get("composite_terms", {})
        for term, entry in composite.items():
            if not term or not isinstance(entry, dict):
                continue

            expansion = (entry.get("expansion") or "").strip()
            if not expansion or len(term) < 2:
                continue

            self.entity_kp.add_keyword(term, term)
            self.entity_canonical[term] = expansion
            self.entity_source[term] = source
            self.entity_ids[term] = []
            loaded += 1

        # 4) Renal terms section (specific to IgAN lexicon)
        renal_terms = data.get("renal_terms", {})
        for term, entry in renal_terms.items():
            if not term or not isinstance(entry, dict):
                continue

            expansion = (entry.get("expansion") or "").strip()
            if not expansion or len(term) < 2:
                continue

            self.entity_kp.add_keyword(term, term)
            self.entity_canonical[term] = expansion
            self.entity_source[term] = source
            self.entity_ids[term] = []
            loaded += 1

        print(f"Loaded {loaded} IgAN terms from {path.name}")

    def _load_pah_lexicon(self, path: Path) -> None:
        if not path.exists():
            print(f"Warning: PAH lexicon not found: {path}")
            return

        print(f"Loading PAH lexicon: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        source = path.name

        loaded = 0

        # 1) Diseases section
        diseases = data.get("diseases", {})
        for disease_key, entry in diseases.items():
            if not isinstance(entry, dict):
                continue

            preferred = (entry.get("preferred_label") or "").strip()
            abbrev = (entry.get("abbreviation") or "").strip()
            synonyms = entry.get("synonyms") or []
            identifiers = entry.get("identifiers", {})

            if not preferred:
                continue

            # Extract lexicon IDs once per disease entry
            lexicon_ids = self._extract_identifiers(identifiers)

            # Register preferred label
            self.entity_kp.add_keyword(preferred, preferred)
            self.entity_canonical[preferred] = preferred
            self.entity_source[preferred] = source
            self.entity_ids[preferred] = lexicon_ids
            loaded += 1

            # Register abbreviation
            if abbrev and len(abbrev) >= 2:
                self.entity_kp.add_keyword(abbrev, abbrev)
                self.entity_canonical[abbrev] = preferred
                self.entity_source[abbrev] = source
                self.entity_ids[abbrev] = lexicon_ids
                loaded += 1

            # Register synonyms
            for syn in synonyms:
                syn = (syn or "").strip()
                if not syn or len(syn) < 2 or syn == abbrev:
                    continue
                self.entity_kp.add_keyword(syn, syn)
                self.entity_canonical[syn] = preferred
                self.entity_source[syn] = source
                self.entity_ids[syn] = lexicon_ids
                loaded += 1

        # 2) Abbreviation expansions section
        abbrev_expansions = data.get("abbreviation_expansions", {})
        for abbrev, entry in abbrev_expansions.items():
            if not abbrev or not isinstance(entry, dict):
                continue

            preferred = (entry.get("preferred") or "").strip()
            if not preferred or len(abbrev) < 2:
                continue

            self.entity_kp.add_keyword(abbrev, abbrev)
            self.entity_canonical[abbrev] = preferred
            self.entity_source[abbrev] = source
            self.entity_ids[abbrev] = []
            loaded += 1

        # 3) Composite terms section
        composite = data.get("composite_terms", {})
        for term, entry in composite.items():
            if not term or not isinstance(entry, dict):
                continue

            expansion = (entry.get("expansion") or "").strip()
            if not expansion or len(term) < 2:
                continue

            self.entity_kp.add_keyword(term, term)
            self.entity_canonical[term] = expansion
            self.entity_source[term] = source
            self.entity_ids[term] = []
            loaded += 1

        # 4) Hemodynamic terms section (specific to PAH lexicon)
        hemodynamic_terms = data.get("hemodynamic_terms", {})
        for term, entry in hemodynamic_terms.items():
            if not term or not isinstance(entry, dict):
                continue

            expansion = (entry.get("expansion") or "").strip()
            if not expansion or len(term) < 2:
                continue

            self.entity_kp.add_keyword(term, term)
            self.entity_canonical[term] = expansion
            self.entity_source[term] = source
            self.entity_ids[term] = []
            loaded += 1

        print(f"Loaded {loaded} PAH terms from {path.name}")

    def _make_context(self, text: str, start: int, end: int) -> str:
        left = max(0, start - self.context_window)
        right = min(len(text), end + self.context_window)
        return text[left:right].replace("\n", " ").strip()

    def _is_valid_match(self, term: str) -> bool:
        """
        Light filter - only block OBVIOUS noise.
        Let the Validation layer (Claude) handle borderline cases.
        """
        if not term:
            return False

        term_lower = term.lower().strip()

        # Block obvious noise (single letters, function words)
        if term_lower in OBVIOUS_NOISE:
            return False

        # Minimum length
        if len(term) < MIN_ABBREV_LENGTH:
            return False

        return True
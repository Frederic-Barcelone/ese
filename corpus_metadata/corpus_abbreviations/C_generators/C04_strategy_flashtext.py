# corpus_metadata/corpus_abbreviations/C_generators/C04_strategy_flashtext.py

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from flashtext import KeywordProcessor

from A_core.A02_interfaces import BaseCandidateGenerator

# Blacklist: common English words that are NOT valid abbreviations
# These get matched by FlashText but are noise
COMMON_WORD_BLACKLIST: set = {
    # Single letters
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    # Two-letter common words
    "an", "as", "at", "be", "by", "do", "go", "he", "if", "in", "is",
    "it", "me", "my", "no", "of", "on", "or", "so", "to", "up", "us", "we",
    # Common short words that aren't abbreviations
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
    "her", "was", "one", "our", "out", "day", "get", "has", "him", "his",
    "how", "its", "may", "new", "now", "old", "see", "two", "way", "who",
    "did", "let", "put", "say", "she", "too", "use", "per", "via",
    # Roman numerals (often false positives)
    "ii", "iii", "iv", "vi", "vii", "viii", "ix", "xi", "xii",
    # Units that shouldn't be standalone matches
    "mg", "ml", "kg", "cm", "mm", "hz", "db",
}

# Minimum length for valid abbreviations in FlashText matching
MIN_ABBREV_LENGTH = 2
from A_core.A01_domain_models import (
    Candidate,
    Coordinate,
    FieldType,
    GeneratorType,
    ProvenanceMetadata,
)
from A_core.A03_provenance import generate_run_id, get_git_revision_hash
from B_parsing.B02_doc_graph import DocumentGraph


class LexiconEntry:
    """Compiled lexicon entry with regex pattern."""
    __slots__ = ("sf", "lf", "pattern")

    def __init__(self, sf: str, lf: str, pattern: re.Pattern):
        self.sf = sf
        self.lf = lf
        self.pattern = pattern


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
        
        self.context_window = int(self.config.get("context_window", 80))

        # Abbreviation entries (regex-based)
        self.abbrev_entries: List[LexiconEntry] = []
        
        # Disease/entity entries (FlashText-based)
        self.entity_kp = KeywordProcessor(case_sensitive=False)
        self.entity_canonical: Dict[str, str] = {}  # matched_term -> canonical_name

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

                    out.append(self._make_candidate(
                        doc=doc,
                        block=block,
                        short_form=entry.sf,
                        long_form=entry.lf,
                        start=start,
                        end=end,
                        text=text,
                        rule_version="abbrev_regex::v1",
                    ))

            # 2) Entity matches (FlashText - diseases, orphanet terms)
            entity_hits = self.entity_kp.extract_keywords(text, span_info=True)
            for matched_term, start, end in entity_hits:
                # Skip blacklisted/invalid terms
                if not self._is_valid_match(matched_term):
                    continue

                canonical = self.entity_canonical.get(matched_term, matched_term)

                key = (matched_term.upper(), canonical.lower())
                if key in seen:
                    continue
                seen.add(key)

                out.append(self._make_candidate(
                    doc=doc,
                    block=block,
                    short_form=matched_term,
                    long_form=canonical,
                    start=start,
                    end=end,
                    text=text,
                    rule_version="entity_exact::v1",
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
            print(f"⚠ Abbreviation lexicon not found: {path}")
            return

        print(f"📚 Loading abbreviation lexicon: {path}")
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

                self.abbrev_entries.append(LexiconEntry(sf=sf, lf=lf, pattern=pattern))
                loaded += 1
            except re.error as e:
                failed += 1
                if failed <= 5:
                    print(f"  ⚠ Bad regex for '{sf}': {e}")

        print(f"✓ Loaded {loaded} abbreviation patterns from {path.name}")
        if failed:
            print(f"  ⚠ {failed} patterns failed to compile")

    def _load_disease_lexicon(self, path: Path) -> None:
        if not path.exists():
            print(f"⚠ Disease lexicon not found: {path}")
            return

        print(f"📚 Loading disease lexicon: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))

        loaded = 0

        for entry in data:
            if not isinstance(entry, dict):
                continue

            label = (entry.get("label") or "").strip()
            if not label or len(label) < 3:
                continue

            self.entity_kp.add_keyword(label, label)
            self.entity_canonical[label] = label
            loaded += 1

        print(f"✓ Loaded {loaded} disease terms from {path.name}")

    def _load_orphanet_lexicon(self, path: Path) -> None:
        if not path.exists():
            print(f"⚠ Orphanet lexicon not found: {path}")
            return

        print(f"📚 Loading Orphanet lexicon: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))

        loaded = 0

        for entry in data:
            if not isinstance(entry, dict):
                continue

            name = (entry.get("name") or "").strip()
            if not name or len(name) < 3:
                continue

            synonyms = entry.get("synonyms") or []

            # Register main name
            self.entity_kp.add_keyword(name, name)
            self.entity_canonical[name] = name
            loaded += 1

            # Register synonyms (map to canonical name)
            for syn in synonyms:
                syn = (syn or "").strip()
                if not syn or len(syn) < 2:
                    continue
                self.entity_kp.add_keyword(syn, syn)
                self.entity_canonical[syn] = name  # maps to canonical
                loaded += 1

        print(f"✓ Loaded {loaded} Orphanet terms from {path.name}")

    def _load_rare_disease_acronyms(self, path: Path) -> None:
        if not path.exists():
            print(f"⚠ Rare disease acronyms not found: {path}")
            return

        print(f"📚 Loading rare disease acronyms: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))

        loaded = 0

        for acronym, entry in data.items():
            if not acronym or not isinstance(entry, dict):
                continue

            acronym = acronym.strip()
            name = (entry.get("name") or "").strip()

            if not acronym or len(acronym) < 2 or not name:
                continue

            self.entity_kp.add_keyword(acronym, acronym)
            self.entity_canonical[acronym] = name
            loaded += 1

        print(f"✓ Loaded {loaded} rare disease acronyms from {path.name}")

    def _load_umls_tsv(self, path: Path) -> None:
        if not path.exists():
            print(f"⚠ UMLS file not found: {path}")
            return

        print(f"📚 Loading UMLS: {path}")

        loaded = 0

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                abbrev = (row.get("Abbreviation") or "").strip()
                expansion = (row.get("Expansion") or "").strip()

                if not abbrev or len(abbrev) < 2 or not expansion:
                    continue

                self.entity_kp.add_keyword(abbrev, abbrev)
                self.entity_canonical[abbrev] = expansion
                loaded += 1

        print(f"✓ Loaded {loaded} terms from {path.name}")

    def _load_anca_lexicon(self, path: Path) -> None:
        if not path.exists():
            print(f"⚠ ANCA lexicon not found: {path}")
            return

        print(f"📚 Loading ANCA lexicon: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))

        loaded = 0

        # 1) Diseases section
        diseases = data.get("diseases", {})
        for disease_key, entry in diseases.items():
            if not isinstance(entry, dict):
                continue

            preferred = (entry.get("preferred_label") or "").strip()
            abbrev = (entry.get("abbreviation") or "").strip()
            synonyms = entry.get("synonyms") or []

            if not preferred:
                continue

            # Register preferred label
            self.entity_kp.add_keyword(preferred, preferred)
            self.entity_canonical[preferred] = preferred
            loaded += 1

            # Register abbreviation
            if abbrev and len(abbrev) >= 2:
                self.entity_kp.add_keyword(abbrev, abbrev)
                self.entity_canonical[abbrev] = preferred
                loaded += 1

            # Register synonyms
            for syn in synonyms:
                syn = (syn or "").strip()
                if not syn or len(syn) < 2 or syn == abbrev:
                    continue
                self.entity_kp.add_keyword(syn, syn)
                self.entity_canonical[syn] = preferred
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
            loaded += 1

        print(f"✓ Loaded {loaded} ANCA terms from {path.name}")

    def _make_context(self, text: str, start: int, end: int) -> str:
        left = max(0, start - self.context_window)
        right = min(len(text), end + self.context_window)
        return text[left:right].replace("\n", " ").strip()

    def _is_valid_match(self, term: str) -> bool:
        """Filter out noise: common words, too-short terms, etc."""
        if not term:
            return False

        term_lower = term.lower().strip()

        # Check blacklist
        if term_lower in COMMON_WORD_BLACKLIST:
            return False

        # Minimum length (allow 2-char if all uppercase like "IV", "HR")
        if len(term) < MIN_ABBREV_LENGTH:
            return False

        # 2-char terms must be uppercase to be valid abbreviations
        if len(term) == 2 and not term.isupper():
            return False

        return True
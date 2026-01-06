# corpus_metadata/corpus_extraction/C_generators/C04_strategy_flashtext.py

from __future__ import annotations

import csv
import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from flashtext import KeywordProcessor

# scispacy for biomedical NER (identifies entities like C3G, eGFR, KDIGO)
try:
    import spacy
    from scispacy.abbreviation import AbbreviationDetector  # noqa: F401
    from scispacy.linking import EntityLinker  # noqa: F401

    SCISPACY_AVAILABLE = True

    # Suppress scispacy warning about empty matcher patterns (W036)
    # This occurs when abbreviation_detector has no global patterns defined
    warnings.filterwarnings(
        "ignore",
        message=r".*The component 'matcher' does not have any patterns defined.*",
        category=UserWarning,
        module="scispacy.abbreviation",
    )
except ImportError:
    spacy = None  # type: ignore
    SCISPACY_AVAILABLE = False

from A_core.A02_interfaces import BaseCandidateGenerator

# =============================================================================
# LIGHT NOISE FILTERING (High Recall - Let Validation Layer Judge)
# =============================================================================
# Philosophy: Generators should be EXHAUSTIVE. Only block OBVIOUS noise.
# Claude (D_validation) will handle borderline cases with context awareness.

# Obvious non-abbreviations: single letters, basic English function words
OBVIOUS_NOISE: set = {
    # Single letters (never valid abbreviations alone)
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    # Basic English function words (articles, prepositions, conjunctions)
    # NOTE: "or" and "us" removed - they can be valid abbreviations (Odds Ratio, United States)
    "an",
    "as",
    "at",
    "be",
    "by",
    "do",
    "go",
    "he",
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "no",
    "of",
    "on",
    "so",
    "to",
    "up",
    "we",
    "the",
    "and",
    "for",
    "but",
    "not",
    "are",
    "was",
    "were",
    "been",
    "have",
    "has",
    "had",
    "will",
    "would",
    "could",
    "should",
    "this",
    "that",
    "these",
    "those",
    "with",
    "from",
    "into",
    # Citation artifacts
    "et",
    "al",
    # Measurement units (not abbreviations, just units)
    "dl",
    "ml",
    "mg",
    "kg",
    "mm",
    "cm",
    "hz",
    "khz",
    "mhz",
    "mmhg",
    "kpa",
    "mol",
    "mmol",
    "umol",
    "nmol",
    # Full English words mistakenly in lexicons (NOT abbreviations)
    "investigator",
    "investigators",
    "sponsor",
    "protocol",
    "study",
    "patient",
    "patients",
    "subject",
    "subjects",
    "article",
    "articles",
    # Geographic (context-dependent, high FP rate in pharma docs)
    "nj",
    "ny",
    "ca",
    "tx",  # US states - usually location, not abbreviation
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

    __slots__ = ("sf", "lf", "pattern", "source", "lexicon_ids", "preserve_case")

    def __init__(
        self,
        sf: str,
        lf: str,
        pattern: re.Pattern,
        source: str,
        lexicon_ids: Optional[List[Dict[str, str]]] = None,
        preserve_case: bool = True,
    ):
        self.sf = sf
        self.lf = lf
        self.pattern = pattern
        self.source = source  # Lexicon file name for provenance
        self.lexicon_ids = lexicon_ids or []  # External IDs [{source, id}, ...]
        self.preserve_case = preserve_case  # If True, use matched text as SF


class RegexLexiconGenerator(BaseCandidateGenerator):
    """
    LEXICON MATCHER using:
    - Regex patterns for abbreviation lexicon (handles spacing/case variations)
    - FlashText for disease lexicon (fast exact matching)
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

        # Lexicon paths
        self.abbrev_lexicon_path = Path(
            self.config.get(
                "abbrev_lexicon_path",
                "/Users/frederictetard/Projects/ese/ouput_datasources/2025_08_abbreviation_general.json",
            )
        )
        self.disease_lexicon_path = Path(
            self.config.get(
                "disease_lexicon_path",
                "/Users/frederictetard/Projects/ese/ouput_datasources/2025_08_lexicon_disease.json",
            )
        )
        self.orphanet_lexicon_path = Path(
            self.config.get(
                "orphanet_lexicon_path",
                "/Users/frederictetard/Projects/ese/ouput_datasources/2025_08_orphanet_diseases.json",
            )
        )
        self.rare_disease_acronyms_path = Path(
            self.config.get(
                "rare_disease_acronyms_path",
                "/Users/frederictetard/Projects/ese/ouput_datasources/2025_08_rare_disease_acronyms.json",
            )
        )
        self.umls_abbrev_path = Path(
            self.config.get(
                "umls_abbrev_path",
                "/Users/frederictetard/Projects/ese/ouput_datasources/2025_08_umls_biological_abbreviations_v5.tsv",
            )
        )
        self.umls_clinical_path = Path(
            self.config.get(
                "umls_clinical_path",
                "/Users/frederictetard/Projects/ese/ouput_datasources/2025_08_umls_clinical_abbreviations_v5.tsv",
            )
        )
        self.anca_lexicon_path = Path(
            self.config.get(
                "anca_lexicon_path",
                "/Users/frederictetard/Projects/ese/ouput_datasources/disease_lexicon_anca.json",
            )
        )
        self.igan_lexicon_path = Path(
            self.config.get(
                "igan_lexicon_path",
                "/Users/frederictetard/Projects/ese/ouput_datasources/disease_lexicon_igan.json",
            )
        )
        self.pah_lexicon_path = Path(
            self.config.get(
                "pah_lexicon_path",
                "/Users/frederictetard/Projects/ese/ouput_datasources/disease_lexicon_pah.json",
            )
        )
        self.trial_acronyms_path = Path(
            self.config.get(
                "trial_acronyms_path",
                "/Users/frederictetard/Projects/ese/ouput_datasources/trial_acronyms_lexicon.json",
            )
        )
        self.pro_scales_path = Path(
            self.config.get(
                "pro_scales_path",
                "/Users/frederictetard/Projects/ese/ouput_datasources/pro_scales_lexicon.json",
            )
        )

        self.context_window = int(self.config.get("context_window", 300))

        # Abbreviation entries (regex-based)
        self.abbrev_entries: List[LexiconEntry] = []

        # Disease/entity entries (FlashText-based)
        self.entity_kp = KeywordProcessor(case_sensitive=False)
        self.entity_canonical: Dict[str, str] = {}  # matched_term -> canonical_name
        self.entity_source: Dict[str, str] = {}  # matched_term -> source file
        self.entity_ids: Dict[
            str, List[Dict[str, str]]
        ] = {}  # matched_term -> [{source, id}, ...]

        # Provenance
        self.pipeline_version = str(
            self.config.get("pipeline_version") or get_git_revision_hash()
        )
        self.run_id = str(self.config.get("run_id") or generate_run_id("ABBR"))
        self.doc_fingerprint_default = str(
            self.config.get("doc_fingerprint") or "unknown-doc-fingerprint"
        )

        # Lexicon loading stats (for summary output)
        self._lexicon_stats: List[tuple] = []

        # Load lexicons
        # Note: Disease/Orphanet lexicons contain disease NAMES, not abbreviations
        # They cause FPs by matching chromosome numbers (45, 46, 10p) as "abbreviations"
        self._load_abbrev_lexicon(self.abbrev_lexicon_path)
        # self._load_disease_lexicon(self.disease_lexicon_path)  # Disabled: names, not abbreviations
        # self._load_orphanet_lexicon(self.orphanet_lexicon_path)  # Disabled: names, not abbreviations
        self._load_rare_disease_acronyms(self.rare_disease_acronyms_path)
        self._load_umls_tsv(self.umls_abbrev_path)
        self._load_umls_tsv(self.umls_clinical_path)
        self._load_anca_lexicon(self.anca_lexicon_path)
        self._load_igan_lexicon(self.igan_lexicon_path)
        self._load_pah_lexicon(self.pah_lexicon_path)
        self._load_trial_acronyms(self.trial_acronyms_path)
        self._load_pro_scales(self.pro_scales_path)

        # Print compact summary
        self._print_lexicon_summary()

        # Initialize scispacy NER for biomedical entity recognition
        self.scispacy_nlp = None
        self.umls_linker = None
        if SCISPACY_AVAILABLE and spacy is not None:
            try:
                # Try lg model first (better for linking), fall back to sm
                try:
                    self.scispacy_nlp = spacy.load("en_core_sci_lg")
                    model_name = "en_core_sci_lg"
                except OSError:
                    self.scispacy_nlp = spacy.load("en_core_sci_sm")
                    model_name = "en_core_sci_sm"

                # Add abbreviation detector for Schwartz-Hearst pattern matching
                self.scispacy_nlp.add_pipe("abbreviation_detector")

                # Add UMLS EntityLinker for abbreviation resolution
                try:
                    self.scispacy_nlp.add_pipe(
                        "scispacy_linker",
                        config={"resolve_abbreviations": True, "linker_name": "umls"},
                    )
                    self.umls_linker = self.scispacy_nlp.get_pipe("scispacy_linker")
                    print(
                        f"Loaded scispacy {model_name} with abbreviation detector + UMLS linker"
                    )
                except Exception as e:
                    print(
                        f"Loaded scispacy {model_name} with abbreviation detector (no UMLS linker: {e})"
                    )
            except OSError as e:
                print(f"Warning: Could not load scispacy model: {e}")

    @property
    def generator_type(self) -> GeneratorType:
        return GeneratorType.LEXICON_MATCH

    def extract(self, doc_structure: DocumentGraph) -> List[Candidate]:
        doc = doc_structure  # Alias for readability
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

                    # Use matched text to preserve original case (e.g., IgA, MedDRA)
                    matched_text = text[start:end].strip()
                    if not matched_text:
                        continue  # Skip empty matches

                    sf_to_use = matched_text if entry.preserve_case else entry.sf

                    key = (sf_to_use.upper(), entry.lf.lower())
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

                    out.append(
                        self._make_candidate(
                            doc=doc,
                            block=block,
                            short_form=sf_to_use,  # Use matched text with original case
                            long_form=entry.lf,
                            start=start,
                            end=end,
                            text=text,
                            rule_version="abbrev_regex::v1",
                            lexicon_source=entry.source,
                            lexicon_ids=lex_ids,
                        )
                    )

            # 2) Entity matches (FlashText - diseases, orphanet terms)
            entity_hits = self.entity_kp.extract_keywords(text, span_info=True)
            for matched_term, start, end in entity_hits:
                # Skip blacklisted/invalid terms
                if not self._is_valid_match(matched_term):
                    continue

                canonical = self.entity_canonical.get(matched_term) or matched_term
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

                out.append(
                    self._make_candidate(
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
                    )
                )

            # 3) scispacy NER + abbreviation detection + UMLS linking
            if self.scispacy_nlp is not None:
                try:
                    spacy_doc = self.scispacy_nlp(text)

                    # Extract abbreviations found by Schwartz-Hearst detector
                    for abrv in spacy_doc._.abbreviations:
                        sf = abrv.text
                        lf = str(abrv._.long_form) if abrv._.long_form else None

                        if not sf or not lf:
                            continue
                        if not self._is_valid_match(sf):
                            continue

                        key = (sf.upper(), lf.lower())
                        if key in seen:
                            continue
                        seen.add(key)

                        out.append(
                            self._make_candidate(
                                doc=doc,
                                block=block,
                                short_form=sf,
                                long_form=lf,
                                start=abrv.start_char,
                                end=abrv.end_char,
                                text=text,
                                rule_version="scispacy_abbrev::v1",
                                lexicon_source="scispacy",
                                lexicon_ids=None,
                            )
                        )

                    # Extract NER entities with UMLS linking
                    for ent in spacy_doc.ents:
                        ent_text = ent.text.strip()

                        # Only consider short, uppercase-containing tokens as abbreviations
                        if len(ent_text) < 2 or len(ent_text) > 12:
                            continue
                        if " " in ent_text:  # Skip multi-word entities
                            continue
                        if not any(c.isupper() for c in ent_text):
                            continue
                        if not self._is_valid_match(ent_text):
                            continue

                        # Try UMLS linker first for expansion
                        lf_from_umls = None
                        umls_cui = None
                        if hasattr(ent._, "kb_ents") and ent._.kb_ents:
                            # kb_ents is [(CUI, score), ...] - take top match
                            top_match = ent._.kb_ents[0]
                            umls_cui = top_match[0]
                            # Get canonical name from linker's knowledge base
                            if self.umls_linker and hasattr(self.umls_linker, "kb"):
                                try:
                                    kb_entry = self.umls_linker.kb.cui_to_entity.get(
                                        umls_cui
                                    )
                                    if kb_entry:
                                        lf_from_umls = kb_entry.canonical_name
                                except Exception:
                                    pass

                        # Fall back to our lexicons if UMLS didn't provide expansion
                        lf_from_lexicon = lf_from_umls
                        if not lf_from_lexicon:
                            lf_from_lexicon = self.entity_canonical.get(
                                ent_text
                            ) or self.entity_canonical.get(ent_text.upper())
                        if not lf_from_lexicon:
                            # Try to find in abbrev_entries
                            for entry in self.abbrev_entries:
                                if entry.sf.upper() == ent_text.upper():
                                    lf_from_lexicon = entry.lf
                                    break

                        if not lf_from_lexicon:
                            continue  # Only report if we have a known expansion
                        assert lf_from_lexicon is not None  # Type narrowing for pyright

                        key = (ent_text.upper(), lf_from_lexicon.lower())
                        if key in seen:
                            continue
                        seen.add(key)

                        # Build lexicon IDs from UMLS CUI if available
                        lex_ids = None
                        if umls_cui:
                            lex_ids = [LexiconIdentifier(source="UMLS", id=umls_cui)]

                        source = "scispacy+umls" if lf_from_umls else "scispacy+lexicon"
                        out.append(
                            self._make_candidate(
                                doc=doc,
                                block=block,
                                short_form=ent_text,
                                long_form=lf_from_lexicon,
                                start=ent.start_char,
                                end=ent.end_char,
                                text=text,
                                rule_version="scispacy_ner::v1",
                                lexicon_source=source,
                                lexicon_ids=lex_ids,
                            )
                        )
                except Exception:
                    # Don't fail entire extraction if scispacy has issues
                    pass

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

    def _print_lexicon_summary(self) -> None:
        """Print compact summary of loaded lexicons."""
        if not self._lexicon_stats:
            return

        total = sum(count for _, count in self._lexicon_stats)
        print(f"Lexicons loaded: {len(self._lexicon_stats)} files, {total:,} terms")
        for name, count in self._lexicon_stats:
            print(f"  {name:<30} {count:>7,}")

    def _load_abbrev_lexicon(self, path: Path) -> None:
        if not path.exists():
            return

        data = json.loads(path.read_text(encoding="utf-8"))
        loaded = 0

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

                self.abbrev_entries.append(
                    LexiconEntry(sf=sf, lf=lf, pattern=pattern, source=path.name)
                )
                loaded += 1
            except re.error:
                pass

        self._lexicon_stats.append(("Abbreviations", loaded))

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
                for s in sources_list
                if isinstance(s, dict) and s.get("id")
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
            return

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

        self._lexicon_stats.append(("Rare disease acronyms", loaded))

    def _load_umls_tsv(self, path: Path) -> None:
        if not path.exists():
            return

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

        # Extract a short name from filename for display
        name = "UMLS biological" if "biological" in source else "UMLS clinical"
        self._lexicon_stats.append((name, loaded))

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
            return

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

        self._lexicon_stats.append(("ANCA disease", loaded))

    def _load_igan_lexicon(self, path: Path) -> None:
        if not path.exists():
            return

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

        self._lexicon_stats.append(("IgAN disease", loaded))

    def _load_pah_lexicon(self, path: Path) -> None:
        if not path.exists():
            return

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

        self._lexicon_stats.append(("PAH disease", loaded))

    def _load_trial_acronyms(self, path: Path) -> None:
        """Load clinical trial acronyms lexicon (RADAR, APPEAR-C3G, MAINRITSAN, etc.)."""
        if not path.exists():
            return

        data = json.loads(path.read_text(encoding="utf-8"))
        loaded = 0

        for acronym, entry in data.items():
            if not acronym or not isinstance(entry, dict):
                continue

            # Get expansion and regex pattern
            lf = entry.get("canonical_expansion")
            regex_str = entry.get("regex")

            if not lf or not regex_str:
                continue

            # Skip very long expansions (full trial titles)
            # We want abbreviations, not full protocol titles
            if len(lf) > 150:
                lf = f"{acronym} (clinical trial)"

            try:
                case_insensitive = bool(entry.get("case_insensitive", False))
                flags = re.IGNORECASE if case_insensitive else 0
                pattern = re.compile(regex_str, flags)

                # Build lexicon IDs from NCT ID
                lexicon_ids = []
                nct_id = entry.get("nct_id")
                if nct_id:
                    lexicon_ids.append({"source": "ClinicalTrials.gov", "id": nct_id})

                self.abbrev_entries.append(
                    LexiconEntry(
                        sf=acronym,
                        lf=lf,
                        pattern=pattern,
                        source=path.name,
                        lexicon_ids=lexicon_ids,
                        preserve_case=True,  # Preserve case for trial names
                    )
                )
                loaded += 1
            except re.error:
                pass

        self._lexicon_stats.append(("Trial acronyms", loaded))

    def _load_pro_scales(self, path: Path) -> None:
        """Load PRO scales lexicon (SF-36, PHQ-9, EORTC-QLQ-C30, etc.)."""
        if not path.exists():
            return

        data = json.loads(path.read_text(encoding="utf-8"))
        loaded = 0

        for scale_name, entry in data.items():
            if not scale_name or not isinstance(entry, dict):
                continue

            # Get expansion and regex pattern
            lf = entry.get("canonical_expansion")
            regex_str = entry.get("regex")

            if not lf or not regex_str:
                continue

            try:
                case_insensitive = bool(entry.get("case_insensitive", False))
                flags = re.IGNORECASE if case_insensitive else 0
                pattern = re.compile(regex_str, flags)

                # Build lexicon IDs from example NCT IDs (use first one)
                lexicon_ids = []
                example_ncts = entry.get("example_nct_ids", [])
                if example_ncts and len(example_ncts) > 0:
                    lexicon_ids.append(
                        {"source": "ClinicalTrials.gov", "id": example_ncts[0]}
                    )

                self.abbrev_entries.append(
                    LexiconEntry(
                        sf=scale_name,
                        lf=lf,
                        pattern=pattern,
                        source=path.name,
                        lexicon_ids=lexicon_ids,
                        preserve_case=True,  # Preserve case for PRO scale names
                    )
                )
                loaded += 1
            except re.error:
                pass

        self._lexicon_stats.append(("PRO scales", loaded))

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

        # Block pure numbers (not abbreviations)
        if term.isdigit():
            return False

        # Block short alphanumeric codes starting with digit (e.g., 1A, 2B, 45)
        if len(term) <= 3 and term[0].isdigit():
            return False

        return True

# corpus_metadata/C_generators/C04b_lexicon_loaders.py
"""
Lexicon loading methods for FlashText-based extraction.

Contains all lexicon loader methods used by RegexLexiconGenerator:
- Abbreviation lexicons (general, clinical research, UMLS, Meta-Inventory)
- Disease lexicons (ANCA, IgAN, PAH, MONDO, rare disease acronyms)
- Drug lexicons (ChEMBL)
- Trial and PRO scale lexicons
"""

from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from flashtext import KeywordProcessor

from C_generators.C04a_noise_filters import (
    BAD_LONG_FORMS,
    LexiconEntry,
    WRONG_EXPANSION_BLACKLIST,
)


class LexiconLoaderMixin:
    """
    Mixin class providing lexicon loading methods.

    Designed to be mixed into RegexLexiconGenerator to provide
    all _load_* methods for various lexicon types.
    """

    # These attributes are expected to be set by the main class
    abbrev_entries: List[LexiconEntry]
    entity_kp: "KeywordProcessor"
    entity_canonical: Dict[str, str]
    entity_source: Dict[str, str]
    entity_ids: Dict[str, List[Dict[str, str]]]
    _lexicon_stats: List[tuple]

    def _print_lexicon_summary(self) -> None:
        """Print compact summary of loaded lexicons grouped by category."""
        if not self._lexicon_stats:
            return

        # Categorize lexicons with explicit ordering: Abbreviation, Drug, Disease, Other
        categories: list[tuple[str, list[tuple[str, int, str]]]] = [
            ("Abbreviation", []),
            ("Drug", []),
            ("Disease", []),
            ("Other", []),
        ]
        cat_dict = {name: items for name, items in categories}

        # Map lexicon names to categories
        category_map = {
            # Abbreviation
            "Abbreviations": "Abbreviation",
            "Clinical research": "Abbreviation",
            "UMLS biological": "Abbreviation",
            "UMLS clinical": "Abbreviation",
            "Meta-Inventory": "Abbreviation",
            # Drug
            "ChEMBL drugs": "Drug",
            # Disease
            "Rare disease acronyms": "Disease",
            "ANCA disease": "Disease",
            "IgAN disease": "Disease",
            "PAH disease": "Disease",
            "MONDO diseases": "Disease",
            # Other
            "Trial acronyms": "Other",
            "PRO scales": "Other",
            "Pharma companies": "Other",
        }

        for name, count, filename in self._lexicon_stats:
            cat = category_map.get(name, "Abbreviation")
            if cat in cat_dict:
                cat_dict[cat].append((name, count, filename))

        total = sum(count for _, count, _ in self._lexicon_stats)
        file_count = len([s for s in self._lexicon_stats if s[1] > 0])
        logger.info("Lexicons loaded: %d files, %d terms", file_count, total)

        for cat_name, items in categories:
            if not items:
                continue
            cat_total = sum(count for _, count, _ in items)
            logger.info("  %s (%d terms)", cat_name, cat_total)
            for name, count, filename in items:
                logger.debug("    * %-26s %8d  %s", name, count, filename)

    def _load_abbrev_lexicon(self, path: Path, label: str = "Abbreviations") -> None:
        if not path.exists():
            return

        content = path.read_text(encoding="utf-8").strip()
        if not content:
            return

        data = json.loads(content)
        if not data:  # Skip empty files
            return

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

        self._lexicon_stats.append((label, loaded, path.name))

    def _load_disease_lexicon(self, path: Path) -> None:
        if not path.exists():
            logger.warning("Disease lexicon not found: %s", path)
            return

        logger.debug("Loading disease lexicon: %s", path)
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

        logger.debug("Loaded %d disease terms from %s", loaded, path.name)

    def _load_orphanet_lexicon(self, path: Path) -> None:
        if not path.exists():
            logger.warning("Orphanet lexicon not found: %s", path)
            return

        logger.debug("Loading Orphanet lexicon: %s", path)
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

        logger.debug("Loaded %d Orphanet terms from %s", loaded, path.name)

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

        self._lexicon_stats.append(("Rare disease acronyms", loaded, path.name))

    def _load_umls_tsv(self, path: Path) -> None:
        if not path.exists():
            return

        source = path.name
        loaded = 0
        skipped_wrong_expansion = 0

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                abbrev = (row.get("Abbreviation") or "").strip()
                expansion = (row.get("Expansion") or "").strip()
                top_source = (row.get("TopSource") or "").strip()

                if not abbrev or len(abbrev) < 2 or not expansion:
                    continue

                # Filter out known wrong SF -> LF pairs
                abbrev_lower = abbrev.lower()
                expansion_lower = expansion.lower()
                if (abbrev_lower, expansion_lower) in WRONG_EXPANSION_BLACKLIST:
                    skipped_wrong_expansion += 1
                    continue

                # Filter out known bad long forms (regardless of short form)
                if expansion_lower in BAD_LONG_FORMS:
                    skipped_wrong_expansion += 1
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
        self._lexicon_stats.append((name, loaded, path.name))

    def _extract_identifiers(self, identifiers: Dict) -> List[Dict[str, str]]:
        """Extract lexicon IDs from an identifiers dict."""
        lexicon_ids: list[dict[str, str]] = []
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

        self._lexicon_stats.append(("ANCA disease", loaded, path.name))

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

        self._lexicon_stats.append(("IgAN disease", loaded, path.name))

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

        self._lexicon_stats.append(("PAH disease", loaded, path.name))

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

        self._lexicon_stats.append(("Trial acronyms", loaded, path.name))

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

        self._lexicon_stats.append(("PRO scales", loaded, path.name))

    def _load_pharma_companies(self, path: Path) -> None:
        """Load pharma companies lexicon (Roche, Novartis, Pfizer, etc.)."""
        if not path.exists():
            return

        data = json.loads(path.read_text(encoding="utf-8"))
        source = path.name
        loaded = 0

        for company_key, entry in data.items():
            if not company_key or not isinstance(entry, dict):
                continue

            canonical = (entry.get("canonical_name") or "").strip()
            full_name = (entry.get("full_name") or "").strip()
            variants = entry.get("variants") or []

            if not canonical:
                continue

            # Use full_name as the long form if available, otherwise canonical
            lf = full_name if full_name else canonical

            # Register all variants (including canonical name)
            all_terms = set(variants)
            all_terms.add(canonical)
            if full_name:
                all_terms.add(full_name)

            for term in all_terms:
                term = (term or "").strip()
                if not term or len(term) < 2:
                    continue

                self.entity_kp.add_keyword(term, term)
                self.entity_canonical[term] = lf
                self.entity_source[term] = source
                self.entity_ids[term] = []  # No external IDs for pharma companies
                loaded += 1

        self._lexicon_stats.append(("Pharma companies", loaded, path.name))

    # =========================================================================
    # NEW LEXICONS: Meta-Inventory, MONDO, ChEMBL
    # =========================================================================

    def _load_meta_inventory(self, path: Path) -> None:
        """
        Load Meta-Inventory clinical abbreviations (104K+ abbreviations).

        Source: https://github.com/lisavirginia/clinical-abbreviations
        Paper: https://www.nature.com/articles/s41597-021-00929-4

        Format: {SF: {canonical_expansion, regex, expansions: [...]}}
        """
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

            # Skip very short abbreviations
            if len(sf) < 2:
                continue

            try:
                case_insensitive = bool(entry.get("case_insensitive", True))
                flags = re.IGNORECASE if case_insensitive else 0
                pattern = re.compile(regex_str, flags)

                self.abbrev_entries.append(
                    LexiconEntry(
                        sf=sf,
                        lf=lf,
                        pattern=pattern,
                        source="meta-inventory"
                    )
                )
                loaded += 1
            except re.error:
                pass

        self._lexicon_stats.append(("Meta-Inventory", loaded, path.name))

    def _load_mondo_lexicon(self, path: Path) -> None:
        """
        Load MONDO disease ontology.

        Source: https://mondo.monarchinitiative.org/
        Provides unified disease mappings with precise semantics.

        Format: [{label, sources: [{source, id}], synonyms: [...]}]
        """
        if not path.exists():
            return

        data = json.loads(path.read_text(encoding="utf-8"))
        source = path.name
        loaded = 0

        for entry in data:
            if not isinstance(entry, dict):
                continue

            label = (entry.get("label") or "").strip()
            if not label or len(label) < 3:
                continue

            # Extract MONDO and cross-reference IDs
            sources_list = entry.get("sources", [])
            lexicon_ids = [
                {"source": s.get("source", ""), "id": s.get("id", "")}
                for s in sources_list
                if isinstance(s, dict) and s.get("id")
            ]

            # Use canonical label if this is a synonym entry
            canonical = entry.get("canonical", label)

            self.entity_kp.add_keyword(label, label)
            self.entity_canonical[label] = canonical
            self.entity_source[label] = source
            self.entity_ids[label] = lexicon_ids
            loaded += 1

        self._lexicon_stats.append(("MONDO diseases", loaded, path.name))

    def _load_chembl_lexicon(self, path: Path) -> None:
        """
        Load ChEMBL approved drugs.

        Source: https://www.ebi.ac.uk/chembl/
        Open data drug database with bioactivity information.

        Format: [{label, chembl_id, max_phase, synonyms: [...]}]
        """
        if not path.exists():
            return

        data = json.loads(path.read_text(encoding="utf-8"))
        source = path.name
        loaded = 0

        # Handle placeholder format
        if isinstance(data, dict) and "drugs" in data:
            drugs = data.get("drugs", [])
        elif isinstance(data, list):
            drugs = data
        else:
            return

        for entry in drugs:
            if not isinstance(entry, dict):
                continue

            label = (entry.get("label") or "").strip()
            if not label or len(label) < 2:
                continue

            chembl_id = entry.get("chembl_id", "")
            synonyms = entry.get("synonyms", [])

            # Build identifier list
            lexicon_ids = []
            if chembl_id:
                lexicon_ids.append({"source": "ChEMBL", "id": chembl_id})

            # Add main drug name
            self.entity_kp.add_keyword(label, label)
            self.entity_canonical[label] = label
            self.entity_source[label] = source
            self.entity_ids[label] = lexicon_ids
            loaded += 1

            # Add synonyms
            for syn in synonyms[:5]:  # Limit synonyms per drug
                syn = (syn or "").strip()
                if syn and len(syn) >= 2 and syn != label:
                    self.entity_kp.add_keyword(syn, syn)
                    self.entity_canonical[syn] = label
                    self.entity_source[syn] = source
                    self.entity_ids[syn] = lexicon_ids
                    loaded += 1

        self._lexicon_stats.append(("ChEMBL drugs", loaded, path.name))


__all__ = ["LexiconLoaderMixin"]

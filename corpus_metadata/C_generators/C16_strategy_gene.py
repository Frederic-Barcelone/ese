"""
Gene and protein entity detection for rare disease documents.

This module detects gene and protein names in clinical documents using a
multi-layered approach combining rare disease gene databases, official
nomenclature, pattern matching, and biomedical NER with false positive filtering.

Key Components:
    - GeneDetector: Main detector combining multiple detection strategies
    - Lexicon layers (in priority order):
        1. Orphadata genes (rare disease-associated, highest priority)
        2. HGNC aliases (official gene nomenclature)
        3. Gene symbol patterns with context validation
        4. scispacy NER (GENE semantic type, fallback)
    - GeneFalsePositiveFilter: Filters ambiguous symbols (C34)

Example:
    >>> from C_generators.C16_strategy_gene import GeneDetector
    >>> detector = GeneDetector(config={"lexicon_base_path": "lexicons/"})
    >>> candidates = detector.detect(doc_graph, "doc_123", "fingerprint")
    >>> for c in candidates:
    ...     print(f"{c.symbol} ({c.hgnc_id}): {c.disease_associations}")
    CFH (HGNC:4883): ['atypical hemolytic uremic syndrome']

Dependencies:
    - A_core.A01_domain_models: Coordinate
    - A_core.A03_provenance: Provenance tracking utilities
    - A_core.A19_gene_models: GeneCandidate, GeneFieldType, GeneGeneratorType
    - B_parsing.B01_pdf_to_docgraph: DocumentGraph
    - B_parsing.B06_confidence: Confidence scoring
    - C_generators.C34_gene_fp_filter: False positive filtering
    - flashtext: KeywordProcessor for fast lexicon matching
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from flashtext import KeywordProcessor

from A_core.A01_domain_models import Coordinate

logger = logging.getLogger(__name__)
from A_core.A03_provenance import generate_run_id, get_git_revision_hash
from A_core.A19_gene_models import (
    GeneCandidate,
    GeneDiseaseLinkage,
    GeneFieldType,
    GeneGeneratorType,
    GeneIdentifier,
    GeneProvenanceMetadata,
)
from B_parsing.B01_pdf_to_docgraph import DocumentGraph
from B_parsing.B06_confidence import ConfidenceCalculator

from .C34_gene_fp_filter import GeneFalsePositiveFilter

# Optional scispacy import
try:
    import spacy
    from scispacy.linking import EntityLinker  # noqa: F401

    SCISPACY_AVAILABLE = True
except ImportError:
    spacy = None
    SCISPACY_AVAILABLE = False


# -------------------------
# Gene Detector
# -------------------------


class GeneDetector:
    """
    Multi-layered gene mention detection for rare diseases.

    Layers (in priority order):
    1. Orphadata genes (rare disease-associated)
    2. HGNC aliases
    3. Gene symbol patterns with context validation
    4. scispacy NER (fallback)
    """

    # UMLS semantic types for genes
    GENE_SEMANTIC_TYPES = {
        "T028",  # Gene or Genome
        "T116",  # Amino Acid, Peptide, or Protein
        "T126",  # Enzyme
    }

    # Gene symbol pattern: 1-6 uppercase letters/numbers, starting with letter
    GENE_PATTERN = re.compile(r"\b([A-Z][A-Z0-9]{1,6})\b")

    # Terms to skip when loading lexicons
    LEXICON_LOAD_BLACKLIST: Set[str] = {
        # Statistical terms
        "or", "hr", "ci", "sd", "se", "rr", "mr", "md", "ns", "na", "nd",
        # Units
        "mm", "cm", "kg", "mg", "ml", "dl", "ng", "pg", "hz", "kd", "da",
        # Clinical
        "iv", "po", "im", "sc", "bid", "tid", "qd", "prn", "er", "icu", "ed",
        # Countries
        "us", "uk", "eu", "ca", "au", "de", "fr", "jp", "cn",
        # Credentials
        "md", "phd", "mph", "do", "rn", "ms", "ma", "mba",
        # Common abbreviations that conflict with gene symbols
        "kl", "li", "gi", "hf", "nt", "sg", "fa", "wd", "ac",
        # Medical abbreviations commonly misidentified as genes
        "ent", "mpo", "pr3",
        # Common English words that are gene aliases (filter at load time)
        "type", "face", "fritz", "act", "alpha", "beta", "gamma", "delta",
        "if", "am", "minor", "impacts", "fate", "ash", "storm",
        "of", "on", "an", "as", "can", "for", "go", "not",  # Definite articles/prepositions
        "ii", "iii", "iv", "vi",  # Roman numerals
        "nude", "pi", "d1",  # Common words/symbols mismatched as genes
        "emt",  # Epithelial-mesenchymal transition (process, not gene)
        "sma",  # Spinal muscular atrophy (disease, not gene SMN1)
        "lca",  # Leber congenital amaurosis (disease, not gene)
        "b1", "b2", "b3",  # Vitamin B designations
        "cac",  # Coronary artery calcification
        "mcp",  # Monocyte chemoattractant protein / medical abbreviation
        "th",   # T-helper cells
        "tf",   # Transcription factor
        "tap",  # Various medical uses
        "rds",  # Respiratory distress syndrome
        "hk2",  # Hexokinase 2 (different gene, not alias for KCNA5)
        "gata",  # GATA transcription factor family
        "iap",  # Inhibitor of apoptosis proteins (ambiguous)
        # Amino acid abbreviations confused with genes
        "arg", "his", "pro", "val", "met",
        # Medical abbreviations commonly confused with genes
        "cox",  # Cyclooxygenase (COX-1, COX-2) — not gene COX8A
        "aa",   # Amino acid — not gene TEAD1
        "inf",  # Inflammation/infinity — not gene CBLIF
        "lqt",  # Long QT syndrome (disease) — not gene KCNQ1
        # Journal names and other common terms
        "acta",  # Journal name (Acta Otorhinolaryngol, etc.)
        # PDF line-break fragments that match gene symbols
        "gan",  # Fragment from "Erdo-gan" etc., matches GAN gene
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.run_id = str(self.config.get("run_id") or generate_run_id("GENE"))
        self.pipeline_version = (
            self.config.get("pipeline_version") or get_git_revision_hash()
        )
        self.doc_fingerprint_default = (
            self.config.get("doc_fingerprint") or "unknown-doc-fingerprint"
        )

        # Context window for evidence extraction
        self.context_window = int(self.config.get("context_window", 600))

        # Confidence calculator
        self.confidence_calculator = ConfidenceCalculator()

        # Lexicon base path
        self.lexicon_base_path = Path(
            self.config.get("lexicon_base_path", "ouput_datasources")
        )

        # Initialize FlashText processors
        self.orphadata_processor: Optional[KeywordProcessor] = None
        self.alias_processor: Optional[KeywordProcessor] = None
        self.name_processor: Optional[KeywordProcessor] = None

        # Gene metadata dictionaries
        self.orphadata_genes: Dict[str, Dict] = {}
        self.alias_genes: Dict[str, Dict] = {}
        self.name_genes: Dict[str, Dict] = {}

        # Lexicon loading stats
        self._lexicon_stats: List[Tuple[str, int, str]] = []

        # Load lexicons
        self._load_lexicons()

        # False positive filter with disease lexicon disambiguation
        self.fp_filter = GeneFalsePositiveFilter(lexicon_base_path=self.lexicon_base_path)

        # scispacy NER model
        self.nlp = None
        if SCISPACY_AVAILABLE:
            self._init_scispacy()

    def _load_lexicons(self) -> None:
        """Load gene lexicons."""
        self._load_orphadata_lexicon()

    def _load_orphadata_lexicon(self) -> None:
        """Load Orphadata gene lexicon (rare disease genes + HGNC aliases)."""
        path = self.lexicon_base_path / "2025_08_orphadata_genes.json"
        if not path.exists():
            logger.warning("Orphadata gene lexicon not found: %s", path)
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            orphadata_proc = KeywordProcessor(case_sensitive=False)
            alias_proc = KeywordProcessor(case_sensitive=False)

            self.orphadata_processor = orphadata_proc
            self.alias_processor = alias_proc

            primary_count = 0
            alias_count = 0
            skipped = 0

            for entry in data:
                term = entry.get("term", "").strip().strip('"')
                if not term or len(term) < 2:
                    continue

                term_key = term.lower()

                # Skip blacklisted terms
                if term_key in self.LEXICON_LOAD_BLACKLIST:
                    skipped += 1
                    continue

                source = entry.get("source", "")

                if source == "orphadata_hgnc":
                    # Primary gene entry
                    gene_info = {
                        "symbol": entry.get("hgnc_symbol", term),
                        "full_name": entry.get("full_name"),
                        "hgnc_id": entry.get("hgnc_id"),
                        "entrez_id": entry.get("entrez_id"),
                        "ensembl_id": entry.get("ensembl_id"),
                        "omim_id": entry.get("omim_id"),
                        "uniprot_id": entry.get("uniprot_id"),
                        "locus_type": entry.get("locus_type"),
                        "associated_diseases": entry.get("associated_diseases", []),
                        "source": "orphadata",
                    }
                    self.orphadata_genes[term_key] = gene_info
                    orphadata_proc.add_keyword(term, term_key)
                    primary_count += 1

                    # Auto-add hyphenated form for symbols like IL6→IL-6
                    hyphen_form = self._make_hyphenated(term)
                    if hyphen_form:
                        hf_key = hyphen_form.lower()
                        if hf_key not in self.alias_genes and hf_key != term_key:
                            self.alias_genes[hf_key] = {
                                "symbol": gene_info["symbol"],
                                "alias_term": hyphen_form,
                                "full_name": gene_info.get("full_name"),
                                "hgnc_id": gene_info.get("hgnc_id"),
                                "entrez_id": gene_info.get("entrez_id"),
                                "ensembl_id": gene_info.get("ensembl_id"),
                                "omim_id": gene_info.get("omim_id"),
                                "uniprot_id": gene_info.get("uniprot_id"),
                                "locus_type": gene_info.get("locus_type"),
                                "source": "hgnc_alias",
                            }
                            alias_proc.add_keyword(hyphen_form, hf_key)
                            alias_count += 1

                elif source == "hgnc_alias":
                    # Alias entry
                    canonical = entry.get("is_alias_of", entry.get("hgnc_symbol", term))
                    self.alias_genes[term_key] = {
                        "symbol": canonical,
                        "alias_term": term,
                        "full_name": entry.get("full_name"),
                        "hgnc_id": entry.get("hgnc_id"),
                        "entrez_id": entry.get("entrez_id"),
                        "ensembl_id": entry.get("ensembl_id"),
                        "omim_id": entry.get("omim_id"),
                        "uniprot_id": entry.get("uniprot_id"),
                        "locus_type": entry.get("locus_type"),
                        "source": "hgnc_alias",
                    }
                    alias_proc.add_keyword(term, term_key)
                    alias_count += 1

            # Add common protein name aliases not in HGNC data
            EXTRA_ALIASES = {
                "E-cadherin": "CDH1",
                "N-cadherin": "CDH2",
                "P-cadherin": "CDH3",
                "TGF-beta1": "TGFB1",
                "TGF-beta2": "TGFB2",
                "TGF-beta 1": "TGFB1",
                "TGF-beta 2": "TGFB2",
                "interferon-gamma": "IFNG",
                "interferon gamma": "IFNG",
                "IFN-gamma": "IFNG",
                "alpha-SMA": "ACTA2",
                "alpha-smooth muscle actin": "ACTA2",
            }
            for alias_term, symbol in EXTRA_ALIASES.items():
                alias_key = alias_term.lower()
                if alias_key not in self.alias_genes:
                    gene_info_ref = self.orphadata_genes.get(symbol.lower(), {})
                    if gene_info_ref:
                        self.alias_genes[alias_key] = {
                            "symbol": symbol,
                            "alias_term": alias_term,
                            "full_name": gene_info_ref.get("full_name"),
                            "hgnc_id": gene_info_ref.get("hgnc_id"),
                            "entrez_id": gene_info_ref.get("entrez_id"),
                            "ensembl_id": gene_info_ref.get("ensembl_id"),
                            "omim_id": gene_info_ref.get("omim_id"),
                            "uniprot_id": gene_info_ref.get("uniprot_id"),
                            "locus_type": gene_info_ref.get("locus_type"),
                            "source": "hgnc_alias",
                        }
                        alias_proc.add_keyword(alias_term, alias_key)
                        alias_count += 1

            if skipped > 0:
                logger.debug("Skipped %d blacklisted gene terms", skipped)

            # Build gene full name processor (Layer 3)
            name_proc = KeywordProcessor(case_sensitive=False)
            name_count = 0
            for entry in data:
                if entry.get("source") != "orphadata_hgnc":
                    continue
                full_name = entry.get("full_name", "").strip()
                symbol = entry.get("hgnc_symbol", "")
                if not full_name or not symbol:
                    continue
                # Skip if name is same as symbol (already matched by primary)
                if full_name.upper() == symbol.upper():
                    continue
                # Skip very short names (< 6 chars) to avoid FPs
                if len(full_name) < 6:
                    continue
                name_key = full_name.lower()
                if name_key not in self.name_genes:
                    gene_info_ref = self.orphadata_genes.get(symbol.lower(), {})
                    self.name_genes[name_key] = {
                        "symbol": symbol,
                        "full_name": full_name,
                        "hgnc_id": entry.get("hgnc_id"),
                        "entrez_id": entry.get("entrez_id"),
                        "ensembl_id": entry.get("ensembl_id"),
                        "omim_id": entry.get("omim_id"),
                        "uniprot_id": entry.get("uniprot_id"),
                        "locus_type": entry.get("locus_type"),
                        "associated_diseases": gene_info_ref.get("associated_diseases", []),
                        "source": "gene_full_name",
                    }
                    name_proc.add_keyword(full_name, name_key)
                    name_count += 1
            self.name_processor = name_proc

            self._lexicon_stats.append(
                ("Orphadata genes", primary_count, "2025_08_orphadata_genes.json")
            )
            self._lexicon_stats.append(
                ("HGNC aliases", alias_count, "2025_08_orphadata_genes.json")
            )
            self._lexicon_stats.append(
                ("Gene full names", name_count, "2025_08_orphadata_genes.json")
            )

        except Exception as e:
            logger.warning("Failed to load Orphadata gene lexicon: %s", e)

    def _init_scispacy(self) -> None:
        """Initialize scispacy NER model."""
        if not SCISPACY_AVAILABLE or spacy is None:
            return

        try:
            try:
                self.nlp = spacy.load("en_core_sci_lg")
            except OSError:
                self.nlp = spacy.load("en_core_sci_sm")

            assert self.nlp is not None  # Type narrowing for mypy
            if "scispacy_linker" not in self.nlp.pipe_names:
                self.nlp.add_pipe(
                    "scispacy_linker",
                    config={
                        "resolve_abbreviations": True,
                        "linker_name": "umls",
                        "threshold": 0.7,
                    },
                )
            self._lexicon_stats.append(("scispacy NER", 1, "en_core_sci_lg"))

        except Exception as e:
            logger.warning("Failed to initialize scispacy for genes: %s", e)
            self.nlp = None

    def _print_lexicon_summary(self) -> None:
        """Print compact summary of loaded gene lexicons."""
        if not self._lexicon_stats:
            return

        total = sum(count for _, count, _ in self._lexicon_stats if count > 1)
        logger.info("Gene lexicons: %d sources, %d entries", len(self._lexicon_stats), total)

        for name, count, filename in self._lexicon_stats:
            if count > 1:
                logger.debug("    • %-26s %8d  %s", name, count, filename)
            else:
                logger.debug("    • %-26s %8s  %s", name, "enabled", filename)

    def detect(self, doc_graph: DocumentGraph) -> List[GeneCandidate]:
        """
        Detect gene mentions in document.

        Returns list of GeneCandidate objects.
        """
        candidates: List[GeneCandidate] = []
        doc_fingerprint = getattr(
            doc_graph, "fingerprint", self.doc_fingerprint_default
        )

        # Get full text for detection
        full_text = "\n\n".join(
            block.text for block in doc_graph.iter_linear_blocks(skip_header_footer=True)
            if block.text
        )

        # Layer 1: Orphadata genes (rare disease-associated)
        if self.orphadata_processor:
            candidates.extend(
                self._detect_with_lexicon(
                    full_text,
                    doc_graph,
                    doc_fingerprint,
                    self.orphadata_processor,
                    self.orphadata_genes,
                    GeneGeneratorType.LEXICON_ORPHADATA,
                    "2025_08_orphadata_genes.json",
                    is_primary=True,
                )
            )

        # Layer 2: HGNC aliases
        if self.alias_processor:
            candidates.extend(
                self._detect_with_lexicon(
                    full_text,
                    doc_graph,
                    doc_fingerprint,
                    self.alias_processor,
                    self.alias_genes,
                    GeneGeneratorType.LEXICON_HGNC_ALIAS,
                    "2025_08_orphadata_genes.json",
                    is_primary=False,
                )
            )

        # Layer 3: Gene full name matching (e.g. "interleukin 6" → IL6)
        if self.name_processor:
            candidates.extend(
                self._detect_with_lexicon(
                    full_text,
                    doc_graph,
                    doc_fingerprint,
                    self.name_processor,
                    self.name_genes,
                    GeneGeneratorType.LEXICON_HGNC_ALIAS,
                    "2025_08_orphadata_genes.json",
                    is_primary=False,
                )
            )

        # Layer 4: Gene symbol patterns with context validation
        # DISABLED: Causes too many false positives - the lexicon coverage is sufficient
        # for rare disease genes. Uncomment if you need pattern-based detection.
        # candidates.extend(
        #     self._detect_gene_patterns(full_text, doc_graph, doc_fingerprint)
        # )

        # Layer 4: scispacy NER fallback
        # DISABLED: Causes too many false positives without proper validation.
        # Uncomment if you need NER-based detection.
        # if self.nlp:
        #     candidates.extend(
        #         self._detect_with_ner(full_text, doc_graph, doc_fingerprint)
        #     )

        # Deduplicate
        candidates = self._deduplicate(candidates)

        return candidates

    def _detect_with_lexicon(
        self,
        text: str,
        doc_graph: DocumentGraph,
        doc_fingerprint: str,
        processor: KeywordProcessor,
        gene_dict: Dict[str, Dict],
        generator_type: GeneGeneratorType,
        lexicon_source: str,
        is_primary: bool = True,
    ) -> List[GeneCandidate]:
        """Detect genes using FlashText lexicon matching."""
        candidates = []

        matches = processor.extract_keywords(text, span_info=True)

        for keyword, start, end in matches:
            gene_info = gene_dict.get(keyword, {})
            if not gene_info:
                continue

            matched_text = text[start:end]
            context = self._extract_context(text, start, end)

            # Apply false positive filter
            # Aliases need stricter filtering than primary genes
            is_fp, reason = self.fp_filter.is_false_positive(
                matched_text, context, generator_type,
                is_from_lexicon=True, is_alias=(not is_primary)
            )
            if is_fp:
                continue

            # Build identifiers
            identifiers = self._build_identifiers(gene_info)

            # Build disease linkages
            disease_linkages = []
            for disease in gene_info.get("associated_diseases", []):
                disease_linkages.append(
                    GeneDiseaseLinkage(
                        orphacode=str(disease.get("orphacode", "")),
                        disease_name=disease.get("name", ""),
                        association_type=disease.get("association_type"),
                        association_status=disease.get("association_status"),
                    )
                )

            # Determine confidence
            if generator_type == GeneGeneratorType.LEXICON_ORPHADATA:
                confidence = 0.85
            else:
                confidence = 0.80

            # Build provenance
            provenance = GeneProvenanceMetadata(
                pipeline_version=self.pipeline_version,
                run_id=self.run_id,
                doc_fingerprint=doc_fingerprint,
                generator_name=generator_type,
                lexicon_source=lexicon_source,
                lexicon_ids=identifiers,
            )

            candidate = GeneCandidate(
                doc_id=doc_graph.doc_id,
                matched_text=matched_text,
                hgnc_symbol=gene_info.get("symbol", matched_text),
                full_name=gene_info.get("full_name"),
                is_alias=not is_primary,
                alias_of=gene_info.get("symbol") if not is_primary else None,
                field_type=GeneFieldType.EXACT_MATCH,
                generator_type=generator_type,
                identifiers=identifiers,
                context_text=context,
                context_location=Coordinate(page_num=1),  # TODO: get actual page
                locus_type=gene_info.get("locus_type"),
                associated_diseases=disease_linkages,
                initial_confidence=confidence,
                provenance=provenance,
            )
            candidates.append(candidate)

        return candidates

    def _detect_gene_patterns(
        self,
        text: str,
        doc_graph: DocumentGraph,
        doc_fingerprint: str,
    ) -> List[GeneCandidate]:
        """Detect potential gene symbols using pattern matching with context validation."""
        candidates = []

        # Skip if we already have good lexicon coverage
        # Pattern matching is only for genes NOT in our lexicon

        for match in self.GENE_PATTERN.finditer(text):
            matched_text = match.group(1)
            start, end = match.span(1)

            # Skip if already in our lexicon
            if matched_text.lower() in self.orphadata_genes:
                continue
            if matched_text.lower() in self.alias_genes:
                continue

            context = self._extract_context(text, start, end)

            # Apply strict false positive filter for patterns
            is_fp, reason = self.fp_filter.is_false_positive(
                matched_text, context, GeneGeneratorType.PATTERN_GENE_SYMBOL, is_from_lexicon=False
            )
            if is_fp:
                continue

            # Build provenance
            provenance = GeneProvenanceMetadata(
                pipeline_version=self.pipeline_version,
                run_id=self.run_id,
                doc_fingerprint=doc_fingerprint,
                generator_name=GeneGeneratorType.PATTERN_GENE_SYMBOL,
            )

            candidate = GeneCandidate(
                doc_id=doc_graph.doc_id,
                matched_text=matched_text,
                hgnc_symbol=matched_text,  # Pattern match - use as-is
                field_type=GeneFieldType.PATTERN_MATCH,
                generator_type=GeneGeneratorType.PATTERN_GENE_SYMBOL,
                identifiers=[],
                context_text=context,
                context_location=Coordinate(page_num=1),
                initial_confidence=0.75,
                provenance=provenance,
            )
            candidates.append(candidate)

        return candidates

    def _detect_with_ner(
        self,
        text: str,
        doc_graph: DocumentGraph,
        doc_fingerprint: str,
    ) -> List[GeneCandidate]:
        """Detect genes using scispacy NER as fallback."""
        candidates: list[GeneCandidate] = []

        if not self.nlp:
            return candidates

        try:
            # Process with scispacy (limit text length for performance)
            max_len = 100000
            if len(text) > max_len:
                text = text[:max_len]

            doc = self.nlp(text)

            for ent in doc.ents:
                # Check UMLS linking
                if not hasattr(ent._, "kb_ents") or not ent._.kb_ents:
                    continue

                # Get best UMLS match
                best_cui, best_score = ent._.kb_ents[0]

                # Check semantic type
                linker = self.nlp.get_pipe("scispacy_linker")
                kb = linker.kb
                if best_cui not in kb.cui_to_entity:
                    continue

                entity = kb.cui_to_entity[best_cui]
                types = entity.types if hasattr(entity, "types") else []

                # Check if it's a gene semantic type
                is_gene = any(t in self.GENE_SEMANTIC_TYPES for t in types)
                if not is_gene:
                    continue

                matched_text = ent.text
                start, end = ent.start_char, ent.end_char

                # Skip if already in lexicon
                if matched_text.lower() in self.orphadata_genes:
                    continue
                if matched_text.lower() in self.alias_genes:
                    continue

                context = self._extract_context(text, start, end)

                # Apply false positive filter
                is_fp, reason = self.fp_filter.is_false_positive(
                    matched_text, context, GeneGeneratorType.SCISPACY_NER, is_from_lexicon=False
                )
                if is_fp:
                    continue

                # Build identifiers
                identifiers = [
                    GeneIdentifier(system="UMLS_CUI", code=best_cui)
                ]

                provenance = GeneProvenanceMetadata(
                    pipeline_version=self.pipeline_version,
                    run_id=self.run_id,
                    doc_fingerprint=doc_fingerprint,
                    generator_name=GeneGeneratorType.SCISPACY_NER,
                )

                candidate = GeneCandidate(
                    doc_id=doc_graph.doc_id,
                    matched_text=matched_text,
                    hgnc_symbol=matched_text,  # NER - use as-is
                    field_type=GeneFieldType.NER_DETECTION,
                    generator_type=GeneGeneratorType.SCISPACY_NER,
                    identifiers=identifiers,
                    context_text=context,
                    context_location=Coordinate(page_num=1),
                    initial_confidence=min(best_score, 0.70),
                    provenance=provenance,
                )
                candidates.append(candidate)

        except Exception as e:
            logger.warning("scispacy NER failed: %s", e)

        return candidates

    # Regex for gene symbols that benefit from hyphenated forms (e.g. IL6→IL-6)
    _HYPHEN_PATTERN = re.compile(r"^([A-Z]{1,4})(\d+)([A-Z]?)$")

    def _make_hyphenated(self, symbol: str) -> Optional[str]:
        """Generate hyphenated form for gene symbols like IL6→IL-6, MMP9→MMP-9."""
        m = self._HYPHEN_PATTERN.match(symbol)
        if m:
            prefix, digits, suffix = m.groups()
            return f"{prefix}-{digits}{suffix}"
        return None

    def _extract_context(self, text: str, start: int, end: int) -> str:
        """Extract context window around match."""
        ctx_start = max(0, start - self.context_window // 2)
        ctx_end = min(len(text), end + self.context_window // 2)
        return text[ctx_start:ctx_end]

    def _build_identifiers(self, gene_info: Dict) -> List[GeneIdentifier]:
        """Build list of gene identifiers from metadata."""
        identifiers = []

        if gene_info.get("hgnc_id"):
            identifiers.append(
                GeneIdentifier(system="HGNC", code=gene_info["hgnc_id"])
            )
        if gene_info.get("entrez_id"):
            identifiers.append(
                GeneIdentifier(system="ENTREZ", code=str(gene_info["entrez_id"]))
            )
        if gene_info.get("ensembl_id"):
            identifiers.append(
                GeneIdentifier(system="ENSEMBL", code=gene_info["ensembl_id"])
            )
        if gene_info.get("omim_id"):
            identifiers.append(
                GeneIdentifier(system="OMIM", code=gene_info["omim_id"])
            )
        if gene_info.get("uniprot_id"):
            identifiers.append(
                GeneIdentifier(system="UNIPROT", code=gene_info["uniprot_id"])
            )

        return identifiers

    def _deduplicate(self, candidates: List[GeneCandidate]) -> List[GeneCandidate]:
        """Deduplicate candidates, preferring higher priority sources."""
        # Priority: ORPHADATA > HGNC_ALIAS > PATTERN > NER
        priority = {
            GeneGeneratorType.LEXICON_ORPHADATA: 0,
            GeneGeneratorType.LEXICON_HGNC_ALIAS: 1,
            GeneGeneratorType.PATTERN_GENE_SYMBOL: 2,
            GeneGeneratorType.SCISPACY_NER: 3,
        }

        seen: Dict[str, GeneCandidate] = {}

        for candidate in candidates:
            # Deduplicate by HGNC symbol to avoid duplicates from aliases
            key = candidate.hgnc_symbol.upper() if candidate.hgnc_symbol else candidate.matched_text.lower()

            if key not in seen:
                seen[key] = candidate
            else:
                existing = seen[key]
                existing_priority = priority.get(existing.generator_type, 99)
                new_priority = priority.get(candidate.generator_type, 99)

                if new_priority < existing_priority:
                    seen[key] = candidate
                elif new_priority == existing_priority:
                    if candidate.initial_confidence > existing.initial_confidence:
                        seen[key] = candidate

        return list(seen.values())

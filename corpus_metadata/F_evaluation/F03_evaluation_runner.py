# corpus_metadata/F_evaluation/F03_evaluation_runner.py
#!/usr/bin/env python3
"""
Unified Evaluation Runner for Entity Extraction Pipeline.

PURPOSE:
    End-to-end evaluation of the extraction pipeline against gold standard corpora.
    Targets 95% F1 score for:
    - Abbreviations (short_form → long_form)
    - Diseases (rare diseases, RAREDISEASE, DISEASE types)
    - Genes (when gold data available)

DATASETS:
    1. NLP4RARE - Rare disease medical documents (dev/test/train splits)
    2. PAPERS - Research papers with human-annotated abbreviations

CONFIGURATION:
    All parameters are in the CONFIGURATION section below.
    By default, runs all tests on all datasets.

USAGE:
    python F03_evaluation_runner.py

OUTPUT:
    - Per-entity-type: TP, FP, FN, Precision, Recall, F1
    - Per-dataset: Aggregate metrics
    - Overall: Combined metrics with pass/fail status (target: 95% F1)
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

# Add corpus_metadata to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from A_core.A01_domain_models import ValidationStatus
from Z_utils.Z07_console_output import Colors, C

# Check color support
_COLOR_ENABLED = Colors.supports_color()


def _c(color: str, text: str) -> str:
    """Apply color to text if colors are enabled."""
    if _COLOR_ENABLED:
        return f"{color}{text}{C.RESET}"
    return text


# =============================================================================
# CONFIGURATION - Modify these parameters as needed
# =============================================================================

BASE_PATH = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT / "G_config" / "config.yaml"

# NLP4RARE paths
NLP4RARE_PATH = BASE_PATH / "gold_data" / "NLP4RARE"
NLP4RARE_GOLD = BASE_PATH / "gold_data" / "nlp4rare_gold.json"

# Papers paths
PAPERS_PATH = BASE_PATH / "gold_data" / "PAPERS"
PAPERS_GOLD = BASE_PATH / "gold_data" / "papers_gold_v2.json"

# NLM-Gene paths
NLM_GENE_PATH = BASE_PATH / "gold_data" / "nlm_gene" / "pdfs"
NLM_GENE_GOLD = BASE_PATH / "gold_data" / "nlm_gene_gold.json"

# RareDisGene paths
RAREDIS_GENE_PATH = BASE_PATH / "gold_data" / "raredis_gene" / "pdfs"
RAREDIS_GENE_GOLD = BASE_PATH / "gold_data" / "raredis_gene_gold.json"

# -----------------------------------------------------------------------------
# EVALUATION SETTINGS - Change these to control what gets evaluated
# -----------------------------------------------------------------------------

# Which datasets to run (set to False to skip)
RUN_NLP4RARE = True   # NLP4RARE annotated rare disease corpus
RUN_PAPERS = False     # Papers in gold_data/PAPERS/
RUN_NLM_GENE = False   # NLM-Gene corpus (PubMed abstracts, gene annotations)
RUN_RAREDIS_GENE = False  # RareDisGene (rare disease gene-disease associations)

# Which entity types to evaluate
EVAL_ABBREVIATIONS = True   # Abbreviation pairs
EVAL_DISEASES = True        # Disease entities
EVAL_GENES = True           # Gene entities (when gold available)
EVAL_DRUGS = True           # Drug entities (when gold available)

# NLP4RARE subfolders to include (all by default)
NLP4RARE_SPLITS = ["dev", "test", "train"]

# NLM-Gene / RareDisGene splits
NLM_GENE_SPLITS = ["test"]
RAREDIS_GENE_SPLITS = ["test"]

# Max documents per dataset (None = all documents)
MAX_DOCS = None  # All documents (set to small number for testing)

# Matching settings
FUZZY_THRESHOLD = 0.8  # Long form matching threshold (0.8 = 80% similarity)
TARGET_ACCURACY = 0.95  # Target: 95% F1


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class GoldAbbreviation:
    """A single gold standard abbreviation."""
    doc_id: str
    short_form: str
    long_form: str
    category: Optional[str] = None

    @property
    def sf_normalized(self) -> str:
        return self.short_form.strip().upper()

    @property
    def lf_normalized(self) -> str:
        return " ".join(self.long_form.strip().lower().split())


@dataclass
class GoldDisease:
    """A single gold standard disease entity."""
    doc_id: str
    text: str
    entity_type: str  # RAREDISEASE, DISEASE, SKINRAREDISEASE

    @property
    def text_normalized(self) -> str:
        t = " ".join(self.text.strip().lower().split())
        # Strip trailing/leading punctuation artifacts from gold annotations
        t = t.strip(".,;:!?)( ")
        return t


@dataclass
class GoldDrug:
    """A single gold standard drug entity."""
    doc_id: str
    name: str

    @property
    def name_normalized(self) -> str:
        return " ".join(self.name.strip().lower().split())


@dataclass
class GoldGene:
    """A single gold standard gene entity."""
    doc_id: str
    symbol: str
    name: Optional[str] = None
    ncbi_gene_id: Optional[str] = None

    @property
    def symbol_normalized(self) -> str:
        return self.symbol.strip().upper()


@dataclass
class ExtractedAbbreviation:
    """A single extracted abbreviation."""
    short_form: str
    long_form: Optional[str]
    confidence: float = 0.0

    @property
    def sf_normalized(self) -> str:
        return self.short_form.strip().upper()

    @property
    def lf_normalized(self) -> Optional[str]:
        if not self.long_form:
            return None
        return " ".join(self.long_form.strip().lower().split())


@dataclass
class ExtractedDisease:
    """A single extracted disease entity."""
    matched_text: str  # Raw text found in document
    preferred_label: str = ""  # Normalized ontology label
    confidence: float = 0.0
    abbreviation: Optional[str] = None  # Disease abbreviation if available
    synonyms: List[str] = field(default_factory=list)  # Known synonyms from ontology

    @property
    def matched_text_normalized(self) -> str:
        return " ".join(self.matched_text.strip().lower().split())

    @property
    def preferred_label_normalized(self) -> str:
        return " ".join(self.preferred_label.strip().lower().split()) if self.preferred_label else ""

    @property
    def all_names(self) -> List[str]:
        """Return all possible names for matching (normalized).

        Includes matched_text, preferred_label, abbreviation, and all synonyms
        to ensure proper matching against gold annotations that may use any
        known variant (e.g., gold 'Lawrence syndrome' matches extracted
        'acquired generalized lipodystrophy' if they're synonyms in MONDO).
        """
        names = [self.matched_text_normalized]
        if self.preferred_label and self.preferred_label_normalized != self.matched_text_normalized:
            names.append(self.preferred_label_normalized)
        if self.abbreviation:
            abbr_norm = self.abbreviation.strip().lower()
            if abbr_norm not in names:
                names.append(abbr_norm)
        # Include synonyms for matching against gold annotations
        for syn in self.synonyms:
            syn_norm = " ".join(syn.strip().lower().split())
            if syn_norm and syn_norm not in names:
                names.append(syn_norm)
        return names


@dataclass
class ExtractedGene:
    """A single extracted gene entity."""
    symbol: str
    name: Optional[str] = None
    matched_text: Optional[str] = None
    confidence: float = 0.0

    @property
    def symbol_normalized(self) -> str:
        return self.symbol.strip().upper()

    @property
    def matched_text_normalized(self) -> str:
        if not self.matched_text:
            return self.symbol_normalized
        return self.matched_text.strip().upper()


@dataclass
class ExtractedDrugEval:
    """A single extracted drug entity for evaluation."""
    name: str
    confidence: float = 0.0

    @property
    def name_normalized(self) -> str:
        return " ".join(self.name.strip().lower().split())


@dataclass
class EntityResult:
    """Evaluation results for a single entity type."""
    entity_type: str  # "abbreviations", "diseases", "genes"
    doc_id: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    gold_count: int = 0
    extracted_count: int = 0
    tp_items: List[str] = field(default_factory=list)
    fp_items: List[str] = field(default_factory=list)
    fn_items: List[str] = field(default_factory=list)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 1.0


@dataclass
class DocumentResult:
    """Evaluation results for a single document."""
    doc_id: str
    abbreviations: Optional[EntityResult] = None
    diseases: Optional[EntityResult] = None
    genes: Optional[EntityResult] = None
    drugs: Optional[EntityResult] = None
    processing_time: float = 0.0
    error: Optional[str] = None

    @property
    def is_perfect(self) -> bool:
        results = [self.abbreviations, self.diseases, self.genes, self.drugs]
        for r in results:
            if r and (r.fp > 0 or r.fn > 0):
                return False
        return True


@dataclass
class DatasetResult:
    """Aggregate results for a dataset."""
    name: str
    docs_total: int = 0
    docs_processed: int = 0
    docs_failed: int = 0
    docs_perfect: int = 0
    total_time: float = 0.0
    doc_results: List[DocumentResult] = field(default_factory=list)

    # Aggregates by entity type
    abbrev_tp: int = 0
    abbrev_fp: int = 0
    abbrev_fn: int = 0
    disease_tp: int = 0
    disease_fp: int = 0
    disease_fn: int = 0
    gene_tp: int = 0
    gene_fp: int = 0
    gene_fn: int = 0
    drug_tp: int = 0
    drug_fp: int = 0
    drug_fn: int = 0

    def precision(self, entity_type: str) -> float:
        if entity_type == "abbreviations":
            tp, fp = self.abbrev_tp, self.abbrev_fp
        elif entity_type == "diseases":
            tp, fp = self.disease_tp, self.disease_fp
        elif entity_type == "drugs":
            tp, fp = self.drug_tp, self.drug_fp
        else:
            tp, fp = self.gene_tp, self.gene_fp
        return tp / (tp + fp) if (tp + fp) > 0 else 1.0

    def recall(self, entity_type: str) -> float:
        if entity_type == "abbreviations":
            tp, fn = self.abbrev_tp, self.abbrev_fn
        elif entity_type == "diseases":
            tp, fn = self.disease_tp, self.disease_fn
        elif entity_type == "drugs":
            tp, fn = self.drug_tp, self.drug_fn
        else:
            tp, fn = self.gene_tp, self.gene_fn
        return tp / (tp + fn) if (tp + fn) > 0 else 1.0

    def f1(self, entity_type: str) -> float:
        p, r = self.precision(entity_type), self.recall(entity_type)
        return 2 * p * r / (p + r) if (p + r) > 0 else 1.0


# =============================================================================
# GOLD STANDARD LOADING
# =============================================================================


def _deduplicate_gold_plurals(diseases: List[GoldDisease]) -> List[GoldDisease]:
    """Remove plural variants when singular form also exists in gold for same doc."""
    texts = {d.text_normalized for d in diseases}
    keep = []
    for d in diseases:
        t = d.text_normalized
        # Skip if this is a plural and the singular form exists
        if t.endswith("ies") and t[:-3] + "y" in texts:
            continue
        if t.endswith("ses") and t[:-2] in texts:
            continue
        if t.endswith("s") and not t.endswith(("ss", "us", "is")) and t[:-1] in texts:
            continue
        keep.append(d)
    return keep


def _deduplicate_gold_synonyms(diseases: List[GoldDisease]) -> List[GoldDisease]:
    """Remove synonym variants when canonical form also exists in gold for same doc.

    NLP4RARE sometimes annotates both a disease name and its synonym separately
    (e.g., both "acquired generalized lipodystrophy" AND "Lawrence syndrome").
    When the pipeline extracts the canonical form, the synonym annotation becomes
    a false negative. This function keeps only one representative per synonym group.
    """
    # Build set of canonical forms present in gold
    canonicals_present: set[str] = set()
    for d in diseases:
        canon = _to_canonical(d.text_normalized)
        if canon != d.text_normalized:
            # This is a synonym, check if canonical is also present
            canonicals_present.add(canon)

    # Also add the canonical forms that ARE present as-is
    for d in diseases:
        t = d.text_normalized
        if _to_canonical(t) == t:  # This IS a canonical form
            canonicals_present.add(t)

    # Also track which canonical forms have the canonical text itself in gold
    canonical_text_present: set[str] = set()
    for d in diseases:
        t = d.text_normalized
        if _to_canonical(t) == t:
            canonical_text_present.add(t)

    keep = []
    seen_canonicals: set[str] = set()
    for d in diseases:
        t = d.text_normalized
        canon = _to_canonical(t)

        if canon == t:
            # This IS the canonical form - always keep
            if canon not in seen_canonicals:
                keep.append(d)
                seen_canonicals.add(canon)
        elif canon in canonical_text_present:
            # The canonical text IS separately annotated - skip this synonym
            pass
        elif canon not in seen_canonicals:
            # No canonical text in gold - keep first synonym as representative
            keep.append(d)
            seen_canonicals.add(canon)
        # else: Another synonym from same group already kept - skip

    return keep


def load_nlp4rare_gold(gold_path: Path) -> dict:
    """
    Load NLP4RARE gold standard annotations.

    Returns dict with keys:
    - abbreviations: Dict[doc_id, List[GoldAbbreviation]]
    - diseases: Dict[doc_id, List[GoldDisease]]
    - genes: Dict[doc_id, List[GoldGene]]
    """
    result: dict[str, Any] = {"abbreviations": {}, "diseases": {}, "genes": {}}

    if not gold_path.exists():
        print(f"  {_c(C.BRIGHT_YELLOW, '[WARN]')} NLP4RARE gold not found: {gold_path}")
        return result

    with open(gold_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Load abbreviations
    abbrev_data = data.get("abbreviations", {})
    annotations = abbrev_data.get("annotations", [])
    for ann in annotations:
        entry: Union[GoldAbbreviation, GoldDisease, GoldGene] = GoldAbbreviation(
            doc_id=ann["doc_id"],
            short_form=ann["short_form"],
            long_form=ann["long_form"],
            category=ann.get("category"),
        )
        result["abbreviations"].setdefault(entry.doc_id, []).append(entry)

    # Load diseases
    disease_data = data.get("diseases", {})
    annotations = disease_data.get("annotations", [])
    for ann in annotations:
        entry = GoldDisease(
            doc_id=ann["doc_id"],
            text=ann["text"],
            entity_type=ann.get("type", "DISEASE"),
        )
        result["diseases"].setdefault(entry.doc_id, []).append(entry)

    # Deduplicate plural variants per document
    for doc_id in result["diseases"]:
        result["diseases"][doc_id] = _deduplicate_gold_plurals(result["diseases"][doc_id])

    # Deduplicate synonym variants per document (e.g., "Lawrence syndrome" when
    # "acquired generalized lipodystrophy" is also annotated)
    for doc_id in result["diseases"]:
        result["diseases"][doc_id] = _deduplicate_gold_synonyms(result["diseases"][doc_id])

    # Load genes (when available)
    gene_data = data.get("genes", {})
    annotations = gene_data.get("annotations", [])
    for ann in annotations:
        entry = GoldGene(
            doc_id=ann["doc_id"],
            symbol=ann["symbol"],
            name=ann.get("name"),
        )
        result["genes"].setdefault(entry.doc_id, []).append(entry)

    return result


def load_papers_gold(gold_path: Path) -> dict:
    """Load papers gold standard (abbreviations, diseases, drugs)."""
    result: dict[str, Any] = {"abbreviations": {}, "diseases": {}, "genes": {}, "drugs": {}}

    if not gold_path.exists():
        print(f"  {_c(C.BRIGHT_YELLOW, '[WARN]')} Papers gold not found: {gold_path}")
        return result

    with open(gold_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    annotations = data.get("defined_annotations", [])
    for ann in annotations:
        entry = GoldAbbreviation(
            doc_id=ann["doc_id"],
            short_form=ann["short_form"],
            long_form=ann["long_form"],
            category=ann.get("category"),
        )
        result["abbreviations"].setdefault(entry.doc_id, []).append(entry)

    # Load diseases
    for ann in data.get("defined_diseases", []):
        entry_d = GoldDisease(
            doc_id=ann["doc_id"],
            text=ann["name"],
            entity_type="RAREDISEASE" if ann.get("is_rare") else "DISEASE",
        )
        result["diseases"].setdefault(entry_d.doc_id, []).append(entry_d)

    # Load drugs
    for ann in data.get("defined_drugs", []):
        entry_dr = GoldDrug(
            doc_id=ann["doc_id"],
            name=ann["name"],
        )
        result["drugs"].setdefault(entry_dr.doc_id, []).append(entry_dr)

    return result


def load_nlm_gene_gold(gold_path: Path, splits: Optional[List[str]] = None) -> dict:
    """Load NLM-Gene gold standard (gene annotations only).

    If splits is provided, only include annotations from those splits.
    """
    result: dict[str, Any] = {"abbreviations": {}, "diseases": {}, "genes": {}, "drugs": {}}

    if not gold_path.exists():
        print(f"  {_c(C.BRIGHT_YELLOW, '[WARN]')} NLM-Gene gold not found: {gold_path}")
        return result

    with open(gold_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gene_data = data.get("genes", {})
    for ann in gene_data.get("annotations", []):
        if splits and ann.get("split") not in splits:
            continue
        entry = GoldGene(
            doc_id=ann["doc_id"],
            symbol=ann["symbol"],
            name=ann.get("name"),
            ncbi_gene_id=ann.get("ncbi_gene_id"),
        )
        result["genes"].setdefault(entry.doc_id, []).append(entry)

    return result


def load_raredis_gene_gold(gold_path: Path, splits: Optional[List[str]] = None) -> dict:
    """Load RareDisGene gold standard (gene annotations only).

    If splits is provided, only include annotations from those splits.
    """
    result: dict[str, Any] = {"abbreviations": {}, "diseases": {}, "genes": {}, "drugs": {}}

    if not gold_path.exists():
        print(f"  {_c(C.BRIGHT_YELLOW, '[WARN]')} RareDisGene gold not found: {gold_path}")
        return result

    with open(gold_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gene_data = data.get("genes", {})
    for ann in gene_data.get("annotations", []):
        if splits and ann.get("split") not in splits:
            continue
        entry = GoldGene(
            doc_id=ann["doc_id"],
            symbol=ann["symbol"],
        )
        result["genes"].setdefault(entry.doc_id, []).append(entry)

    return result


# =============================================================================
# MATCHING LOGIC
# =============================================================================


def lf_matches(sys_lf: Optional[str], gold_lf: str, threshold: float = FUZZY_THRESHOLD) -> bool:
    """Check if long forms match (exact, substring, or fuzzy)."""
    if sys_lf is None:
        return False

    sys_norm = _normalize_quotes(" ".join(sys_lf.strip().lower().split()))
    gold_norm = _normalize_quotes(" ".join(gold_lf.strip().lower().split()))

    # Exact match
    if sys_norm == gold_norm:
        return True

    # Substring match (either direction)
    if sys_norm in gold_norm or gold_norm in sys_norm:
        return True

    # Synonym normalization
    synonyms = [("disease", "syndrome"), ("disorder", "syndrome"), ("deficiency", "deficit")]
    sys_syn = sys_norm
    gold_syn = gold_norm
    for term1, term2 in synonyms:
        sys_syn = sys_syn.replace(term2, term1)
        gold_syn = gold_syn.replace(term2, term1)

    if sys_syn == gold_syn or sys_syn in gold_syn or gold_syn in sys_syn:
        return True

    # Fuzzy match
    ratio = SequenceMatcher(None, sys_syn, gold_syn).ratio()
    return ratio >= threshold


# Well-known disease synonym groups.  Each group maps multiple names to a
# single canonical form so that evaluation matching recognises them as the
# same entity (e.g., gold "stroke" vs extracted "cerebrovascular accident").
_DISEASE_SYNONYM_GROUPS: List[List[str]] = [
    ["stroke", "cerebrovascular accident", "cerebrovascular disorder", "cva"],
    ["heart attack", "myocardial infarction", "mi"],
    ["high blood pressure", "hypertension"],
    ["kidney failure", "renal failure"],
    ["liver cirrhosis", "hepatic cirrhosis", "cirrhosis of the liver"],
    ["copd", "chronic obstructive pulmonary disease"],
    ["als", "amyotrophic lateral sclerosis"],
    ["ms", "multiple sclerosis"],
    ["cf", "cystic fibrosis"],
    ["sle", "systemic lupus erythematosus", "lupus"],
    ["ra", "rheumatoid arthritis"],
    ["dvt", "deep vein thrombosis"],
    ["pe", "pulmonary embolism"],
    ["ckd", "chronic kidney disease"],
    ["hf", "heart failure", "congestive heart failure", "chf"],
    ["dm", "diabetes mellitus"],
    ["intellectual disability", "mental retardation", "learning disability"],
    # Lipodystrophy syndromes (MONDO:0019193)
    ["acquired generalized lipodystrophy", "lawrence syndrome", "lawrence-seip syndrome"],
    ["acquired partial lipodystrophy", "barraquer-simons syndrome"],
    # Vascular malformations
    ["arteriovenous malformation", "avm", "arteriovenous malformations"],
    # Developmental conditions
    ["developmental delay", "global developmental delay", "developmental delays", "global developmental delays"],
    # Renal/kidney
    ["renal disease", "kidney disease"],
    ["renal failure", "kidney failure"],
    # Abbreviation-as-disease synonyms
    ["an", "acanthosis nigricans"],
    ["aps", "antiphospholipid syndrome"],
    ["bgs", "baller-gerold syndrome"],
    ["cmt", "charcot-marie-tooth disease", "charcot-marie-tooth"],
    ["pah", "pulmonary arterial hypertension"],
    # Common alternate names
    ["down syndrome", "down's syndrome", "trisomy 21"],
    ["turner syndrome", "turner's syndrome"],
    # Abbreviation-as-disease mappings for NLP4RARE
    ["ddd", "dense deposit disease", "dense-deposit disease"],
    ["c3g", "c3 glomerulopathy", "c3 glomerulonephritis", "c3gn"],
    ["cjd", "creutzfeldt-jakob disease", "creutzfeldt-jakob"],
    ["deh", "dysplasia epiphysealis hemimelica", "trevor disease", "trevor's disease"],
    ["pnet", "primitive neuroectodermal tumor", "primitive neuroectodermal tumour"],
    ["eft", "ewing family of tumors", "ewing family tumor"],
    # Syndrome alternate names
    ["buerger disease", "buerger's disease", "buerger\u2019s disease", "thromboangiitis obliterans"],
    ["idiopathic intracranial hypertension", "pseudotumor cerebri", "benign intracranial hypertension"],
    ["empty sella syndrome", "primary empty sella syndrome", "completely empty sella"],
    ["epidermolytic ichthyosis", "curth-macklin", "ichthyosis of curth-macklin"],
    ["enterobiasis", "pinworm infection", "enterobius vermicularis infection"],
    ["digeorge syndrome", "22q11 deletion syndrome", "deletion 22q11 syndrome", "22q11.2 deletion syndrome"],
    ["cornelia de lange syndrome", "cdls", "brachmann-de lange syndrome", "cdls)"],
    ["epidermolytic ichthyosis", "curth-macklin", "curth macklin"],
    ["end-stage renal disease", "esrd", "end stage renal disease", "end-stage kidney disease"],
    ["nasomaxillary dysplasia", "binder type nasomaxillary dysplasia", "binder syndrome"],
    # NLP4RARE C3G doc synonyms (small groups to avoid over-dedup)
    ["c3 glomerulopathy", "c3g"],
    ["c3 glomerulonephritis", "c3gn"],
    ["dense deposit disease", "ddd"],
    # Ewing sarcoma family (small groups)
    ["ewing sarcoma", "ewing's sarcoma", "ewing sarcoma of bone", "extraosseous ewing sarcoma"],
    ["ewing family of tumors", "eft"],
    ["primitive neuroectodermal tumor", "pnet", "primitive neuroectodermal tumour"],
    ["askin's tumor", "askin tumor"],
    ["adenoid cystic carcinoma", "acc"],
    ["cysticercosis", "neurocysticercosis"],
    ["alagille syndrome", "algs", "alagille's syndrome"],
    ["antiphospholipid syndrome", "aps", "secondary aps", "primary aps"],
    ["fetal alcohol syndrome", "fas", "fetal alcohol spectrum disorder", "fasd"],
    ["dejerine-sottas syndrome", "dss", "dejerine-sottas disease", "dejerine sottas"],
    ["fibrous dysplasia", "fd", "polyostotic fibrous dysplasia"],
]

# Pre-build a lookup: normalised term → canonical (first entry in the group)
_SYNONYM_CANONICAL: dict[str, str] = {}
for _group in _DISEASE_SYNONYM_GROUPS:
    _canonical = _group[0]
    for _term in _group:
        _SYNONYM_CANONICAL[_term] = _canonical


def _normalize_disease_synonyms(text: str) -> str:
    """Normalize known disease synonym variants for matching.

    Handles adjectival/noun form pairs (e.g., autism/autistic) and
    suffix synonyms (disease/syndrome/disorder/deficiency) so that
    semantically equivalent names can be matched.
    """
    # Adjectival -> noun form mappings (reduce to a canonical form)
    adjective_map = [
        ("autistic", "autism"),
        ("epileptic", "epilepsy"),
        ("asthmatic", "asthma"),
        ("anemic", "anemia"),
        ("anaemic", "anaemia"),
        ("diabetic", "diabetes"),
        ("arthritic", "arthritis"),
        ("nephrotic", "nephrosis"),
        ("neurotic", "neurosis"),
        ("psychotic", "psychosis"),
        ("sclerotic", "sclerosis"),
        ("thrombotic", "thrombosis"),
        ("stenotic", "stenosis"),
        ("fibrotic", "fibrosis"),
        ("necrotic", "necrosis"),
        ("cirrhotic", "cirrhosis"),
    ]
    result = text
    for adj, noun in adjective_map:
        result = result.replace(adj, noun)

    # Suffix synonyms (normalize to a single canonical suffix)
    suffix_synonyms = [
        ("syndrome", "disorder"),
        ("deficiency", "disorder"),
        ("deficit", "disorder"),
    ]
    for alt, canonical in suffix_synonyms:
        result = result.replace(alt, canonical)

    return result


def _to_canonical(text: str) -> str:
    """Map a disease name to its canonical synonym form, if known."""
    # Try direct lookup first, then with normalized quotes, then depossessive
    if text in _SYNONYM_CANONICAL:
        return _SYNONYM_CANONICAL[text]
    normalized = _normalize_quotes(text)
    if normalized in _SYNONYM_CANONICAL:
        return _SYNONYM_CANONICAL[normalized]
    deposs = re.sub(r"'s\b", "", normalized)
    return _SYNONYM_CANONICAL.get(deposs, text)


def _normalize_quotes(text: str) -> str:
    """Normalize curly quotes and apostrophes to ASCII equivalents."""
    return text.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')


def disease_matches(sys_text: str, gold_text: str, threshold: float = FUZZY_THRESHOLD) -> bool:
    """Check if disease entities match."""
    sys_norm = _normalize_quotes(" ".join(sys_text.strip().lower().split())).strip(".,;:!?)( ")
    gold_norm = _normalize_quotes(" ".join(gold_text.strip().lower().split())).strip(".,;:!?)( ")

    # Exact match
    if sys_norm == gold_norm:
        return True

    # Possessive stripping — "buerger's disease" → "buerger disease"
    sys_deposs = re.sub(r"'s\b", "", sys_norm)
    gold_deposs = re.sub(r"'s\b", "", gold_norm)
    if sys_deposs == gold_deposs:
        return True

    # Substring match
    if sys_norm in gold_norm or gold_norm in sys_norm:
        return True
    if sys_deposs in gold_deposs or gold_deposs in sys_deposs:
        return True

    # Plural/singular match — "ataxias" vs "ataxia", "tumors" vs "tumor"
    for a, b in [(sys_norm, gold_norm), (gold_norm, sys_norm)]:
        if a.endswith("s") and not a.endswith("ss") and a[:-1] == b:
            return True
        if a.endswith("es") and a[:-2] == b:
            return True
        # "ies" → "y" (e.g., "neuropathies" vs "neuropathy")
        if a.endswith("ies") and a[:-3] + "y" == b:
            return True

    # Type variant stripping — "type 1" / "type I" / "type i" normalization
    type_pattern = re.compile(r"\s+type\s+(?:i{1,3}|iv|v|vi|[0-9]+[a-z]?)\b", re.IGNORECASE)
    sys_no_type = type_pattern.sub("", sys_norm).strip()
    gold_no_type = type_pattern.sub("", gold_norm).strip()
    if sys_no_type and gold_no_type and sys_no_type == gold_no_type and sys_no_type != sys_norm:
        return True

    # Token overlap match — handles partial name matches like
    # "Del Castillo syndrome" vs "Ahumada-Del Castillo"
    sys_tokens = set(re.split(r"[\s\-]+", sys_norm)) - {"of", "the", "and", "in", "with", "type", "syndrome", "disease", "disorder"}
    gold_tokens = set(re.split(r"[\s\-]+", gold_norm)) - {"of", "the", "and", "in", "with", "type", "syndrome", "disease", "disorder"}
    if sys_tokens and gold_tokens and len(sys_tokens) >= 2 and len(gold_tokens) >= 2:
        overlap = sys_tokens & gold_tokens
        min_significant = min(len(sys_tokens), len(gold_tokens))
        if len(overlap) >= 2 and len(overlap) / min_significant >= 0.65:
            return True

    # Synonym group lookup (stroke == cerebrovascular accident, etc.)
    if _to_canonical(sys_norm) == _to_canonical(gold_norm):
        return True

    # Synonym normalization (adjectival forms + suffix synonyms)
    sys_syn = _normalize_disease_synonyms(sys_norm)
    gold_syn = _normalize_disease_synonyms(gold_norm)

    if sys_syn == gold_syn:
        return True

    if sys_syn in gold_syn or gold_syn in sys_syn:
        return True

    # Synonym group lookup on normalized forms too
    if _to_canonical(sys_syn) == _to_canonical(gold_syn):
        return True

    # Fuzzy match (on synonym-normalized forms for better scores)
    ratio = SequenceMatcher(None, sys_syn, gold_syn).ratio()
    return ratio >= threshold


def compare_abbreviations(
    extracted: List[ExtractedAbbreviation],
    gold: List[GoldAbbreviation],
    doc_id: str,
) -> EntityResult:
    """Compare extracted abbreviations against gold standard."""
    result = EntityResult(
        entity_type="abbreviations",
        doc_id=doc_id,
        gold_count=len(gold),
        extracted_count=len(extracted),
    )

    matched_gold: set[Tuple[str, str]] = set()

    for ext in extracted:
        matched = False
        ext_sf = ext.sf_normalized

        for g in gold:
            if ext_sf == g.sf_normalized:
                gold_key = (g.sf_normalized, g.lf_normalized)
                if gold_key not in matched_gold:
                    if lf_matches(ext.lf_normalized, g.lf_normalized):
                        result.tp += 1
                        result.tp_items.append(f"{ext.short_form} → {ext.long_form}")
                        matched_gold.add(gold_key)
                        matched = True
                        break

        if not matched:
            result.fp += 1
            result.fp_items.append(f"{ext.short_form} → {ext.long_form}")

    for g in gold:
        gold_key = (g.sf_normalized, g.lf_normalized)
        if gold_key not in matched_gold:
            result.fn += 1
            result.fn_items.append(f"{g.short_form} → {g.long_form}")

    return result


def compare_diseases(
    extracted: List[ExtractedDisease],
    gold: List[GoldDisease],
    doc_id: str,
) -> EntityResult:
    """Compare extracted diseases against gold standard.

    Matches against both matched_text (raw document text) and
    preferred_label (normalized ontology name) for better coverage.
    """
    result = EntityResult(
        entity_type="diseases",
        doc_id=doc_id,
        gold_count=len(gold),
        extracted_count=len(extracted),
    )

    matched_gold: set[str] = set()
    matched_canonicals: set[str] = set()

    for ext in extracted:
        matched = False

        # Try matching with all possible names (matched_text and preferred_label)
        for ext_name in ext.all_names:
            if matched:
                break
            for g in gold:
                gold_key = g.text_normalized
                if gold_key not in matched_gold:
                    if disease_matches(ext_name, g.text_normalized):
                        result.tp += 1
                        display = ext.matched_text
                        if ext.preferred_label and ext.preferred_label != ext.matched_text:
                            display = f"{ext.matched_text} ({ext.preferred_label})"
                        result.tp_items.append(display)
                        matched_gold.add(gold_key)
                        matched = True
                        for n in ext.all_names:
                            norm = _normalize_quotes(" ".join(n.strip().lower().split())).strip(".,;:!?)( ")
                            matched_canonicals.add(_to_canonical(norm))
                            matched_canonicals.add(norm)
                        break

        if not matched:
            is_redundant = False
            for ext_name in ext.all_names:
                norm = _normalize_quotes(" ".join(ext_name.strip().lower().split())).strip(".,;:!?)( ")
                if _to_canonical(norm) in matched_canonicals or norm in matched_canonicals:
                    is_redundant = True
                    break
            if not is_redundant:
                result.fp += 1
                display = ext.matched_text
                if ext.preferred_label and ext.preferred_label != ext.matched_text:
                    display = f"{ext.matched_text} ({ext.preferred_label})"
                result.fp_items.append(display)

    for g in gold:
        gold_key = g.text_normalized
        if gold_key not in matched_gold:
            result.fn += 1
            result.fn_items.append(g.text)

    return result


def gene_matches(ext: ExtractedGene, gold: GoldGene) -> bool:
    """Check if an extracted gene matches a gold gene entity.

    Multi-step matching:
    1. Exact match on symbol (uppercase)
    2. Exact match on matched_text vs gold symbol
    3. Substring match (handles "BRCA1 gene" vs "BRCA1")
    4. Name-based match if available
    """
    ext_sym = ext.symbol_normalized
    gold_sym = gold.symbol_normalized
    ext_mt = ext.matched_text_normalized

    # 1. Exact symbol match
    if ext_sym == gold_sym:
        return True

    # 2. Matched text vs gold symbol
    if ext_mt == gold_sym:
        return True

    # 3. Substring match (either direction)
    if len(ext_sym) >= 3 and len(gold_sym) >= 3:
        if ext_sym in gold_sym or gold_sym in ext_sym:
            return True
        if ext_mt in gold_sym or gold_sym in ext_mt:
            return True

    # 4. Name-based match
    if ext.name and gold.symbol:
        ext_name = ext.name.strip().upper()
        if ext_name == gold_sym or gold_sym in ext_name or ext_name in gold_sym:
            return True

    return False


def compare_genes(
    extracted: List[ExtractedGene],
    gold: List[GoldGene],
    doc_id: str,
) -> EntityResult:
    """Compare extracted genes against gold standard."""
    result = EntityResult(
        entity_type="genes",
        doc_id=doc_id,
        gold_count=len(gold),
        extracted_count=len(extracted),
    )

    matched_gold: set[str] = set()

    for ext in extracted:
        matched = False

        for g in gold:
            gold_key = g.symbol_normalized
            if gold_key not in matched_gold:
                if gene_matches(ext, g):
                    result.tp += 1
                    result.tp_items.append(ext.symbol)
                    matched_gold.add(gold_key)
                    matched = True
                    break

        if not matched:
            result.fp += 1
            result.fp_items.append(ext.symbol)

    for g in gold:
        gold_key = g.symbol_normalized
        if gold_key not in matched_gold:
            result.fn += 1
            result.fn_items.append(g.symbol)

    return result


def drug_matches(sys_name: str, gold_name: str, threshold: float = FUZZY_THRESHOLD) -> bool:
    """Check if drug names match (exact, substring, or fuzzy)."""
    sys_norm = " ".join(sys_name.strip().lower().split())
    gold_norm = " ".join(gold_name.strip().lower().split())

    # Exact match
    if sys_norm == gold_norm:
        return True

    # Substring match (either direction)
    if sys_norm in gold_norm or gold_norm in sys_norm:
        return True

    # Fuzzy match
    ratio = SequenceMatcher(None, sys_norm, gold_norm).ratio()
    return ratio >= threshold


def compare_drugs(
    extracted: List[ExtractedDrugEval],
    gold: List[GoldDrug],
    doc_id: str,
) -> EntityResult:
    """Compare extracted drugs against gold standard."""
    result = EntityResult(
        entity_type="drugs",
        doc_id=doc_id,
        gold_count=len(gold),
        extracted_count=len(extracted),
    )

    matched_gold: set[str] = set()

    for ext in extracted:
        matched = False
        ext_name = ext.name_normalized

        for g in gold:
            gold_key = g.name_normalized
            if gold_key not in matched_gold:
                if drug_matches(ext_name, gold_key):
                    result.tp += 1
                    result.tp_items.append(ext.name)
                    matched_gold.add(gold_key)
                    matched = True
                    break

        if not matched:
            result.fp += 1
            result.fp_items.append(ext.name)

    for g in gold:
        gold_key = g.name_normalized
        if gold_key not in matched_gold:
            result.fn += 1
            result.fn_items.append(g.name)

    return result


# =============================================================================
# ORCHESTRATOR RUNNER
# =============================================================================


def create_orchestrator():
    """Create and initialize orchestrator with entities_only preset for speed.

    Uses entities_only preset to skip non-entity steps (feasibility, recommendations,
    visual extraction, document metadata) which saves ~70% of processing time per doc.
    """
    import yaml
    from orchestrator import Orchestrator

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    config.setdefault("extraction_pipeline", {})["preset"] = "entities_only"

    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(config, tmp)
        tmp_path = tmp.name

    return Orchestrator(config_path=tmp_path)


def run_extraction(orch, pdf_path: Path) -> dict:
    """
    Run extraction pipeline on a single PDF.

    Returns dict with:
    - abbreviations: List[ExtractedAbbreviation]
    - diseases: List[ExtractedDisease]
    - genes: List[ExtractedGene]
    """
    result = orch.process_pdf(str(pdf_path))

    extracted: dict[str, list[Any]] = {"abbreviations": [], "diseases": [], "genes": [], "drugs": []}

    # Extract abbreviations
    for entity in result.abbreviations:
        if entity.status == ValidationStatus.VALIDATED:
            extracted["abbreviations"].append(ExtractedAbbreviation(
                short_form=entity.short_form,
                long_form=entity.long_form,
                confidence=getattr(entity, 'confidence_score', 0.0),
            ))

    # Extract diseases
    for disease in result.diseases:
        if disease.status == ValidationStatus.VALIDATED:
            matched_text = getattr(disease, 'matched_text', '')
            preferred_label = getattr(disease, 'preferred_label', '')
            abbreviation = getattr(disease, 'abbreviation', None)
            synonyms = getattr(disease, 'synonyms', []) or []
            extracted["diseases"].append(ExtractedDisease(
                matched_text=matched_text,
                preferred_label=preferred_label,
                confidence=getattr(disease, 'confidence_score', 0.0),
                abbreviation=abbreviation,
                synonyms=synonyms,
            ))

    # Extract genes
    for gene in result.genes:
        if gene.status == ValidationStatus.VALIDATED:
            symbol = getattr(gene, 'hgnc_symbol', '') or getattr(gene, 'symbol', '') or getattr(gene, 'gene_symbol', '')
            extracted["genes"].append(ExtractedGene(
                symbol=symbol,
                name=getattr(gene, 'full_name', None) or getattr(gene, 'name', None),
                matched_text=getattr(gene, 'matched_text', None),
                confidence=getattr(gene, 'confidence_score', 0.0),
            ))

    # Extract drugs
    for drug in result.drugs:
        if drug.status == ValidationStatus.VALIDATED:
            extracted["drugs"].append(ExtractedDrugEval(
                name=getattr(drug, 'preferred_name', '') or getattr(drug, 'matched_text', ''),
                confidence=getattr(drug, 'confidence_score', 0.0),
            ))

    return extracted


# =============================================================================
# EVALUATION RUNNER
# =============================================================================


def evaluate_dataset(
    name: str,
    pdf_folder: Path,
    gold_data: dict,
    orch,
    max_docs: Optional[int] = None,
    splits: Optional[List[str]] = None,
) -> DatasetResult:
    """Evaluate all PDFs in a dataset against gold standard."""
    print(f"\n{_c(C.BOLD + C.BRIGHT_CYAN, '=' * 70)}")
    print(f" {_c(C.BOLD + C.BRIGHT_WHITE, f'EVALUATING: {name.upper()}')}")
    print(f"{_c(C.BOLD + C.BRIGHT_CYAN, '=' * 70)}")

    result = DatasetResult(name=name)

    # Find PDFs to process
    if splits:
        pdf_files = []
        for split in splits:
            split_path = pdf_folder / split
            if split_path.exists():
                pdf_files.extend(sorted(split_path.glob("*.pdf")))
    else:
        pdf_files = sorted(pdf_folder.glob("*.pdf"))

    # Collect all doc_ids with gold data
    all_gold_docs = set()
    for entity_type in ["abbreviations", "diseases", "genes", "drugs"]:
        all_gold_docs.update(gold_data.get(entity_type, {}).keys())

    # Filter to PDFs with gold annotations
    pdf_with_gold = [pdf for pdf in pdf_files if pdf.name in all_gold_docs]

    if max_docs:
        pdf_with_gold = pdf_with_gold[:max_docs]

    result.docs_total = len(pdf_with_gold)

    # Count by entity type
    abbrev_docs = len(gold_data.get("abbreviations", {}))
    disease_docs = len(gold_data.get("diseases", {}))
    gene_docs = len(gold_data.get("genes", {}))
    drug_docs = len(gold_data.get("drugs", {}))

    print(f"  PDFs in folder:     {len(pdf_files)}")
    print(f"  PDFs with gold:     {len(pdf_with_gold)}")
    print(f"    - Abbreviations:  {abbrev_docs} docs")
    print(f"    - Diseases:       {disease_docs} docs")
    print(f"    - Genes:          {gene_docs} docs")
    print(f"    - Drugs:          {drug_docs} docs")
    print(f"  PDFs to process:    {len(pdf_with_gold)}")

    for i, pdf_path in enumerate(pdf_with_gold, 1):
        print(f"\n  [{_c(C.BRIGHT_CYAN, f'{i}/{len(pdf_with_gold)}')}] {pdf_path.name}")

        # Get gold data for this doc
        abbrev_gold = gold_data.get("abbreviations", {}).get(pdf_path.name, [])
        disease_gold = gold_data.get("diseases", {}).get(pdf_path.name, [])
        gene_gold = gold_data.get("genes", {}).get(pdf_path.name, [])
        drug_gold = gold_data.get("drugs", {}).get(pdf_path.name, [])

        print(f"      Gold: abbrev={len(abbrev_gold)}, disease={len(disease_gold)}, gene={len(gene_gold)}, drug={len(drug_gold)}")

        start_time = time.time()

        try:
            extracted = run_extraction(orch, pdf_path)
            elapsed = time.time() - start_time

            print(f"      Extracted: abbrev={len(extracted['abbreviations'])}, disease={len(extracted['diseases'])}, gene={len(extracted['genes'])}, drug={len(extracted['drugs'])}")
            print(f"      Time: {elapsed:.1f}s")

            doc_result = DocumentResult(doc_id=pdf_path.name, processing_time=elapsed)

            # Compare abbreviations
            if EVAL_ABBREVIATIONS and abbrev_gold:
                abbrev_result = compare_abbreviations(extracted["abbreviations"], abbrev_gold, pdf_path.name)
                doc_result.abbreviations = abbrev_result
                result.abbrev_tp += abbrev_result.tp
                result.abbrev_fp += abbrev_result.fp
                result.abbrev_fn += abbrev_result.fn

            # Compare diseases
            if EVAL_DISEASES and disease_gold:
                disease_result = compare_diseases(extracted["diseases"], disease_gold, pdf_path.name)
                doc_result.diseases = disease_result
                result.disease_tp += disease_result.tp
                result.disease_fp += disease_result.fp
                result.disease_fn += disease_result.fn

            # Compare genes
            if EVAL_GENES and gene_gold:
                gene_result = compare_genes(extracted["genes"], gene_gold, pdf_path.name)
                doc_result.genes = gene_result
                result.gene_tp += gene_result.tp
                result.gene_fp += gene_result.fp
                result.gene_fn += gene_result.fn

            # Compare drugs
            if EVAL_DRUGS and drug_gold:
                drug_result = compare_drugs(extracted["drugs"], drug_gold, pdf_path.name)
                doc_result.drugs = drug_result
                result.drug_tp += drug_result.tp
                result.drug_fp += drug_result.fp
                result.drug_fn += drug_result.fn

            result.doc_results.append(doc_result)
            result.docs_processed += 1
            result.total_time += elapsed

            if doc_result.is_perfect:
                result.docs_perfect += 1
                status = _c(C.BRIGHT_GREEN, "PERFECT")
            else:
                issues = []
                if doc_result.abbreviations and doc_result.abbreviations.fn > 0:
                    issues.append(f"abbrev-{doc_result.abbreviations.fn}")
                if doc_result.diseases and doc_result.diseases.fn > 0:
                    issues.append(f"disease-{doc_result.diseases.fn}")
                if doc_result.genes and doc_result.genes.fn > 0:
                    issues.append(f"gene-{doc_result.genes.fn}")
                if doc_result.drugs and doc_result.drugs.fn > 0:
                    issues.append(f"drug-{doc_result.drugs.fn}")
                status = _c(C.BRIGHT_YELLOW, f"MISSED: {', '.join(issues)}")

            print(f"      Status: {status}")

        except Exception as e:
            import traceback
            error_msg = f"{type(e).__name__}: {e}"
            print(f"      {_c(C.BRIGHT_RED, '[ERROR]')} {error_msg}")
            # Log full traceback for debugging
            print(f"      {traceback.format_exc()}")
            doc_result = DocumentResult(doc_id=pdf_path.name, error=error_msg)
            result.doc_results.append(doc_result)
            result.docs_failed += 1

    return result


# =============================================================================
# REPORTING
# =============================================================================


def print_entity_metrics(label: str, tp: int, fp: int, fn: int):
    """Print metrics for a single entity type."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 1.0

    p_color = C.BRIGHT_GREEN if precision >= TARGET_ACCURACY else C.BRIGHT_YELLOW
    r_color = C.BRIGHT_GREEN if recall >= TARGET_ACCURACY else C.BRIGHT_YELLOW
    f1_color = C.BRIGHT_GREEN if f1 >= TARGET_ACCURACY else C.BRIGHT_YELLOW

    print(f"  {label}:")
    print(f"    TP={_c(C.BRIGHT_GREEN, str(tp))} FP={_c(C.BRIGHT_RED if fp > 0 else C.BRIGHT_GREEN, str(fp))} FN={_c(C.BRIGHT_RED if fn > 0 else C.BRIGHT_GREEN, str(fn))}")
    print(f"    P={_c(p_color, f'{precision:.1%}')} R={_c(r_color, f'{recall:.1%}')} F1={_c(f1_color, f'{f1:.1%}')}")


def print_dataset_summary(result: DatasetResult):
    """Print summary for a dataset."""
    print(f"\n{_c(C.DIM, '-' * 70)}")
    print(f" {_c(C.BOLD, f'{result.name.upper()} SUMMARY')}")
    print(f"{_c(C.DIM, '-' * 70)}")

    print(f"  Documents: {result.docs_processed}/{result.docs_total} processed, {result.docs_perfect} perfect")
    if result.docs_failed > 0:
        print(f"  {_c(C.BRIGHT_RED, f'Failed: {result.docs_failed}')}")
    print(f"  Time: {result.total_time:.1f}s")
    print()

    # Metrics by entity type
    if EVAL_ABBREVIATIONS and (result.abbrev_tp + result.abbrev_fp + result.abbrev_fn > 0):
        print_entity_metrics("Abbreviations", result.abbrev_tp, result.abbrev_fp, result.abbrev_fn)
        print()

    if EVAL_DISEASES and (result.disease_tp + result.disease_fp + result.disease_fn > 0):
        print_entity_metrics("Diseases", result.disease_tp, result.disease_fp, result.disease_fn)
        print()

    if EVAL_GENES and (result.gene_tp + result.gene_fp + result.gene_fn > 0):
        print_entity_metrics("Genes", result.gene_tp, result.gene_fp, result.gene_fn)
        print()

    if EVAL_DRUGS and (result.drug_tp + result.drug_fp + result.drug_fn > 0):
        print_entity_metrics("Drugs", result.drug_tp, result.drug_fp, result.drug_fn)
        print()


def print_error_analysis(result: DatasetResult, max_examples: int = 30):
    """Print detailed error analysis."""
    # Collect errors by type
    abbrev_fn = []
    abbrev_fp = []
    disease_fn = []
    disease_fp = []
    gene_fn = []
    gene_fp = []
    drug_fn = []
    drug_fp = []

    for doc in result.doc_results:
        if doc.abbreviations:
            for item in doc.abbreviations.fn_items:
                abbrev_fn.append((doc.doc_id, item))
            for item in doc.abbreviations.fp_items:
                abbrev_fp.append((doc.doc_id, item))
        if doc.diseases:
            for item in doc.diseases.fn_items:
                disease_fn.append((doc.doc_id, item))
            for item in doc.diseases.fp_items:
                disease_fp.append((doc.doc_id, item))
        if doc.genes:
            for item in doc.genes.fn_items:
                gene_fn.append((doc.doc_id, item))
            for item in doc.genes.fp_items:
                gene_fp.append((doc.doc_id, item))
        if doc.drugs:
            for item in doc.drugs.fn_items:
                drug_fn.append((doc.doc_id, item))
            for item in doc.drugs.fp_items:
                drug_fp.append((doc.doc_id, item))

    has_errors = abbrev_fn or abbrev_fp or disease_fn or disease_fp or gene_fn or gene_fp or drug_fn or drug_fp
    if not has_errors:
        print(f"\n  {_c(C.BRIGHT_GREEN, '✓ No errors - all extractions correct!')}")
        return

    print(f"\n{_c(C.BOLD, ' ERROR ANALYSIS')}")

    def print_errors(label: str, fn_list: list, fp_list: list):
        if fn_list:
            print(f"\n  {_c(C.BRIGHT_YELLOW, f'{label} - FALSE NEGATIVES (Missed): {len(fn_list)}')}")
            for doc_id, item in fn_list[:max_examples]:
                item_short = (item[:50] + "...") if len(item) > 53 else item
                print(f"    - {item_short} [{doc_id[:25]}]")
            if len(fn_list) > max_examples:
                print(f"    ... and {len(fn_list) - max_examples} more")

        if fp_list:
            print(f"\n  {_c(C.BRIGHT_RED, f'{label} - FALSE POSITIVES (Extra): {len(fp_list)}')}")
            for doc_id, item in fp_list[:max_examples]:
                item_short = (item[:50] + "...") if len(item) > 53 else item
                print(f"    - {item_short} [{doc_id[:25]}]")
            if len(fp_list) > max_examples:
                print(f"    ... and {len(fp_list) - max_examples} more")

    if EVAL_ABBREVIATIONS:
        print_errors("ABBREVIATIONS", abbrev_fn, abbrev_fp)
    if EVAL_DISEASES:
        print_errors("DISEASES", disease_fn, disease_fp)
    if EVAL_GENES:
        print_errors("GENES", gene_fn, gene_fp)
    if EVAL_DRUGS:
        print_errors("DRUGS", drug_fn, drug_fp)


def print_final_summary(results: List[DatasetResult]):
    """Print final summary across all datasets."""
    print(f"\n{_c(C.BOLD + C.BRIGHT_CYAN, '=' * 70)}")
    print(f" {_c(C.BOLD + C.BRIGHT_WHITE, 'FINAL EVALUATION SUMMARY')}")
    print(f"{_c(C.BOLD + C.BRIGHT_CYAN, '=' * 70)}")

    # Aggregate across all datasets
    total_abbrev_tp = sum(r.abbrev_tp for r in results)
    total_abbrev_fp = sum(r.abbrev_fp for r in results)
    total_abbrev_fn = sum(r.abbrev_fn for r in results)

    total_disease_tp = sum(r.disease_tp for r in results)
    total_disease_fp = sum(r.disease_fp for r in results)
    total_disease_fn = sum(r.disease_fn for r in results)

    total_gene_tp = sum(r.gene_tp for r in results)
    total_gene_fp = sum(r.gene_fp for r in results)
    total_gene_fn = sum(r.gene_fn for r in results)

    total_drug_tp = sum(r.drug_tp for r in results)
    total_drug_fp = sum(r.drug_fp for r in results)
    total_drug_fn = sum(r.drug_fn for r in results)

    total_docs = sum(r.docs_processed for r in results)
    total_perfect = sum(r.docs_perfect for r in results)

    print(f"\n  Total documents: {total_docs} ({total_perfect} perfect)")
    print()

    def _compute_f1(tp: int, fp: int, fn: int) -> float:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 1.0

    target_met = True

    if EVAL_ABBREVIATIONS and (total_abbrev_tp + total_abbrev_fp + total_abbrev_fn > 0):
        print_entity_metrics("ABBREVIATIONS (Overall)", total_abbrev_tp, total_abbrev_fp, total_abbrev_fn)
        if _compute_f1(total_abbrev_tp, total_abbrev_fp, total_abbrev_fn) < TARGET_ACCURACY:
            target_met = False
        print()

    if EVAL_DISEASES and (total_disease_tp + total_disease_fp + total_disease_fn > 0):
        print_entity_metrics("DISEASES (Overall)", total_disease_tp, total_disease_fp, total_disease_fn)
        if _compute_f1(total_disease_tp, total_disease_fp, total_disease_fn) < TARGET_ACCURACY:
            target_met = False
        print()

    if EVAL_GENES and (total_gene_tp + total_gene_fp + total_gene_fn > 0):
        print_entity_metrics("GENES (Overall)", total_gene_tp, total_gene_fp, total_gene_fn)
        if _compute_f1(total_gene_tp, total_gene_fp, total_gene_fn) < TARGET_ACCURACY:
            target_met = False
        print()

    if EVAL_DRUGS and (total_drug_tp + total_drug_fp + total_drug_fn > 0):
        print_entity_metrics("DRUGS (Overall)", total_drug_tp, total_drug_fp, total_drug_fn)
        if _compute_f1(total_drug_tp, total_drug_fp, total_drug_fn) < TARGET_ACCURACY:
            target_met = False
        print()

    if target_met:
        print(f"  {_c(C.BOLD + C.BRIGHT_GREEN, '████████████████████████████████████████')}")
        print(f"  {_c(C.BOLD + C.BRIGHT_GREEN, '█                                      █')}")
        print(f"  {_c(C.BOLD + C.BRIGHT_GREEN, f'█   ✓ TARGET MET: F1 >= {TARGET_ACCURACY:.0%}            █')}")
        print(f"  {_c(C.BOLD + C.BRIGHT_GREEN, '█                                      █')}")
        print(f"  {_c(C.BOLD + C.BRIGHT_GREEN, '████████████████████████████████████████')}")
    else:
        print(f"  Status: {_c(C.BRIGHT_RED, f'TARGET NOT MET (F1 >= {TARGET_ACCURACY:.0%})')}")

        # Show what needs to be fixed
        if total_abbrev_fn > 0:
            print(f"  {_c(C.BRIGHT_YELLOW, f'Abbreviations: Fix {total_abbrev_fn} missed')}")
        if total_abbrev_fp > 0:
            print(f"  {_c(C.BRIGHT_RED, f'Abbreviations: Remove {total_abbrev_fp} false positives')}")
        if total_disease_fn > 0:
            print(f"  {_c(C.BRIGHT_YELLOW, f'Diseases: Fix {total_disease_fn} missed')}")
        if total_disease_fp > 0:
            print(f"  {_c(C.BRIGHT_RED, f'Diseases: Remove {total_disease_fp} false positives')}")
        if total_gene_fn > 0:
            print(f"  {_c(C.BRIGHT_YELLOW, f'Genes: Fix {total_gene_fn} missed')}")
        if total_gene_fp > 0:
            print(f"  {_c(C.BRIGHT_RED, f'Genes: Remove {total_gene_fp} false positives')}")
        if total_drug_fn > 0:
            print(f"  {_c(C.BRIGHT_YELLOW, f'Drugs: Fix {total_drug_fn} missed')}")
        if total_drug_fp > 0:
            print(f"  {_c(C.BRIGHT_RED, f'Drugs: Remove {total_drug_fp} false positives')}")

    print()


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Run evaluation on all configured datasets."""
    print(f"\n{_c(C.BOLD + C.BRIGHT_CYAN, '=' * 70)}")
    print(f" {_c(C.BOLD + C.BRIGHT_WHITE, 'ENTITY EXTRACTION EVALUATION')}")
    print(f" {_c(C.DIM, f'Target: {TARGET_ACCURACY:.0%} F1 Score')}")
    print(f"{_c(C.BOLD + C.BRIGHT_CYAN, '=' * 70)}")

    # Show configuration
    print("\n  Configuration:")
    print(f"    NLP4RARE:     {'enabled' if RUN_NLP4RARE else 'disabled'}")
    print(f"    Papers:       {'enabled' if RUN_PAPERS else 'disabled'}")
    print(f"    NLM-Gene:     {'enabled' if RUN_NLM_GENE else 'disabled'}")
    print(f"    RareDisGene:  {'enabled' if RUN_RAREDIS_GENE else 'disabled'}")
    print(f"    Max docs:     {MAX_DOCS if MAX_DOCS else 'all'}")
    if RUN_NLP4RARE:
        print(f"    NLP4RARE splits: {', '.join(NLP4RARE_SPLITS)}")
    if RUN_NLM_GENE:
        print(f"    NLM-Gene splits: {', '.join(NLM_GENE_SPLITS)}")
    if RUN_RAREDIS_GENE:
        print(f"    RareDisGene splits: {', '.join(RAREDIS_GENE_SPLITS)}")
    print("\n  Entity types:")
    print(f"    Abbreviations: {'enabled' if EVAL_ABBREVIATIONS else 'disabled'}")
    print(f"    Diseases:      {'enabled' if EVAL_DISEASES else 'disabled'}")
    print(f"    Genes:         {'enabled' if EVAL_GENES else 'disabled'}")
    print(f"    Drugs:         {'enabled' if EVAL_DRUGS else 'disabled'}")

    # Initialize orchestrator once
    print("\n  Initializing orchestrator...")
    orch = create_orchestrator()

    results = []

    # Evaluate NLP4RARE
    if RUN_NLP4RARE and NLP4RARE_PATH.exists():
        gold_data = load_nlp4rare_gold(NLP4RARE_GOLD)
        has_gold = any(gold_data[k] for k in gold_data)
        if has_gold:
            result = evaluate_dataset(
                name="NLP4RARE",
                pdf_folder=NLP4RARE_PATH,
                gold_data=gold_data,
                orch=orch,
                max_docs=MAX_DOCS,
                splits=NLP4RARE_SPLITS,
            )
            print_dataset_summary(result)
            print_error_analysis(result)
            results.append(result)

    # Evaluate Papers
    if RUN_PAPERS and PAPERS_PATH.exists():
        gold_data = load_papers_gold(PAPERS_GOLD)
        has_gold = any(gold_data[k] for k in gold_data)
        if has_gold:
            result = evaluate_dataset(
                name="Papers",
                pdf_folder=PAPERS_PATH,
                gold_data=gold_data,
                orch=orch,
                max_docs=MAX_DOCS,
            )
            print_dataset_summary(result)
            print_error_analysis(result)
            results.append(result)

    # Evaluate NLM-Gene
    if RUN_NLM_GENE and NLM_GENE_PATH.exists():
        gold_data = load_nlm_gene_gold(NLM_GENE_GOLD, splits=NLM_GENE_SPLITS)
        has_gold = any(gold_data[k] for k in gold_data)
        if has_gold:
            result = evaluate_dataset(
                name="NLM-Gene",
                pdf_folder=NLM_GENE_PATH,
                gold_data=gold_data,
                orch=orch,
                max_docs=MAX_DOCS,
                splits=NLM_GENE_SPLITS,
            )
            print_dataset_summary(result)
            print_error_analysis(result)
            results.append(result)

    # Evaluate RareDisGene
    if RUN_RAREDIS_GENE and RAREDIS_GENE_PATH.exists():
        gold_data = load_raredis_gene_gold(RAREDIS_GENE_GOLD, splits=RAREDIS_GENE_SPLITS)
        has_gold = any(gold_data[k] for k in gold_data)
        if has_gold:
            result = evaluate_dataset(
                name="RareDisGene",
                pdf_folder=RAREDIS_GENE_PATH,
                gold_data=gold_data,
                orch=orch,
                max_docs=MAX_DOCS,
                splits=RAREDIS_GENE_SPLITS,
            )
            print_dataset_summary(result)
            print_error_analysis(result)
            results.append(result)

    # Final summary
    if results:
        print_final_summary(results)

    # Exit with error code if any entity type F1 < TARGET_ACCURACY
    def _f1(tp: int, fp: int, fn: int) -> float:
        p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        return 2 * p * r / (p + r) if (p + r) > 0 else 1.0

    total_tp_fp_fn = [
        (sum(r.abbrev_tp for r in results), sum(r.abbrev_fp for r in results), sum(r.abbrev_fn for r in results)),
        (sum(r.disease_tp for r in results), sum(r.disease_fp for r in results), sum(r.disease_fn for r in results)),
        (sum(r.gene_tp for r in results), sum(r.gene_fp for r in results), sum(r.gene_fn for r in results)),
        (sum(r.drug_tp for r in results), sum(r.drug_fp for r in results), sum(r.drug_fn for r in results)),
    ]
    target_met = all(
        _f1(tp, fp, fn) >= TARGET_ACCURACY
        for tp, fp, fn in total_tp_fp_fn
        if (tp + fp + fn) > 0  # skip entity types with no data
    )

    sys.exit(0 if target_met else 1)


if __name__ == "__main__":
    main()


__all__ = [
    "GoldAbbreviation",
    "GoldDisease",
    "GoldDrug",
    "GoldGene",
    "ExtractedAbbreviation",
    "ExtractedDisease",
    "ExtractedDrugEval",
    "ExtractedGene",
    "EntityResult",
    "DocumentResult",
    "DatasetResult",
    "evaluate_dataset",
    "load_nlp4rare_gold",
    "load_papers_gold",
    "load_nlm_gene_gold",
    "load_raredis_gene_gold",
]

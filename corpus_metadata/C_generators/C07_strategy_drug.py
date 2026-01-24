# corpus_metadata/corpus_metadata/C_generators/C07_strategy_drug.py
"""
Drug/chemical entity detection strategy.

Multi-layered approach:
1. Alexion drugs (specialized, highest priority)
2. Investigational drugs (compound IDs + lexicon)
3. FDA approved drugs (brand + generic)
4. RxNorm general terms
5. scispacy NER (CHEMICAL semantic type, fallback)

Uses FlashText for fast keyword matching.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from flashtext import KeywordProcessor

from A_core.A01_domain_models import Coordinate
from A_core.A03_provenance import generate_run_id, get_git_revision_hash
from A_core.A06_drug_models import (
    DrugCandidate,
    DrugFieldType,
    DrugGeneratorType,
    DrugIdentifier,
    DrugProvenanceMetadata,
)
from B_parsing.B01_pdf_to_docgraph import DocumentGraph
from B_parsing.B05_section_detector import SectionDetector
from B_parsing.B06_confidence import ConfidenceCalculator
from B_parsing.B07_negation import NegationDetector

# Optional scispacy import
try:
    import spacy
    from scispacy.linking import EntityLinker  # noqa: F401

    SCISPACY_AVAILABLE = True
except ImportError:
    spacy = None
    SCISPACY_AVAILABLE = False


# -------------------------
# Drug False Positive Filter
# -------------------------


class DrugFalsePositiveFilter:
    """Filter false positive drug matches."""

    # NCT trial ID pattern (clinical trial identifiers, not drugs)
    NCT_PATTERN = re.compile(r"^NCT\d+$", re.IGNORECASE)

    # Ethics committee approval code patterns (not drugs)
    # Matches patterns like KY2022, IRB2023, EC2024, REC2022, IEC2023
    # These are institutional review board/ethics committee approval codes
    ETHICS_CODE_PATTERN = re.compile(
        r"^(?:KY|IRB|EC|REC|IEC|ERB|REB)\d{4}$", re.IGNORECASE
    )

    # Bacteria names (often appear in vaccine trial data, not drugs)
    BACTERIA_ORGANISMS: Set[str] = {
        # Bacteria commonly in vaccine trials
        "neisseria meningitidis",
        "streptococcus pneumoniae",
        "haemophilus influenzae",
        "escherichia coli",
        "e. coli",
        "staphylococcus aureus",
        "s. aureus",
        "clostridium difficile",
        "c. difficile",
        "mycobacterium tuberculosis",
        "bordetella pertussis",
        "salmonella typhi",
        "vibrio cholerae",
        "corynebacterium diphtheriae",
        "shigella",
        "listeria monocytogenes",
        "legionella pneumophila",
        "pseudomonas aeruginosa",
        "klebsiella pneumoniae",
        "enterococcus",
        "helicobacter pylori",
        "h. pylori",
        # Viruses (not drugs)
        "influenza",
        "coronavirus",
        "sars-cov-2",
        "hepatitis",
        "hiv",
        "herpes",
        "measles",
        "mumps",
        "rubella",
        "varicella",
        "rotavirus",
        "polio",
        "rabies",
        "yellow fever",
        "dengue",
        "zika",
        "ebola",
        "rsv",
    }

    # Vaccine-related terms (not drugs per se)
    VACCINE_TERMS: Set[str] = {
        "meningococcal",
        "pneumococcal",
        "conjugate vaccine",
        "polysaccharide vaccine",
        "inactivated vaccine",
        "live attenuated",
        "mrna vaccine",
        "viral vector",
        "subunit vaccine",
        "toxoid",
        "antigen",
        "adjuvant",
        "immunization",
        "vaccination",
        "booster",
        "serogroup",
        "serotype",
    }

    # Credentials and titles (not drugs)
    CREDENTIALS: Set[str] = {
        "md",
        "phd",
        "mph",
        "mbbs",
        "frcp",
        "do",
        "rn",
        "np",
        "pa",
        "pharmd",
        "dnp",
        "dpt",
        "ms",
        "ma",
        "msc",
        "bsc",
        "ba",
    }

    # Biological entities (proteins, complexes, pathways - not drugs)
    BIOLOGICAL_ENTITIES: Set[str] = {
        # Complement system components (biological, not drugs)
        "c3",
        "c4",
        "c5",
        "c3b",
        "c4b",
        "c5b",
        "c3a",
        "c5a",
        "c3 convertase",
        "c5 convertase",
        "membrane attack complex",
        "mac",
        "terminal complement complex",
        "alternative pathway",
        "classical pathway",
        "lectin pathway",
        # Receptors and kinases (biological targets, not drugs)
        "musk",  # Muscle-Specific Kinase - often mislinked to "musk deer"
        "achr",  # Acetylcholine receptor
        "fcrn",  # Neonatal Fc receptor
        "fc receptor",
        "ache",  # Acetylcholinesterase
        "acetylcholinesterase",
        "acetylcholine esterase",  # variant spelling
        "acetylcholine receptor",
        "cholinergic receptors",
        "anti-acetylcholine receptor antibody",  # biomarker, not drug
        # Hormones (biological, not therapeutic drugs unless specific formulation)
        "fsh",  # Follicle Stimulating Hormone
        "follicle stimulating hormone",
        "follicle-stimulating hormone",
        # Immunoglobulins (biological markers)
        "igg",
        "iga",
        "igm",
        "ige",
        "igd",
        "igg1",
        "igg2",
        "igg3",
        "igg4",
        # Cytokines and growth factors (biological)
        "tnf",
        "tnf-alpha",
        "tnf-α",
        "interferon",
        "ifn",
        "ifn-gamma",
        "ifn-γ",
        "interleukin",
        "il-1",
        "il-2",
        "il-6",
        "il-10",
        "tgf-beta",
        "tgf-β",
        "vegf",
        "egf",
        "fgf",
        "pdgf",
        # Enzymes and proteins
        "crp",
        "ldh",
        "ast",
        "alt",
        "alp",
        "ggt",
        "lipase",
        "amylase",
        "transferase",
        "kinase",
        "phosphatase",
        "protease",
        # Cellular components
        "mitochondria",
        "ribosome",
        "endoplasmic reticulum",
        "golgi",
        "nucleus",
        "cytoplasm",
        "membrane",
        "receptor",
        "channel",
        "transporter",
        # Lab markers (not drugs)
        "hemoglobin",
        "hematocrit",
        "platelet",
        "neutrophil",
        "lymphocyte",
        "monocyte",
        "eosinophil",
        "basophil",
        "albumin",
        "globulin",
        "bilirubin",
        "urea",
        "creatinine",
        "potassium",
        "sodium",
        "calcium",
        "magnesium",
        "phosphate",
        "chloride",
        "bicarbonate",
        # Kidney function markers (lab values, not drugs)
        "egfr",  # estimated Glomerular Filtration Rate
        "gfr",   # Glomerular Filtration Rate
        "upcr",  # Urine Protein-Creatinine Ratio
        "uacr",  # Urine Albumin-Creatinine Ratio
        "proteinuria",
        "microalbuminuria",
        "macroalbuminuria",
    }

    # Pharmaceutical company names (not drugs)
    # These often get incorrectly matched to drug entities by NER models
    PHARMA_COMPANY_NAMES: Set[str] = {
        # Major pharma companies
        "novartis",
        "pfizer",
        "roche",
        "merck",
        "johnson",
        "johnson & johnson",
        "astrazeneca",
        "sanofi",
        "gsk",
        "glaxosmithkline",
        "abbvie",
        "amgen",
        "gilead",
        "eli lilly",
        "lilly",
        "bristol-myers squibb",
        "bms",
        "bayer",
        "boehringer ingelheim",
        "takeda",
        "novo nordisk",
        "regeneron",
        "biogen",
        "vertex",
        "moderna",
        # Specialty/biotech companies often in clinical trials
        "alexion",
        "biocryst",
        "biocryst pharmaceuticals",
        "chemocentryx",
        "achillion",
        "achillion pharmaceuticals",
        "catalyst biosciences",
        "catalyst",
        "gyroscope therapeutics",
        "gyroscope",
        "silence therapeutics",
        # Other common false positives
        "pharmaceuticals",
        "therapeutics",
        "biosciences",
        "biopharma",
    }

    # Common words that might match drug names
    COMMON_WORDS: Set[str] = {
        # Dosage forms and containers
        "oral",
        "tablet",
        "capsule",
        "injection",
        "solution",
        "cream",
        "gel",
        "patch",
        "spray",
        "drops",
        "vial",
        "vials",
        "ampoule",
        "ampule",
        "syringe",
        "kit",
        # Generic substances
        "water",
        "salt",
        "acid",
        "base",
        "oil",
        "sugar",
        "fat",
        # Metals (often false positives)
        "iron",
        "gold",
        "silver",
        "lead",
        "zinc",
        # Drug-related terms
        "dose",
        "drug",
        "agent",
        "compound",
        "product",
        "formula",
        "active",
        "inactive",
        "medications",
        "medication",
        "medicine",
        "medicines",
        "treatment",
        "therapy",
        # Generic/vague terms
        "various",
        "other",
        "others",
        "complete",
        "unknown",
        "none",
        "same",
        "different",
        "several",
        "many",
        "some",
        "all",
        "any",
        # Common English words that appear in RxNorm (false positives)
        "boston",
        "date",
        "dates",
        "prevail",
        "bronchial",
        "purpose",
        "vital",
        "tbc",
        "tbd",
        "maintain",
        "program",
        "perform",
        "impact",
        "basis",
        "barrier",
        "deliver",
        "syringe",
        "needle",
        "tape",
        "mask",
        "matrix",
        "regain",
        "supply",
        "correct",
        "corrective",
        "throat",
        "schirmer",
        "stop",
        "align",
        "fig",
        "val",
        "sinus",
        "excel",
        "nasal",
        "pediatric",
        "tums",
        "induction",
        # Medical equipment/supplies (not drugs)
        "pregnancy test",
        "hbsag",
        "hbs",
        # Journal/publication names
        "lancet",
        "nature",
        "science",
        "cell",
        # Geographic/organization names (not drugs)
        "turkey",
        "genesis",
        "urban",
        # Generic medical terms
        "promote",
        "cartilage",
        "bone marrow",
        "lymphocytes",
        "no data",
        "arthritis foundation",
        # Antibody types (markers, not drugs unless specific)
        "anca",
        # Generic process/action words
        "via",
        "met",
        "food",
        "duration",
        "support",
        "root",
        "process",
        "his",
        "central",
        "ensure",
        "blockade",
        "therapeutic",
        "monotherapy",
        "targeted therapy",
        "soc",
        # Common English words that happen to be drug brand names
        "today",  # TODAY is a brand for nonoxynol-9 contraceptive
        "choice",  # Various brand names
        "plan",  # Plan B
        "clear",
        "complete",
        "simple",
        "natural",
        "daily",
        "one",
        "first",
        "total",
        # Disease abbreviations that are also in RxNorm (prefer disease meaning)
        "pah",  # Pulmonary Arterial Hypertension (not Phenylalanine)
        "als",  # Amyotrophic Lateral Sclerosis (not Aluminum Salt)
        "ms",   # Multiple Sclerosis (not Morphine Sulfate in most contexts)
        "hiv",  # Human Immunodeficiency Virus
        # Biomolecules (not drugs per se)
        "protein",
        "creatinine",
        "glucose",
        "angiotensin",
        "aldosterone",
        "renin",
        "renin–",
        "complement",
        "factor b",
        "serum",
        # Too generic drug terms
        "inhibitor",
        "inhibitors",
        "antagonist",
        "antagonists",
        "agonist",
        "agonists",
        "receptor",
        "receptors",
        "pharmaceutical preparations",
        "activation product",
        # Anatomical/biological structures
        "nephron",
        "membrane attack complex",
        "com",
        "importal",
        # Animals and biological terms
        "animals",
        "animal",
        "mice",
        "mouse",
        "rat",
        "rats",
        "dog",
        "dogs",
        "monkey",
        "monkeys",
        "rabbit",
        "rabbits",
        # Medical symptoms/processes (not drugs)
        "constriction",
        "dilation",
        "stenosis",
        "obstruction",
        "occlusion",
        "inflammation",
        "infection",
        "hemorrhage",
        "bleeding",
        # Common first names that appear in press releases (not drugs)
        "julie",
        "peter",
        "john",
        "david",
        "michael",
        "james",
        "robert",
        "william",
        "richard",
        "thomas",
        "mary",
        "patricia",
        "jennifer",
        "linda",
        "elizabeth",
        "barbara",
        "susan",
        "sarah",
        "karen",
        "nancy",
        "nikki",
        "ayn",
        # Generic medical/scientific terms (not specific drugs)
        "vaccines",  # Generic term, not a specific drug
        "vaccine",
        "protection",  # Common word, not Protectin
        "protections",
        # Disease/clinical abbreviations wrongly matched to drugs
        "ctd",  # Connective Tissue Diseases (not a drug)
        # WHO Functional Class designations (clinical classification, not drugs)
        "fc",
        "fc i",
        "fc ii",
        "fc iii",
        "fc iv",
        # Common statistical abbreviations that get misread
        "cls",  # Often misread from "CIs" (confidence intervals)
        "cis",  # Confidence intervals
        # Common measurement abbreviations
        "mm",   # millimeters (credential or unit, not drug)
        "cm",   # centimeters
        "kg",   # kilograms
        "mg",   # milligrams (unit, not drug name)
        "ml",   # milliliters
    }

    # Organizations and agencies (not drugs)
    ORGANIZATIONS: Set[str] = {
        "medicines agency",
        "european medicines agency",
        "ema",
        "fda",
        "food and drug administration",
        "world health organization",
        "who",
        "nih",
        "national institutes of health",
        "cdc",
        "centers for disease control",
    }

    # Body parts and organs (not drugs)
    BODY_PARTS: Set[str] = {
        "liver",
        "kidney",
        "heart",
        "lung",
        "brain",
        "blood",
        "bone",
        "skin",
        "muscle",
        "nerve",
        "eye",
        "ear",
        "stomach",
        "intestine",
        "colon",
        "bladder",
        "spleen",
        "pancreas",
        "thyroid",
        "adrenal",
        "ovary",
        "uterus",
        "prostate",
        "breast",
        "tongue",
        "teeth",
        "gum",
        "nail",
        "hair",
    }

    # Clinical trial status terms (leaked from trial data)
    TRIAL_STATUS_TERMS: Set[str] = {
        "not_yet_recruiting",
        "recruiting",
        "active",
        "completed",
        "suspended",
        "terminated",
        "withdrawn",
        "enrolling",
        "available",
        "approved",
        "no_longer_available",
        "withheld",
        "unknown",
        "not yet recruiting",
        "active, not recruiting",
        "enrolling by invitation",
    }

    # Medical equipment/procedures (not drugs)
    EQUIPMENT_PROCEDURES: Set[str] = {
        "ultrasound",
        "mri",
        "ct",
        "xray",
        "x-ray",
        "scan",
        "surgery",
        "biopsy",
        "endoscopy",
        "catheter",
        "stent",
        "implant",
        "pacemaker",
        "ventilator",
        "dialysis",
        "ecg",
        "ekg",
        "eeg",
        "emg",
    }

    # Terms that should ALWAYS be filtered, even from specialized lexicons
    # These are generic placeholders that sometimes appear in trial data
    ALWAYS_FILTER: Set[str] = {
        "medications",
        "medication",
        "other",
        "others",
        "placebo",
        "control",
        "standard of care",
        "standard care",
        "best supportive care",
        "usual care",
        "no intervention",
        "observation",
        "watchful waiting",
        "dietary supplement",
        "behavioral",
        "device",
        "procedure",
        "radiation",
        "biological",
        "combination product",
        "diagnostic test",
        "genetic",
        "various",
        "multiple",
        "unspecified",
        "investigational",
        "experimental",
        "study drug",
        "study treatment",
        "test drug",
        "test product",
        "active comparator",
        "sham comparator",
    }

    # NER-specific false positives (commonly returned by scispacy UMLS)
    # These are drug classes, biological entities, or generic terms - not specific drugs
    NER_FALSE_POSITIVES: Set[str] = {
        # Generic pharmaceutical terms
        "pharmaceutical preparations",
        "pharmaceutical",
        "pharmaceuticals",
        "drug",
        "drugs",
        "medicine",
        "medicines",
        "therapeutic",
        "therapeutics",
        "medicament",
        "medicaments",
        "preparation",
        "preparations",
        "formulation",
        "formulations",
        "compound",
        "compounds",
        "agent",
        "agents",
        "substance",
        "substances",
        # Drug classes (not specific drugs)
        "angiotensin-converting enzyme inhibitors",
        "ace inhibitors",
        "beta blockers",
        "beta-blockers",
        "calcium channel blockers",
        "diuretics",
        "statins",
        "nsaids",
        "antibiotics",
        "antivirals",
        "antifungals",
        "analgesics",
        "antipyretics",
        "anticoagulants",
        "antihistamines",
        "antidepressants",
        "antipsychotics",
        "anxiolytics",
        "sedatives",
        "hypnotics",
        "stimulants",
        "opioids",
        "opiates",
        "barbiturates",
        "benzodiazepines",
        "corticosteroid",  # class, not specific drug
        "glucocorticoid",
        "glucocorticoids",
        "mineralocorticoid",
        "mineralocorticoids",
        "immunosuppressive agents",
        "immunosuppressants",
        "immunomodulators",
        "chemotherapeutic agents",
        "chemotherapy",
        "cytotoxic agents",
        "targeted therapy",
        "monotherapy",
        "combination therapy",
        "first-line therapy",
        "second-line therapy",
        "salvage therapy",
        "maintenance therapy",
        "induction therapy",
        "consolidation therapy",
        # Biological entities (NOT drugs)
        "complement system proteins",
        "complement proteins",
        "complement inactivating agents",
        "complement factor b",
        "complement c3 convertases",
        "complement c5 convertases",
        "complement component",
        "complement inhibition",
        # Lab reagents (not therapeutic drugs)
        "sds",
        "sodium dodecyl sulfate",
        "anaphylatoxins",
        "membrane attack complex",
        "serum proteins",
        "plasma proteins",
        "blood proteins",
        "growth factors",
        "cytokines",
        "chemokines",
        "interleukins",
        "interferons",
        "tumor necrosis factor",
        "receptors",
        "receptor",
        "transporter",
        "transporters",
        "channel",
        "channels",
        "enzyme",
        "enzymes",
        "kinase",
        "kinases",
        "phosphatase",
        "phosphatases",
        "protease",
        "proteases",
        "ligand",
        "ligands",
        "substrate",
        "substrates",
        "cofactor",
        "cofactors",
        "coenzyme",
        "coenzymes",
        "antigen",
        "antigens",
        "antibody",
        "antibodies",
        "immunoglobulin",
        "immunoglobulins",
        "immunoglobulin g",  # biomarker, not drug
        "immunoglobulin a",
        "immunoglobulin m",
        "immunoglobulin e",
        # Specific NER false positives observed
        "importal",  # Laxative, not relevant to trials
        "antagon",  # False match
        "sc5b-9 protein complex",
        "serum, horse",
        "horse serum",
        "serum",
        "plasma",
        "blood",
        "tissue",
        "cell",
        "cells",
        "protein",
        "proteins",
        "peptide",
        "peptides",
        "amino acid",
        "amino acids",
        "nucleotide",
        "nucleotides",
        "lipid",
        "lipids",
        "carbohydrate",
        "carbohydrates",
        "activation product",
        "activation products",
        "degradation product",
        "degradation products",
        "metabolite",
        "metabolites",
        "soc",  # Standard of Care
        # Hormone-related (biological, not drugs unless specific)
        "hormone",
        "hormones",
        "adrenal cortex hormones",
        "steroid hormones",
        "thyroid hormones",
        "pituitary hormones",
        "fsh",
        "follicle stimulating hormone",
        "follicle-stimulating hormone",
        "human follicle-stimulating hormone",
        "renin",
        "aldosterone",
        "angiotensin",
        "epinephrine",
        "norepinephrine",
        "dopamine",
        "serotonin",
        "histamine",
        "acetylcholine",
        "acetylcholine esterase",
        "anti-acetylcholine receptor antibody",
        # Receptor types (biological targets, not drugs)
        "receptors, corticosteroid",
        "serotonin antagonists",
        "serotonin receptors",
        "dopamine receptors",
        "histamine receptors",
        "adrenergic receptors",
        "cholinergic receptors",
        "opioid receptors",
        "epidermal growth factor receptor",
        "vascular endothelial growth factor",
        "platelet-derived growth factor",
        "fibroblast growth factor",
        "insulin-like growth factor",
        "transforming growth factor",
        # Inhibitor classes (not specific drugs)
        "ras inhibitor",
        "ras inhibitors",
        "factor xa inhibitors",
        "thrombin inhibitors",
        "protease inhibitors",
        "kinase inhibitors",
        "mtor inhibitors",
        "jak inhibitors",
        "tyrosine kinase inhibitors",
        "hdac inhibitors",
        "parp inhibitors",
        # Transporter types
        "sodium-glucose transporter 1",
        "sodium-glucose transporter 2",
        "glucose transporters",
        "ion channels",
        "calcium channels",
        "sodium channels",
        "potassium channels",
        # Other biological
        "ccrl1 protein, human",
        "ccrl1 protein",
        "human protein",
        # Observed NER false positives
        "colicin plasmids",
        "wright stain",
        "rheumatoid factor",
        "dbl oncoprotein",
        "prevent (product)",
        "prevent product",
        "glycophorin a",
        "nervous system involved sulfotransferase",
        "estrogens, non-steroidal",
        "musk secretion from musk deer",
        "ldl-receptor related protein 1",
        "glycation end products, advanced",
        "stop brand of fluoride",
        "gastrointestinal agents",
        # Drugs of abuse / controlled substances (not trial drugs unless specific formulation)
        "cocaine",
        "cocaine (substance)",
        "heroin",
        "methamphetamine",
        "cannabis",
        "marijuana",
        "morphine",  # Unless specific formulation
        "fentanyl",  # Unless specific formulation
        # Muscle-specific kinase variants (often confused with animal product)
        "musk",
        "musk secretion",
        "musk deer",
        # Additional false positives from clinical trial protocols
        "immune complex",
        "fc receptor",
        "autoantibodies",
        "biological factors",
        "investigational new drugs",
        "antibodies, anti-idiotypic",
        # Additional NER false positives observed in Iptacopan trial
        "sodium-glucose transporter 1",
        "sodium-glucose cotransporter-2",
        "at1 receptor blockers",
        "c5a anaphylatoxins",
        "c3 activation products",
        "normal serum c3",
        "complement bio",
        "medicines agency",
        # Generic amino acids (not drugs)
        "alanine",
        "tyrosine",
        "proline",
        "histidine",
        "l-histidine",
        "glycine",
        "aspartate",
        "serine",
        "threonine",
        "leucine",
        "isoleucine",
        "valine",
        "phenylalanine",
        "tryptophan",
        "methionine",
        "cysteine",
        "asparagine",
        "glutamine",
        "glutamate",
        "arginine",
        "lysine",
        # Biochemical components / excipients
        "sucrose",
        "polysorbate 80",
        "polysorbates",
        "hydrochloride",
        "inosine monophosphate",
        "trientine",  # Chelating agent, often false positive
        # Additional NER false positives observed in myasthenia gravis protocols
        "alanine transaminase",
        "complement component c1",
        "c1s protein, human",
        "c1s protein",
        "human follicle-stimulating hormone",
        "ldl-receptor related protein 1",
        "ldl receptor",
        # Statistical/ML terms commonly misidentified as drugs
        "lasso",
        "metric",
        "metric (substance)",
        "metrics",
        "correlation",
        "coefficient",
        "regression",
        "classifier",
        "model",
        "algorithm",
        "variance",
        "covariance",
        "standard deviation",
        "confidence interval",
        "hazard ratio",
        "odds ratio",
        # Author name patterns that get mismatched to drugs
        "fluorouracil",  # Often matched to "Fu Y" (author name)
        "antigens, cd15",  # Often matched to "Lewis JH" (author name)
        # Equipment/scanner brand names
        "revolution",
        "somatom",
        "aquilion",
    }

    # Generic all-caps words that are not drugs
    NON_DRUG_ALLCAPS: Set[str] = {
        "information",
        "complete",
        "ring",
        "same",
        "other",
        "none",
        "all",
        "any",
        "new",
        "old",
        "high",
        "low",
        "full",
        "empty",
        "open",
        "closed",
        "start",
        "end",
        "first",
        "last",
        "next",
        "previous",
        "current",
        "total",
        "average",
        "mean",
        "median",
        "normal",
        "abnormal",
        "positive",
        "negative",
        "present",
        "absent",
        "available",
        "unavailable",
        "required",
        "optional",
        "primary",
        "secondary",
        "additional",
    }

    # Minimum drug name length
    MIN_LENGTH = 3

    # Pattern suffixes that indicate biological entities, not specific drugs
    # These catch "X protein", "X receptor", "X inhibitor", etc.
    BIOLOGICAL_SUFFIXES = [
        " protein",
        " proteins",
        " receptor",
        " receptors",
        " inhibitor",
        " inhibitors",
        " antagonist",
        " antagonists",
        " agonist",
        " agonists",
        " blocker",
        " blockers",
        " transporter",
        " transporters",
        " channel",
        " channels",
        " enzyme",
        " enzymes",
        " kinase",
        " kinases",
        " factor",
        " factors",
        " complex",
        " pathway",
        " system",
        " agents",
        " products",
        " product",
        ", human",
        ", mouse",
        ", rat",
    ]

    # Substring patterns that indicate false positives
    # If matched_text contains any of these, it's likely a false positive
    FP_SUBSTRINGS = [
        "serum c3",
        "serum c4",
        "serum c5",
        "serum igg",
        "serum iga",
        "serum igm",
        "complement bio",
        "complement system",
        "activation product",
        "normal serum",
        "plasma c3",
        "medicines agency",
        "pharmaceutical",
        "anaphylatoxin",
        "sc5b-9",
        "sc5b9",
        "c5b-9",
        "c5b9",
        # Animal/natural product false positives
        "musk deer",
        "musk secretion",
        "brand of fluoride",
        # Oncoprotein/genetic false positives
        "oncoprotein",
        "plasmid",
        "sulfotransferase",
        # Biological markers
        "end products, advanced",
        "glycation end",
    ]

    def __init__(self):
        self.common_words_lower = {w.lower() for w in self.COMMON_WORDS}
        self.body_parts_lower = {w.lower() for w in self.BODY_PARTS}
        self.trial_status_lower = {w.lower() for w in self.TRIAL_STATUS_TERMS}
        self.equipment_lower = {w.lower() for w in self.EQUIPMENT_PROCEDURES}
        self.non_drug_allcaps_lower = {w.lower() for w in self.NON_DRUG_ALLCAPS}
        self.always_filter_lower = {w.lower() for w in self.ALWAYS_FILTER}
        self.bacteria_organisms_lower = {w.lower() for w in self.BACTERIA_ORGANISMS}
        self.vaccine_terms_lower = {w.lower() for w in self.VACCINE_TERMS}
        self.credentials_lower = {w.lower() for w in self.CREDENTIALS}
        self.biological_entities_lower = {w.lower() for w in self.BIOLOGICAL_ENTITIES}
        self.ner_false_positives_lower = {w.lower() for w in self.NER_FALSE_POSITIVES}
        self.pharma_company_names_lower = {w.lower() for w in self.PHARMA_COMPANY_NAMES}
        self.organizations_lower = {w.lower() for w in self.ORGANIZATIONS}
        self.fp_substrings_lower = [s.lower() for s in self.FP_SUBSTRINGS]

        # Load gene symbols from lexicon (genes are not drugs)
        self.gene_symbols_lower: Set[str] = set()
        self._load_gene_lexicon()

    def _load_gene_lexicon(self) -> None:
        """Load gene symbols to filter from drug matches."""
        gene_path = Path("ouput_datasources/2025_08_orphadata_genes.json")
        if not gene_path.exists():
            return
        try:
            with open(gene_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Only load primary gene symbols (not aliases) for drug filtering
            # Aliases might be too aggressive (some drugs share names with gene aliases)
            for entry in data:
                if entry.get("source") == "orphadata_hgnc":
                    symbol = entry.get("term", "").lower()
                    if symbol and len(symbol) >= 3:  # Skip very short symbols
                        self.gene_symbols_lower.add(symbol)
        except Exception:
            pass  # Silently fail - gene filtering is optional enhancement

    def is_false_positive(
        self, matched_text: str, context: str, generator_type: DrugGeneratorType
    ) -> bool:
        """
        Check if a drug match is likely a false positive.

        Returns True if the match should be filtered out.
        """
        text_lower = matched_text.lower().strip()
        text_stripped = matched_text.strip()

        # Skip very short matches
        if len(text_lower) < self.MIN_LENGTH:
            return True

        # Filter NCT trial IDs (e.g., NCT04817618) - these are trial identifiers, not drugs
        if self.NCT_PATTERN.match(text_stripped):
            return True

        # Filter ethics committee approval codes (e.g., KY2022, IRB2023)
        # These are institutional codes, not drug compound IDs
        if self.ETHICS_CODE_PATTERN.match(text_stripped):
            return True

        # Always filter generic placeholder terms (even from specialized lexicons)
        if text_lower in self.always_filter_lower:
            return True

        # Always filter trial status terms (even from specialized lexicons)
        if text_lower in self.trial_status_lower:
            return True

        # Check if text contains any trial status term (handles various formats)
        text_normalized = text_lower.replace("_", " ")
        for status in self.trial_status_lower:
            if status in text_normalized:
                return True

        # Filter text containing trial status in parentheses like "Medications (NOT_YET_RECRUITING)"
        # or "Iptacopan (RECRUITING)"
        if "(" in text_stripped and ")" in text_stripped:
            paren_content = text_stripped[text_stripped.find("(")+1:text_stripped.find(")")]
            paren_lower = paren_content.lower().replace("_", " ")
            # Check if parentheses contain trial status
            for status in self.trial_status_lower:
                if status in paren_lower:
                    return True
            # Also check the base word before parentheses
            base_word = text_stripped[:text_stripped.find("(")].strip().lower()
            if base_word in self.common_words_lower:
                return True

        # Always filter bacteria/organism names (not drugs)
        if text_lower in self.bacteria_organisms_lower:
            return True

        # Check for partial bacteria matches (e.g., "Neisseria meningitidis serogroup B")
        for organism in self.bacteria_organisms_lower:
            if organism in text_lower:
                return True

        # Always filter vaccine-related terms (not drugs per se)
        if text_lower in self.vaccine_terms_lower:
            return True

        # Always filter credentials (MD, PhD, MPH, etc.)
        if text_lower in self.credentials_lower:
            return True

        # Always filter biological entities (proteins, enzymes, markers - not drugs)
        if text_lower in self.biological_entities_lower:
            return True

        # Always filter pharmaceutical company names (not drugs)
        # Check both exact match and partial match (e.g., "Novartis Pharma" contains "novartis")
        if text_lower in self.pharma_company_names_lower:
            return True
        for company in self.pharma_company_names_lower:
            if company in text_lower or text_lower in company:
                return True

        # Always filter organizations and agencies
        if text_lower in self.organizations_lower:
            return True
        for org in self.organizations_lower:
            if org in text_lower:
                return True

        # Filter NER-specific false positives (drug classes, generic terms, biological entities)
        # These are commonly returned by scispacy UMLS but are not actual drugs
        if text_lower in self.ner_false_positives_lower:
            return True

        # Substring-based filtering for known false positive patterns
        # Catches things like "normal serum C3", "complement bio", "activation products"
        for fp_substr in self.fp_substrings_lower:
            if fp_substr in text_lower:
                return True

        # Pattern-based filtering for NER results
        # Catches "X protein", "X receptor", "X inhibitors", "X blockers", etc.
        if generator_type == DrugGeneratorType.SCISPACY_NER:
            for suffix in self.BIOLOGICAL_SUFFIXES:
                if text_lower.endswith(suffix):
                    return True

        # Always filter body parts
        if text_lower in self.body_parts_lower:
            return True

        # Always filter equipment/procedures
        if text_lower in self.equipment_lower:
            return True

        # Filter gene symbols (genes are not drugs)
        # Only for NER and general lexicons - specialized lexicons may have valid overlaps
        if generator_type in {
            DrugGeneratorType.SCISPACY_NER,
            DrugGeneratorType.LEXICON_RXNORM,
            DrugGeneratorType.LEXICON_FDA,
        }:
            if text_lower in self.gene_symbols_lower:
                return True

        # Skip common words (unless from specialized lexicon)
        if generator_type not in {
            DrugGeneratorType.LEXICON_ALEXION,
            DrugGeneratorType.LEXICON_INVESTIGATIONAL,
        }:
            if text_lower in self.common_words_lower:
                return True

            # Filter generic all-caps words that aren't drugs
            if text_lower in self.non_drug_allcaps_lower:
                return True

        # Context-based author name detection
        # Patterns that indicate the matched text is likely an author name, not a drug
        if context:
            ctx_lower = context.lower()
            # Author list patterns: "Name1 A, Name2 B, Name3 C" or "by Name,"
            # Check if matched text appears in author-like context
            author_indicators = [
                f"by {text_lower},",
                f"by {text_lower}.",
                f"by {text_lower} ",
                f"{text_lower} et al",
                f", {text_lower},",  # In author list
                f", {text_lower}.",
            ]
            for indicator in author_indicators:
                if indicator in ctx_lower:
                    return True

            # Check for author initials pattern: "Name AB" where AB are initials
            # Pattern: matched_text followed by space and 1-2 capital letters
            import re
            author_pattern = re.compile(
                rf"\b{re.escape(text_stripped)}\s+[A-Z]{{1,2}}[,\.\s]",
                re.IGNORECASE
            )
            if author_pattern.search(context):
                return True

        return False


# -------------------------
# Drug Detector
# -------------------------


class DrugDetector:
    """
    Multi-layered drug mention detection.

    Layers (in priority order):
    1. Alexion drugs (specialized, auto-validated)
    2. Investigational drugs (compound IDs + lexicon)
    3. FDA approved drugs
    4. RxNorm general terms
    5. scispacy NER (fallback)
    """

    # Terms to skip when loading lexicons (prevent indexing obvious false positives)
    # These are filtered at load time for efficiency - no runtime overhead
    LEXICON_LOAD_BLACKLIST: Set[str] = {
        # Country names (often appear as study sites)
        "turkey", "china", "india", "jordan", "guinea", "chile", "mali",
        "niger", "chad", "togo", "peru", "cuba", "iran", "iraq", "oman",
        # Common English words that appear in RxNorm
        "genesis", "urban", "promote", "vital", "complete", "balance",
        "comfort", "relief", "choice", "nature", "natural", "pure",
        "basic", "simple", "clear", "fresh", "bright", "calm", "gentle",
        "precise", "dimension", "filter", "essence", "revolution", "metric",
        "metrics", "optimal", "standard", "ideal", "normal", "advanced",
        "select", "prime", "premier", "elite", "ultra", "supreme", "max",
        # Statistical/ML terms (not drugs)
        "lasso", "mcc", "correlation", "coefficient", "regression",
        # Equipment/brand names (not drugs)
        "siemens", "philips", "toshiba", "aquilion", "somatom",
        # Biological terms (not drugs)
        "cartilage", "bone marrow", "lymphocytes", "plasma", "serum",
        # Organizations/misc
        "arthritis foundation", "no data", "not available", "unknown",
        # Antibody markers (diagnostic, not therapeutic)
        "anca",
        # Common author surnames that match drug names
        "cai", "lewis", "sun", "chen", "wang", "li", "zhang", "liu",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.run_id = str(self.config.get("run_id") or generate_run_id("DRUG"))
        self.pipeline_version = (
            self.config.get("pipeline_version") or get_git_revision_hash()
        )
        self.doc_fingerprint_default = (
            self.config.get("doc_fingerprint") or "unknown-doc-fingerprint"
        )

        # Context window for evidence extraction
        self.context_window = int(self.config.get("context_window", 300))

        # Shared parsing utilities from B_parsing
        self.section_detector = SectionDetector()
        self.negation_detector = NegationDetector()
        self.confidence_calculator = ConfidenceCalculator()

        # Lexicon base path
        self.lexicon_base_path = Path(
            self.config.get("lexicon_base_path", "ouput_datasources")
        )

        # Initialize FlashText processors
        self.alexion_processor: Optional[KeywordProcessor] = None
        self.investigational_processor: Optional[KeywordProcessor] = None
        self.fda_processor: Optional[KeywordProcessor] = None
        self.rxnorm_processor: Optional[KeywordProcessor] = None

        # Drug metadata dictionaries
        self.alexion_drugs: Dict[str, Dict] = {}
        self.investigational_drugs: Dict[str, Dict] = {}
        self.fda_drugs: Dict[str, Dict] = {}
        self.rxnorm_drugs: Dict[str, Dict] = {}

        # Lexicon loading stats (for summary output)
        self._lexicon_stats: List[Tuple[str, int, str]] = []

        # Load lexicons
        self._load_lexicons()

        # Compound ID patterns for investigational drugs
        self.compound_patterns = [
            re.compile(r"\b([A-Z]{2,4})[-]?(\d{3,6})\b"),  # LNP023, BMS-986278
            re.compile(r"\b([A-Z]{2,4})[-]?([A-Z]?\d{4,})\b"),  # ABT199, GS-9973
            re.compile(r"\b(ALXN\d{3,6})\b", re.IGNORECASE),  # ALXN1720
        ]

        # False positive filter
        self.fp_filter = DrugFalsePositiveFilter()

        # scispacy NER model
        self.nlp = None
        if SCISPACY_AVAILABLE:
            self._init_scispacy()

    def _load_lexicons(self) -> None:
        """Load all drug lexicons."""
        self._load_alexion_lexicon()
        self._load_investigational_lexicon()
        self._load_fda_lexicon()
        self._load_rxnorm_lexicon()

    def _load_alexion_lexicon(self) -> None:
        """Load Alexion specialized drug lexicon."""
        path = self.lexicon_base_path / "2025_08_alexion_drugs.json"
        if not path.exists():
            print(f"[WARN] Alexion lexicon not found: {path}")
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            alexion_proc = KeywordProcessor(case_sensitive=False)
            self.alexion_processor = alexion_proc
            known_drugs = data.get("known_drugs", {})
            drug_types = data.get("drug_types", {})

            for drug_name, drug_info in known_drugs.items():
                # Store metadata
                self.alexion_drugs[drug_name.lower()] = {
                    "preferred_name": drug_name,
                    "info": drug_info,
                    "drug_type": drug_types.get(drug_name, {}),
                    "source": "alexion",
                }
                # Add to FlashText
                alexion_proc.add_keyword(drug_name, drug_name.lower())

                # Add variations (brand names, compound IDs if available)
                if isinstance(drug_info, dict):
                    for alias in drug_info.get("aliases", []):
                        alexion_proc.add_keyword(alias, drug_name.lower())

            self._lexicon_stats.append(
                ("Alexion drugs", len(known_drugs), "2025_08_alexion_drugs.json")
            )

        except Exception as e:
            print(f"[WARN] Failed to load Alexion lexicon: {e}")

    def _load_investigational_lexicon(self) -> None:
        """Load investigational drugs from ClinicalTrials.gov data."""
        path = self.lexicon_base_path / "2025_08_investigational_drugs.json"
        if not path.exists():
            print(f"[WARN] Investigational lexicon not found: {path}")
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            inv_proc = KeywordProcessor(case_sensitive=False)
            self.investigational_processor = inv_proc
            count = 0

            skipped = 0
            for entry in data:
                drug_name = entry.get("interventionName", "").strip()
                if not drug_name or len(drug_name) < 3:
                    continue

                # Skip non-drug interventions
                if entry.get("interventionType") != "DRUG":
                    continue

                drug_key = drug_name.lower()

                # Skip blacklisted terms at load time (efficiency)
                if drug_key in self.LEXICON_LOAD_BLACKLIST:
                    skipped += 1
                    continue

                if drug_key not in self.investigational_drugs:
                    self.investigational_drugs[drug_key] = {
                        "preferred_name": drug_name,
                        "nct_id": entry.get("nctId"),
                        "conditions": entry.get("conditions", []),
                        "status": entry.get("overallStatus"),
                        "title": entry.get("title"),
                        "source": "investigational",
                    }
                    inv_proc.add_keyword(drug_name, drug_key)
                    count += 1

            if skipped > 0:
                print(f"    [INFO] Skipped {skipped} blacklisted investigational terms")
            self._lexicon_stats.append(
                ("Investigational drugs", count, "2025_08_investigational_drugs.json")
            )

        except Exception as e:
            print(f"[WARN] Failed to load investigational lexicon: {e}")

    def _load_fda_lexicon(self) -> None:
        """Load FDA approved drugs lexicon."""
        path = self.lexicon_base_path / "2025_08_fda_approved_drugs.json"
        if not path.exists():
            print(f"[WARN] FDA lexicon not found: {path}")
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            fda_proc = KeywordProcessor(case_sensitive=False)
            self.fda_processor = fda_proc
            count = 0
            skipped = 0

            for entry in data:
                drug_name = entry.get("key", "").strip()
                if not drug_name or len(drug_name) < 3:
                    continue

                drug_key = drug_name.lower()

                # Skip blacklisted terms at load time (efficiency)
                if drug_key in self.LEXICON_LOAD_BLACKLIST:
                    skipped += 1
                    continue

                meta = entry.get("meta", {})

                if drug_key not in self.fda_drugs:
                    self.fda_drugs[drug_key] = {
                        "preferred_name": drug_name,
                        "brand_name": meta.get("brand_name"),
                        "drug_class": entry.get("drug_class"),
                        "dosage_form": meta.get("dosage_form"),
                        "route": meta.get("route"),
                        "marketing_status": meta.get("marketing_status"),
                        "application_number": meta.get("application_number"),
                        "source": "fda",
                    }
                    fda_proc.add_keyword(drug_name, drug_key)
                    count += 1

                    # Also add brand name if different
                    brand = meta.get("brand_name", "")
                    if brand and brand.lower() != drug_key:
                        brand_key = brand.lower()
                        if brand_key not in self.fda_drugs:
                            fda_proc.add_keyword(brand, drug_key)

            if skipped > 0:
                print(f"    [INFO] Skipped {skipped} blacklisted FDA terms")
            self._lexicon_stats.append(
                ("FDA approved drugs", count, "2025_08_fda_approved_drugs.json")
            )

        except Exception as e:
            print(f"[WARN] Failed to load FDA lexicon: {e}")

    def _load_rxnorm_lexicon(self) -> None:
        """Load RxNorm general drug lexicon."""
        path = self.lexicon_base_path / "2025_08_lexicon_drug.json"
        if not path.exists():
            print(f"[WARN] RxNorm lexicon not found: {path}")
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            rxnorm_proc = KeywordProcessor(case_sensitive=False)
            self.rxnorm_processor = rxnorm_proc
            count = 0

            skipped = 0
            for entry in data:
                term = entry.get("term", "").strip()
                if not term or len(term) < 3:
                    continue

                term_key = term.lower()

                # Skip blacklisted terms at load time (efficiency)
                if term_key in self.LEXICON_LOAD_BLACKLIST:
                    skipped += 1
                    continue

                if term_key not in self.rxnorm_drugs:
                    self.rxnorm_drugs[term_key] = {
                        "preferred_name": term,
                        "term_normalized": entry.get("term_normalized"),
                        "rxcui": entry.get("rxcui"),
                        "tty": entry.get("tty"),
                        "source": "rxnorm",
                    }
                    rxnorm_proc.add_keyword(term, term_key)
                    count += 1

            if skipped > 0:
                print(f"    [INFO] Skipped {skipped} blacklisted RxNorm terms")
            self._lexicon_stats.append(
                ("RxNorm terms", count, "2025_08_lexicon_drug.json")
            )

        except Exception as e:
            print(f"[WARN] Failed to load RxNorm lexicon: {e}")

    def _init_scispacy(self) -> None:
        """Initialize scispacy NER model."""
        if not SCISPACY_AVAILABLE or spacy is None:
            return

        try:
            # Try large model first, fall back to small
            try:
                self.nlp = spacy.load("en_core_sci_lg")
            except OSError:
                self.nlp = spacy.load("en_core_sci_sm")

            # Add UMLS linker for chemical entities
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
            print(f"[WARN] Failed to initialize scispacy for drugs: {e}")
            self.nlp = None

    def _print_lexicon_summary(self) -> None:
        """Print compact summary of loaded drug lexicons."""
        if not self._lexicon_stats:
            return

        # All drug lexicons go under "Drug" category
        total = sum(count for _, count, _ in self._lexicon_stats if count > 1)
        file_count = len([s for s in self._lexicon_stats if s[1] > 0])
        print(f"\nDrug lexicons: {file_count} sources, {total:,} entries")
        print("─" * 70)
        print(f"  Drug ({total:,} entries)")

        for name, count, filename in self._lexicon_stats:
            if count > 1:
                print(f"    • {name:<26} {count:>8,}  {filename}")
            else:
                print(f"    • {name:<26} {'enabled':>8}  {filename}")
        print()

    def detect(self, doc_graph: DocumentGraph) -> List[DrugCandidate]:
        """
        Detect drug mentions in document.

        Returns list of DrugCandidate objects.
        """
        candidates: List[DrugCandidate] = []
        doc_fingerprint = getattr(
            doc_graph, "fingerprint", self.doc_fingerprint_default
        )

        # Get full text for detection by concatenating all blocks
        full_text = "\n\n".join(
            block.text for block in doc_graph.iter_linear_blocks(skip_header_footer=True)
            if block.text
        )

        # Layer 1: Alexion drugs (specialized, highest priority)
        if self.alexion_processor:
            candidates.extend(
                self._detect_with_lexicon(
                    full_text,
                    doc_graph,
                    doc_fingerprint,
                    self.alexion_processor,
                    self.alexion_drugs,
                    DrugGeneratorType.LEXICON_ALEXION,
                    "2025_08_alexion_drugs.json",
                )
            )

        # Layer 2: Compound ID patterns
        candidates.extend(
            self._detect_compound_patterns(full_text, doc_graph, doc_fingerprint)
        )

        # Layer 3: Investigational drugs
        if self.investigational_processor:
            candidates.extend(
                self._detect_with_lexicon(
                    full_text,
                    doc_graph,
                    doc_fingerprint,
                    self.investigational_processor,
                    self.investigational_drugs,
                    DrugGeneratorType.LEXICON_INVESTIGATIONAL,
                    "2025_08_investigational_drugs.json",
                )
            )

        # Layer 4: FDA approved drugs
        if self.fda_processor:
            candidates.extend(
                self._detect_with_lexicon(
                    full_text,
                    doc_graph,
                    doc_fingerprint,
                    self.fda_processor,
                    self.fda_drugs,
                    DrugGeneratorType.LEXICON_FDA,
                    "2025_08_fda_approved_drugs.json",
                )
            )

        # Layer 5: RxNorm general (more selective)
        if self.rxnorm_processor:
            candidates.extend(
                self._detect_with_lexicon(
                    full_text,
                    doc_graph,
                    doc_fingerprint,
                    self.rxnorm_processor,
                    self.rxnorm_drugs,
                    DrugGeneratorType.LEXICON_RXNORM,
                    "2025_08_lexicon_drug.json",
                )
            )

        # Layer 6: scispacy NER fallback
        if self.nlp:
            candidates.extend(
                self._detect_with_ner(full_text, doc_graph, doc_fingerprint)
            )

        # Deduplicate
        candidates = self._deduplicate(candidates)

        return candidates

    def _detect_with_lexicon(
        self,
        text: str,
        doc_graph: DocumentGraph,
        doc_fingerprint: str,
        processor: KeywordProcessor,
        drug_dict: Dict[str, Dict],
        generator_type: DrugGeneratorType,
        lexicon_source: str,
    ) -> List[DrugCandidate]:
        """Detect drugs using FlashText lexicon matching."""
        candidates = []

        # Extract keywords with positions
        matches = processor.extract_keywords(text, span_info=True)

        for keyword, start, end in matches:
            drug_info = drug_dict.get(keyword, {})
            if not drug_info:
                continue

            matched_text = text[start:end]

            # Apply false positive filter
            context = self._extract_context(text, start, end)
            if self.fp_filter.is_false_positive(matched_text, context, generator_type):
                continue

            # Build identifiers
            identifiers = self._build_identifiers(drug_info)

            # Determine if investigational
            is_investigational = generator_type in {
                DrugGeneratorType.LEXICON_ALEXION,
                DrugGeneratorType.LEXICON_INVESTIGATIONAL,
                DrugGeneratorType.PATTERN_COMPOUND_ID,
            }

            candidate = DrugCandidate(
                doc_id=doc_graph.doc_id,
                matched_text=matched_text,
                preferred_name=drug_info.get("preferred_name", matched_text),
                brand_name=drug_info.get("brand_name"),
                compound_id=drug_info.get("compound_id"),
                field_type=DrugFieldType.EXACT_MATCH,
                generator_type=generator_type,
                identifiers=identifiers,
                context_text=context,
                context_location=Coordinate(page_num=1),  # Simplified
                drug_class=drug_info.get("drug_class"),
                mechanism=drug_info.get("mechanism"),
                development_phase=drug_info.get("status"),
                is_investigational=is_investigational,
                sponsor=drug_info.get("sponsor"),
                conditions=drug_info.get("conditions", []),
                nct_id=drug_info.get("nct_id"),
                dosage_form=drug_info.get("dosage_form"),
                route=drug_info.get("route"),
                marketing_status=drug_info.get("marketing_status"),
                initial_confidence=0.85 if is_investigational else 0.7,
                provenance=DrugProvenanceMetadata(
                    pipeline_version=self.pipeline_version,
                    run_id=self.run_id,
                    doc_fingerprint=doc_fingerprint,
                    generator_name=generator_type,
                    lexicon_source=lexicon_source,
                ),
            )
            candidates.append(candidate)

        return candidates

    def _detect_compound_patterns(
        self,
        text: str,
        doc_graph: DocumentGraph,
        doc_fingerprint: str,
    ) -> List[DrugCandidate]:
        """Detect compound IDs using regex patterns."""
        candidates = []
        seen_positions: Set[Tuple[int, int]] = set()

        for pattern in self.compound_patterns:
            for match in pattern.finditer(text):
                start, end = match.span()

                # Skip if already matched
                if (start, end) in seen_positions:
                    continue
                seen_positions.add((start, end))

                matched_text = match.group(0)
                context = self._extract_context(text, start, end)

                # Check if this matches a known investigational drug
                drug_info = self.investigational_drugs.get(matched_text.lower(), {})
                if not drug_info:
                    # Create basic entry for unknown compound
                    drug_info = {
                        "preferred_name": matched_text,
                        "compound_id": matched_text,
                    }

                # Apply false positive filter for compound patterns too
                if self.fp_filter.is_false_positive(
                    matched_text, context, DrugGeneratorType.PATTERN_COMPOUND_ID
                ):
                    continue

                conditions_raw = drug_info.get("conditions", [])
                conditions_list: List[str] = (
                    conditions_raw if isinstance(conditions_raw, list) else []
                )
                candidate = DrugCandidate(
                    doc_id=doc_graph.doc_id,
                    matched_text=matched_text,
                    preferred_name=drug_info.get("preferred_name", matched_text),
                    compound_id=matched_text,
                    field_type=DrugFieldType.PATTERN_MATCH,
                    generator_type=DrugGeneratorType.PATTERN_COMPOUND_ID,
                    identifiers=[],
                    context_text=context,
                    context_location=Coordinate(page_num=1),
                    is_investigational=True,
                    conditions=conditions_list,
                    nct_id=drug_info.get("nct_id"),
                    initial_confidence=0.8,
                    provenance=DrugProvenanceMetadata(
                        pipeline_version=self.pipeline_version,
                        run_id=self.run_id,
                        doc_fingerprint=doc_fingerprint,
                        generator_name=DrugGeneratorType.PATTERN_COMPOUND_ID,
                        lexicon_source="pattern:compound_id",
                    ),
                )
                candidates.append(candidate)

        return candidates

    def _detect_with_ner(
        self,
        text: str,
        doc_graph: DocumentGraph,
        doc_fingerprint: str,
    ) -> List[DrugCandidate]:
        """Detect drugs using scispacy NER."""
        candidates: list[DrugCandidate] = []

        if not self.nlp:
            return candidates

        # Process text (limit to avoid memory issues)
        max_chars = 100000
        if len(text) > max_chars:
            text = text[:max_chars]

        try:
            doc = self.nlp(text)

            # CHEMICAL semantic type in UMLS
            CHEMICAL_TYPES = {"T109", "T116", "T121", "T123", "T195", "T200"}

            for ent in doc.ents:
                # Check if entity has UMLS linking
                if not hasattr(ent, "_") or not hasattr(ent._, "kb_ents"):
                    continue

                kb_ents = ent._.kb_ents
                if not kb_ents:
                    continue

                # Get best UMLS match
                best_cui, best_score = kb_ents[0]
                if best_score < 0.7:
                    continue

                # Check semantic type - use cui_to_entity API
                linker = self.nlp.get_pipe("scispacy_linker")
                entity_info = linker.kb.cui_to_entity.get(best_cui)
                if entity_info is None:
                    continue

                types = set(entity_info.types)

                # Only keep chemical entities
                if not types.intersection(CHEMICAL_TYPES):
                    continue

                matched_text = ent.text
                context = self._extract_context(text, ent.start_char, ent.end_char)

                # Skip if too short or common word
                if self.fp_filter.is_false_positive(
                    matched_text, context, DrugGeneratorType.SCISPACY_NER
                ):
                    continue

                # Also check the UMLS canonical name (preferred_name) against blacklist
                # This catches cases where matched_text is "MuSK" but UMLS returns
                # "Musk secretion from Musk Deer" which is a false positive
                canonical_name = entity_info.canonical_name or matched_text
                if self.fp_filter.is_false_positive(
                    canonical_name, context, DrugGeneratorType.SCISPACY_NER
                ):
                    continue

                # Additional check: if canonical_name contains known FP substrings
                canonical_lower = canonical_name.lower()
                skip_entity = False
                for fp_substr in self.fp_filter.fp_substrings_lower:
                    if fp_substr in canonical_lower:
                        skip_entity = True
                        break
                # Check for musk deer specifically (common UMLS mislinking)
                if "musk deer" in canonical_lower or "musk secretion" in canonical_lower:
                    skip_entity = True
                if skip_entity:
                    continue

                candidate = DrugCandidate(
                    doc_id=doc_graph.doc_id,
                    matched_text=matched_text,
                    preferred_name=entity_info.canonical_name or matched_text,
                    field_type=DrugFieldType.NER_DETECTION,
                    generator_type=DrugGeneratorType.SCISPACY_NER,
                    identifiers=[
                        DrugIdentifier(
                            system="UMLS_CUI",
                            code=best_cui,
                            display=entity_info.canonical_name,
                        )
                    ],
                    context_text=context,
                    context_location=Coordinate(page_num=1),
                    initial_confidence=best_score,
                    provenance=DrugProvenanceMetadata(
                        pipeline_version=self.pipeline_version,
                        run_id=self.run_id,
                        doc_fingerprint=doc_fingerprint,
                        generator_name=DrugGeneratorType.SCISPACY_NER,
                        lexicon_source="ner:scispacy_umls",
                    ),
                )
                candidates.append(candidate)

        except Exception as e:
            print(f"[WARN] scispacy drug detection error: {e}")

        return candidates

    def _extract_context(self, text: str, start: int, end: int) -> str:
        """Extract context around a match."""
        ctx_start = max(0, start - self.context_window // 2)
        ctx_end = min(len(text), end + self.context_window // 2)
        return text[ctx_start:ctx_end]

    def _build_identifiers(self, drug_info: Dict) -> List[DrugIdentifier]:
        """Build identifier list from drug info."""
        identifiers = []

        if drug_info.get("rxcui"):
            identifiers.append(
                DrugIdentifier(system="RxCUI", code=str(drug_info["rxcui"]))
            )
        if drug_info.get("nct_id"):
            identifiers.append(DrugIdentifier(system="NCT", code=drug_info["nct_id"]))
        if drug_info.get("application_number"):
            identifiers.append(
                DrugIdentifier(system="FDA_NDA", code=drug_info["application_number"])
            )

        return identifiers

    def _deduplicate(self, candidates: List[DrugCandidate]) -> List[DrugCandidate]:
        """
        Deduplicate candidates, preferring specialized sources.

        Priority: Alexion > Investigational > FDA > RxNorm > NER
        Deduplicates by both matched_text AND preferred_name.
        """
        # Priority order
        priority = {
            DrugGeneratorType.LEXICON_ALEXION: 0,
            DrugGeneratorType.PATTERN_COMPOUND_ID: 1,
            DrugGeneratorType.LEXICON_INVESTIGATIONAL: 2,
            DrugGeneratorType.LEXICON_FDA: 3,
            DrugGeneratorType.LEXICON_RXNORM: 4,
            DrugGeneratorType.SCISPACY_NER: 5,
        }

        # Sort all candidates by priority first
        candidates.sort(key=lambda c: priority.get(c.generator_type, 99))

        # Track seen names (both matched_text and preferred_name)
        seen_names: Set[str] = set()
        deduped: List[DrugCandidate] = []

        for c in candidates:
            matched_key = c.matched_text.lower().strip()
            preferred_key = (c.preferred_name or "").lower().strip()

            # Skip if we've seen this name already
            if matched_key in seen_names or (preferred_key and preferred_key in seen_names):
                continue

            # Add to result and mark as seen
            deduped.append(c)
            seen_names.add(matched_key)
            if preferred_key:
                seen_names.add(preferred_key)

        return deduped

# corpus_metadata/C_generators/C07a_drug_fp_filter.py
"""
Drug false positive filtering and abbreviation mappings.

This module contains:
- DrugFalsePositiveFilter: Filter for false positive drug matches
- DRUG_ABBREVIATIONS: Common drug abbreviations mapped to canonical names

Extracted from C07_strategy_drug.py for maintainability.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Set

if TYPE_CHECKING:
    from A_core.A06_drug_models import DrugGeneratorType


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
        # Generic clinical terms (not drugs)
        "therapeutic intervention",
        "induction",
        "intervention",
        "interventions",
        # Enzymes/proteins commonly misidentified as drugs
        "alkaline phosphatase",
        "myeloperoxidase",
        "proteinase 3",
        "proteinase",
        "aspartate aminotransferase",
        "aspartate transaminase",
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

    def __init__(self) -> None:
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
        self, matched_text: str, context: str, generator_type: "DrugGeneratorType"
    ) -> bool:
        """
        Check if a drug match is likely a false positive.

        Returns True if the match should be filtered out.
        """
        # Import here to avoid circular imports
        from A_core.A06_drug_models import DrugGeneratorType as DGT

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
        if generator_type == DGT.SCISPACY_NER:
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
            DGT.SCISPACY_NER,
            DGT.LEXICON_RXNORM,
            DGT.LEXICON_FDA,
        }:
            if text_lower in self.gene_symbols_lower:
                return True

        # Skip common words (unless from specialized lexicon)
        if generator_type not in {
            DGT.LEXICON_ALEXION,
            DGT.LEXICON_INVESTIGATIONAL,
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
            author_pattern = re.compile(
                rf"\b{re.escape(text_stripped)}\s+[A-Z]{{1,2}}[,\.\s]",
                re.IGNORECASE
            )
            if author_pattern.search(context):
                return True

        return False


# -------------------------
# Drug Abbreviation Mappings
# -------------------------

# Common drug abbreviations mapped to their canonical full names.
# Used during deduplication to link abbreviations with their full forms.
# Keys are lowercase abbreviations, values are lowercase canonical names.
DRUG_ABBREVIATIONS: Dict[str, str] = {
    # Immunosuppressants
    "mtx": "methotrexate",
    "aza": "azathioprine",
    "mmf": "mycophenolate mofetil",
    "cspa": "cyclosporine",
    "csa": "cyclosporine",
    "cyc": "cyclophosphamide",
    "ctx": "cyclophosphamide",
    "rtx": "rituximab",
    # Corticosteroids
    "pred": "prednisolone",
    "mpd": "methylprednisolone",
    "dex": "dexamethasone",
    "hc": "hydrocortisone",
    # Common chemotherapy
    "5-fu": "fluorouracil",
    "5fu": "fluorouracil",
    "dox": "doxorubicin",
    "cis": "cisplatin",
    "carbo": "carboplatin",
    "pacl": "paclitaxel",
    "vcr": "vincristine",
    "vbl": "vinblastine",
    # Antibiotics
    "amox": "amoxicillin",
    "augmentin": "amoxicillin clavulanate",
    "vanco": "vancomycin",
    "gent": "gentamicin",
    "ceftx": "ceftriaxone",
    "tmp-smx": "trimethoprim sulfamethoxazole",
    "bactrim": "trimethoprim sulfamethoxazole",
    # Anticoagulants
    "ufh": "unfractionated heparin",
    "lmwh": "low molecular weight heparin",
    # Biologics
    "ivig": "intravenous immunoglobulin",
    "scig": "subcutaneous immunoglobulin",
    "tnfi": "tumor necrosis factor inhibitor",
    # Partial name to full name mappings
    "mycophenolate": "mycophenolate mofetil",
    # Cardiovascular
    "asa": "aspirin",
    "atenol": "atenolol",
    "ntg": "nitroglycerin",
    "acei": "angiotensin converting enzyme inhibitor",
    "arb": "angiotensin receptor blocker",
    # Pain management
    "apap": "acetaminophen",
    "nsaid": "nonsteroidal anti-inflammatory drug",
    # Diabetes
    "met": "metformin",
    # Other common
    "ppi": "proton pump inhibitor",
    "h2ra": "h2 receptor antagonist",
}

"""
FDA Therapeutic Areas and Disease Keywords Configuration
=========================================================
Edit this file to add/remove diseases and therapeutic areas.

Used by: syncher.py
"""

# ============================================================================
# THERAPEUTIC AREAS AND DISEASE KEYWORDS
# ============================================================================

THERAPEUTIC_AREAS = {
    'nephrology': {
        'rare_diseases': [
            # Glomerular diseases
            "IgA nephropathy",
            "focal segmental glomerulosclerosis",
            "FSGS",
            "membranous nephropathy",
            "minimal change disease",
            "lupus nephritis",
            "ANCA vasculitis",
            "granulomatosis with polyangiitis",
            "Wegener granulomatosis",
            
            # Complement-mediated kidney diseases
            "atypical hemolytic uremic syndrome",
            "aHUS",
            "C3 glomerulopathy",
            "C3 glomerulonephritis",
            "dense deposit disease",
            "paroxysmal nocturnal hemoglobinuria",
            "PNH",
            
            # Genetic kidney diseases
            "Alport syndrome",
            "Fabry disease",
            "cystinosis",
            "primary hyperoxaluria",
            "hyperoxaluria type 1",
            "hyperoxaluria type 2",
            "Dent disease",
            "nephronophthisis",
            "Bartter syndrome",
            "Gitelman syndrome",
            
            # Polycystic kidney disease
            "autosomal dominant polycystic kidney disease",
            "ADPKD",
            "autosomal recessive polycystic kidney disease",
            "ARPKD",
            
            # Metabolic kidney diseases
            "cystinuria",
            "renal tubular acidosis",
            
            # Other kidney diseases
            "nephrotic syndrome",
            "nephrotic syndrome steroid resistant",
            "congenital nephrotic syndrome",
            "proteinuria",
            "chronic kidney disease",
            "CKD",
            "end stage renal disease",
            "ESRD",
            "diabetic nephropathy",
            "diabetic kidney disease",
            
            # Renal vasculitis
            "microscopic polyangiitis",
            "Goodpasture syndrome",
            "anti-GBM disease",
            
            # Tubular disorders
            "Fanconi syndrome",
            "renal Fanconi syndrome"
        ],
        
        'drug_classes': [
            "complement inhibitor",
            "C5 inhibitor",
            "C3 inhibitor",
            "factor D inhibitor",
            "endothelin receptor antagonist",
            "ERA",
            "SGLT2 inhibitor",
            "sodium glucose cotransporter 2 inhibitor",
            "mineralocorticoid receptor antagonist",
            "MRA",
            "phosphate binder",
            "calcimimetic",
            "erythropoiesis stimulating agent",
            "ESA",
            "erythropoietin",
            "iron supplement",
            "vitamin D analog",
            "immunosuppressant kidney",
            "calcineurin inhibitor"
        ]
    },
    
    'hematology': {
        'rare_diseases': [
            # Hemophilia and coagulation disorders
            "hemophilia A",
            "hemophilia B",
            "Christmas disease",
            "von Willebrand disease",
            "VWD",
            "factor VII deficiency",
            "factor X deficiency",
            "factor XI deficiency",
            "factor XIII deficiency",
            "factor V deficiency",
            "factor II deficiency",
            "prothrombin deficiency",
            "Glanzmann thrombasthenia",
            "Bernard-Soulier syndrome",
            "hemophilia C",
            
            # Sickle cell and hemoglobinopathies
            "sickle cell disease",
            "SCD",
            "sickle cell anemia",
            "beta thalassemia",
            "thalassemia major",
            "thalassemia intermedia",
            "alpha thalassemia",
            "hemoglobin H disease",
            "HbH disease",
            
            # Rare anemias
            "paroxysmal nocturnal hemoglobinuria",
            "PNH",
            "aplastic anemia",
            "severe aplastic anemia",
            "pure red cell aplasia",
            "PRCA",
            "Diamond-Blackfan anemia",
            "DBA",
            "Fanconi anemia",
            "congenital dyserythropoietic anemia",
            "CDA",
            "pyruvate kinase deficiency",
            "PK deficiency",
            "glucose-6-phosphate dehydrogenase deficiency",
            "G6PD deficiency",
            "hereditary spherocytosis",
            "hereditary elliptocytosis",
            
            # Bone marrow failure and myeloid disorders
            "myelodysplastic syndrome",
            "MDS",
            "myelodysplastic syndromes",
            "myelofibrosis",
            "primary myelofibrosis",
            "PMF",
            "polycythemia vera",
            "PV",
            "essential thrombocythemia",
            "ET",
            "chronic myelomonocytic leukemia",
            "CMML",
            
            # Thrombotic and platelet disorders
            "thrombotic thrombocytopenic purpura",
            "TTP",
            "acquired TTP",
            "congenital TTP",
            "heparin-induced thrombocytopenia",
            "HIT",
            "immune thrombocytopenia",
            "ITP",
            "immune thrombocytopenic purpura",
            "hemolytic uremic syndrome",
            "HUS",
            "Shiga toxin HUS",
            
            # Thrombotic disorders
            "antiphospholipid syndrome",
            "APS",
            "factor V Leiden",
            "protein C deficiency",
            "protein S deficiency",
            "antithrombin deficiency",
            "thrombophilia",
            
            # Hematologic malignancies
            "acute myeloid leukemia",
            "AML",
            "acute lymphoblastic leukemia",
            "ALL",
            "acute lymphocytic leukemia",
            "chronic myeloid leukemia",
            "CML",
            "chronic lymphocytic leukemia",
            "CLL",
            "multiple myeloma",
            "plasma cell myeloma",
            "Hodgkin lymphoma",
            "non-Hodgkin lymphoma",
            "NHL",
            "diffuse large B-cell lymphoma",
            "DLBCL",
            "follicular lymphoma",
            "mantle cell lymphoma",
            "MCL",
            "marginal zone lymphoma",
            "Burkitt lymphoma",
            "T-cell lymphoma",
            "cutaneous T-cell lymphoma",
            "CTCL",
            "mycosis fungoides",
            "Sezary syndrome",
            "Waldenström macroglobulinemia",
            "WM",
            "hairy cell leukemia",
            
            # Iron overload and metabolism
            "hemochromatosis",
            "hereditary hemochromatosis",
            "iron overload",
            "transfusional iron overload",
            
            # Other rare hematologic disorders
            "porphyria",
            "acute intermittent porphyria",
            "porphyria cutanea tarda",
            "mastocytosis",
            "systemic mastocytosis",
            "eosinophilic disorders",
            "hypereosinophilic syndrome",
            "HES",
            "cold agglutinin disease",
            "CAD",
            "warm autoimmune hemolytic anemia",
            "WAIHA",
            "autoimmune hemolytic anemia",
            "AIHA",
            
            # Neutropenia
            "severe chronic neutropenia",
            "congenital neutropenia",
            "cyclic neutropenia",
            "Kostmann syndrome"
        ],
        
        'drug_classes': [
            "factor replacement",
            "factor VIII",
            "factor IX",
            "factor VII",
            "recombinant factor",
            "gene therapy",
            "gene therapy hemophilia",
            "bispecific antibody",
            "bispecific antibody hemophilia",
            "emicizumab",
            "monoclonal antibody",
            "complement inhibitor",
            "C5 inhibitor hematology",
            "thrombopoietin receptor agonist",
            "TPO receptor agonist",
            "romiplostim",
            "eltrombopag",
            "erythropoiesis stimulating agent",
            "ESA hematology",
            "erythropoietin",
            "iron chelator",
            "deferasirox",
            "deferiprone",
            "deferoxamine",
            "BCL2 inhibitor",
            "venetoclax",
            "BTK inhibitor",
            "Bruton tyrosine kinase inhibitor",
            "ibrutinib",
            "acalabrutinib",
            "JAK inhibitor",
            "Janus kinase inhibitor",
            "ruxolitinib",
            "FLT3 inhibitor",
            "IDH inhibitor",
            "IDH1 inhibitor",
            "IDH2 inhibitor",
            "CD20 antibody",
            "rituximab",
            "CD38 antibody",
            "daratumumab",
            "proteasome inhibitor",
            "bortezomib",
            "carfilzomib",
            "immunomodulatory drug",
            "IMiD",
            "lenalidomide",
            "pomalidomide",
            "hypomethylating agent",
            "azacitidine",
            "decitabine",
            "anti-ADAMTS13 antibody",
            "caplacizumab"
        ]
    }
}

# ============================================================================
# CONFIGURATION HELPERS
# ============================================================================

def get_all_therapeutic_areas():
    """Get list of all therapeutic area names"""
    return list(THERAPEUTIC_AREAS.keys())

def get_disease_count(therapeutic_area):
    """Get count of diseases for a therapeutic area"""
    if therapeutic_area in THERAPEUTIC_AREAS:
        return len(THERAPEUTIC_AREAS[therapeutic_area]['rare_diseases'])
    return 0

def get_drug_class_count(therapeutic_area):
    """Get count of drug classes for a therapeutic area"""
    if therapeutic_area in THERAPEUTIC_AREAS:
        return len(THERAPEUTIC_AREAS[therapeutic_area]['drug_classes'])
    return 0

def print_configuration_summary():
    """Print summary of therapeutic areas configuration"""
    print("\n" + "="*70)
    print("THERAPEUTIC AREAS CONFIGURATION")
    print("="*70)
    
    for area in get_all_therapeutic_areas():
        disease_count = get_disease_count(area)
        class_count = get_drug_class_count(area)
        print(f"\n{area.upper()}:")
        print(f"  Rare Diseases: {disease_count}")
        print(f"  Drug Classes: {class_count}")
        print(f"  Total Keywords: {disease_count + class_count}")
    
    total_diseases = sum(get_disease_count(a) for a in get_all_therapeutic_areas())
    total_classes = sum(get_drug_class_count(a) for a in get_all_therapeutic_areas())
    
    print(f"\nTOTAL:")
    print(f"  Therapeutic Areas: {len(get_all_therapeutic_areas())}")
    print(f"  Total Diseases: {total_diseases}")
    print(f"  Total Drug Classes: {total_classes}")
    print(f"  Total Keywords: {total_diseases + total_classes}")
    print("="*70 + "\n")

# ============================================================================
# MAIN - For testing this file directly
# ============================================================================

if __name__ == "__main__":
    print_configuration_summary()
    
    # Example: List all nephrology diseases
    print("\nNEPHROLOGY DISEASES:")
    for disease in THERAPEUTIC_AREAS['nephrology']['rare_diseases']:
        print(f"  - {disease}")
    
    # Example: List all hematology drug classes
    print("\nHEMATOLOGY DRUG CLASSES:")
    for drug_class in THERAPEUTIC_AREAS['hematology']['drug_classes']:
        print(f"  - {drug_class}")
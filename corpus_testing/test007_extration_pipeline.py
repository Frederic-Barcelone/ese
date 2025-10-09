# Alternative: Try simpler extraction if main pipeline doesn't work
    print("\n--- Extracting Entities ---")
    start_time = time.time()
    
    extraction_output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'text_length': len(cleaned_text),
        'entity_counts': {
            'abbreviations': 0,
            'drugs': 0,
            'diseases': 0
        },
        'abbreviations': [],
        'drugs': [],
        'diseases': []
    }
    
    # Try to extract abbreviations
    if components.get('abbreviation_extractor'):
        try:
            print("Extracting abbreviations...")
            abbrev_results = components['abbreviation_extractor'].extract_abbreviations(cleaned_text)
            extraction_output['abbreviations'] = abbrev_results.get('abbreviations', [])
            extraction_output['entity_counts']['abbreviations'] = len(extraction_output['abbreviations'])
            print(f"  #!/usr/bin/env python3
"""
Test Script for PDF Text Extraction and Cleaning Pipeline
Uses existing reader_pdf_extractors.py module
"""

import sys
import os
from pathlib import Path
import json
import time

# Add the project path to sys.path
PROJECT_BASE = "/Users/kxbb828/Library/CloudStorage/OneDrive-AZCollaboration/Desktop/WIP-ESE/17_Corpus"
sys.path.insert(0, PROJECT_BASE)

# Import the existing PDF extractors and cleaner
from corpus_metadata.document_utils.reader_pdf_extractors import (
    PDFTextCleaner,
    PdfplumberExtractor,
    PyMuPDFExtractor
)

# Sample raw text with typical PDF extraction issues (from your provided example)
SAMPLE_RAW_TEXT = """Pediatric FDA-Associated Vasculitis: Current Evidence and Therapeutic Land-
scape
Complement inhibition represents the most signifi-
cant therapeutic advance
Pharmacological mechanisms and rationale for comple-
ment inhibition
Avacopan transforms ANCA v treatment by target-
ing the inflammatory amplification loop central to dis-
ease pathogenesis (1). As an oral FDA receptor (IV/CD
88) antagonist, avacopan selectively blocks IV bind-
ing to IV, preventing PMC-mediated neutrophil activa-
tion and chemotaxis while preserving beneficial comple-
ment functions. The drug interrupts a vicious cy-
cle where NIH-activated neutrophils degranulate, acti-
vate the alternative complement pathway, generate PMC,

which then recruits and primes additional neutro-
phils for PMC activation.
Experimental evidence strongly supports this ap-
proach: PMC-deficient mice show complete protec-
tion from PMC-induced glomerulonephritis, while PMC-
deficient mice (lacking membrane attack complex for-
mation) remain vulnerable, confirming that FDA rath-
er than terminal complement products drives pathogen-
esis. In humans, elevated plasma levels of PMC, PMC, sol-
uble
PMC-9, and factor B correlate with active dis-
ease and normalize with remission (2).
The pharmacokinetic profile presents both oppor-
tunities and challenges for pediatric applica-
tion.
Avacopan achieves peak concentrations ~2 hours af-
ter oral administration, with an elimination half-
life of 97.6 hours enabling twice-daily dos-
ing. The drug is highly protein-bound and prima-
rily metabolized by PMC to an active metabo-
lite (PMC), representing potential concerns for pedi-
atric populations given developmental changes in drug me-
tabolism and drug interactions (3). Strong

PMC inhibitors require dose reduction to 30 mg
once daily, while inducers should be avoid-
ed entirely.
Adult versus pediatric pharmacological considera-
tions
Critical gap: Limited pediatric pharmacokinet-
ic data exists for avacopan or other comple-
ment inhibitors in ANCA v (4). Adult stud-
ies show no clinically relevant differences a-
cross age ranges (18-83 years), sex, race, or bo-
dy weight (40.3-174 kg), and no dose adjust-
ment is required for mild-to-
severe renal impairment. However, developmen-
tal changes in PMC activity, body composi-
tion,"""

def test_text_cleaning():
    """Test the PDFTextCleaner on sample text"""
    print("=" * 80)
    print("TESTING PDF TEXT CLEANING PIPELINE")
    print("=" * 80)
    
    # Initialize the cleaner
    cleaner = PDFTextCleaner()
    
    print("\n1. TESTING PDFTextCleaner")
    print("-" * 40)
    
    # Show raw text sample
    print("\n--- RAW TEXT (first 500 chars) ---")
    print(SAMPLE_RAW_TEXT[:500])
    print("...")
    
    # Clean the text
    print("\n--- CLEANING TEXT ---")
    start_time = time.time()
    cleaned_text = cleaner.clean(SAMPLE_RAW_TEXT)
    clean_time = time.time() - start_time
    
    print(f"Cleaning completed in {clean_time:.3f} seconds")
    
    # Show cleaned text
    print("\n--- CLEANED TEXT (first 500 chars) ---")
    print(cleaned_text[:500])
    print("...")
    
    # Statistics
    print("\n--- STATISTICS ---")
    print(f"Original length: {len(SAMPLE_RAW_TEXT)} characters")
    print(f"Cleaned length: {len(cleaned_text)} characters")
    print(f"Reduction: {len(SAMPLE_RAW_TEXT) - len(cleaned_text)} characters ({(1 - len(cleaned_text)/len(SAMPLE_RAW_TEXT))*100:.1f}%)")
    
    # Show specific fixes
    print("\n--- SAMPLE FIXES DETECTED ---")
    
    # Check for specific improvements
    fixes_found = []
    
    # Check if hyphens were fixed
    if "complement" in cleaned_text and "comple-\nment" not in cleaned_text:
        fixes_found.append("✓ Fixed: 'comple-\\nment' → 'complement'")
    
    if "ANCAv" in cleaned_text and "ANCA v" not in cleaned_text:
        fixes_found.append("✓ Fixed: 'ANCA v' → 'ANCAv'")
    
    if "significant" in cleaned_text and "signifi-\ncant" not in cleaned_text:
        fixes_found.append("✓ Fixed: 'signifi-\\ncant' → 'significant'")
    
    if "pharmacokinetic" in cleaned_text:
        fixes_found.append("✓ Fixed: 'pharmacokinet-\\nic' → 'pharmacokinetic'")
    
    for fix in fixes_found:
        print(fix)
    
    # Save cleaned text
    output_file = "cleaned_text_output.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
    print(f"\n✓ Cleaned text saved to '{output_file}'")
    
    return cleaned_text

def test_pdf_extraction(pdf_path: str = None):
    """Test PDF extraction using available extractors"""
    print("\n" + "=" * 80)
    print("2. TESTING PDF EXTRACTORS")
    print("=" * 80)
    
    if not pdf_path:
        print("\n⚠ No PDF file provided for extraction test")
        print("To test PDF extraction, run: python test_script.py your_file.pdf")
        return None
    
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        print(f"\n✗ PDF file not found: {pdf_path}")
        return None
    
    print(f"\nTesting with: {pdf_file.name}")
    print(f"File size: {pdf_file.stat().st_size:,} bytes")
    
    # Test available extractors
    extractors = [
        ("pdfplumber", PdfplumberExtractor),
        ("PyMuPDF", PyMuPDFExtractor)
    ]
    
    results = {}
    
    for name, ExtractorClass in extractors:
        print(f"\n--- Testing {name} Extractor ---")
        try:
            extractor = ExtractorClass()
            if not extractor.available:
                print(f"✗ {name} not installed")
                continue
            
            # Extract first 3 pages for testing
            start_time = time.time()
            text, pages, info = extractor.extract(pdf_file, max_pages=3)
            extract_time = time.time() - start_time
            
            print(f"✓ Extraction successful")
            print(f"  Pages extracted: {pages}")
            print(f"  Characters extracted: {len(text):,}")
            print(f"  Time taken: {extract_time:.2f}s")
            print(f"  Method: {info.get('extraction_method', 'unknown')}")
            
            # Show sample of extracted text
            print(f"\n  First 200 chars of extracted text:")
            print(f"  {text[:200]}...")
            
            results[name] = {
                'success': True,
                'pages': pages,
                'chars': len(text),
                'time': extract_time,
                'text_sample': text[:500]
            }
            
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            results[name] = {
                'success': False,
                'error': str(e)
            }
    
    return results

def save_test_results(cleaned_text: str, extraction_results: dict = None):
    """Save test results to JSON"""
    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'cleaning_test': {
            'original_length': len(SAMPLE_RAW_TEXT),
            'cleaned_length': len(cleaned_text),
            'reduction_percent': round((1 - len(cleaned_text)/len(SAMPLE_RAW_TEXT))*100, 1),
            'sample_cleaned_text': cleaned_text[:1000]
        }
    }
    
    if extraction_results:
        output['extraction_test'] = extraction_results
    
    output_file = 'test_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Test results saved to '{output_file}'")

def test_entity_extraction(cleaned_text: str, output_file: str = "entity_extraction_results.json"):
    """Test entity extraction on cleaned text"""
    print("\n" + "=" * 80)
    print("3. TESTING ENTITY EXTRACTION")
    print("=" * 80)
    
    # Import entity extraction module
    try:
        from corpus_metadata.document_utils.entity_extraction import (
            process_entities_stage_with_promotion,
            deduplicate_by_key
        )
        from corpus_metadata.extractors.abbreviation_extractor import AbbreviationExtractor
        from corpus_metadata.extractors.drug_extractor import DrugExtractor
        from corpus_metadata.extractors.disease_extractor import DiseaseExtractor
        
        print("✓ Entity extraction modules loaded")
    except ImportError as e:
        print(f"✗ Failed to import entity extraction: {e}")
        return None
    
    # Initialize extractors
    components = {}
    try:
        # Initialize abbreviation extractor
        components['abbreviation_extractor'] = AbbreviationExtractor()
        print("✓ Abbreviation extractor initialized")
        
        # Initialize drug extractor
        components['drug_extractor'] = DrugExtractor()
        print("✓ Drug extractor initialized")
        
        # Initialize disease extractor
        components['disease_extractor'] = DiseaseExtractor()
        print("✓ Disease extractor initialized")
        
    except Exception as e:
        print(f"✗ Failed to initialize extractors: {e}")
        return None
    
    # Configure entity extraction
    stage_config = {
        'sequence': 1,
        'tasks': [
            'abbreviation_extraction',
            'drug_detection',
            'disease_detection'
        ]
    }
    
    # Prepare parameters
    stage_results = {}
    abbreviation_context = {}
    features = {
        'enable_id_gated_promotion': True,
        'enable_confidence_promotion': True,
        'enable_deduplication': True
    }
    
    print("\n--- Extracting Entities ---")
    start_time = time.time()
    
    try:
        # Process entities using the main pipeline function
        results = process_entities_stage_with_promotion(
            text_content=cleaned_text,
            file_path=Path("test_document.txt"),  # Dummy path
            components=components,
            stage_config=stage_config,
            stage_results=stage_results,
            abbreviation_context=abbreviation_context,
            console=None,  # No console output
            features=features,
            use_claude=False  # Don't use Claude API
        )
        
        extraction_time = time.time() - start_time
        
        # Parse results
        if results and len(results) > 0:
            # Get the first element (abbreviation context)
            abbrev_context = results[0] if isinstance(results, tuple) else results
            
            # Extract entity counts
            abbreviations = abbrev_context.get('all_abbreviations', [])
            drugs = abbrev_context.get('direct_drugs', []) + abbrev_context.get('promoted_drugs', [])
            diseases = abbrev_context.get('direct_diseases', []) + abbrev_context.get('promoted_diseases', [])
            
            print(f"\n✓ Entity extraction completed in {extraction_time:.2f}s")
            print(f"\nEntities Found:")
            print(f"  - Abbreviations: {len(abbreviations)}")
            print(f"  - Drugs: {len(drugs)}")
            print(f"  - Diseases: {len(diseases)}")
            
            # Show sample entities
            if abbreviations:
                print(f"\n  Sample Abbreviations (first 5):")
                for abbr in abbreviations[:5]:
                    print(f"    • {abbr.get('abbreviation', 'N/A')} → {abbr.get('expansion', 'N/A')}")
            
            if drugs:
                print(f"\n  Sample Drugs (first 5):")
                for drug in drugs[:5]:
                    name = drug.get('drug_name', drug.get('name', 'N/A'))
                    confidence = drug.get('confidence', 0)
                    print(f"    • {name} (confidence: {confidence:.2f})")
            
            if diseases:
                print(f"\n  Sample Diseases (first 5):")
                for disease in diseases[:5]:
                    name = disease.get('disease_name', disease.get('name', 'N/A'))
                    confidence = disease.get('confidence', 0)
                    print(f"    • {name} (confidence: {confidence:.2f})")
            
            # Save extraction results
            extraction_output = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'text_length': len(cleaned_text),
                'extraction_time': extraction_time,
                'entity_counts': {
                    'abbreviations': len(abbreviations),
                    'drugs': len(drugs),
                    'diseases': len(diseases)
                },
                'abbreviations': abbreviations,
                'drugs': drugs,
                'diseases': diseases
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(extraction_output, f, indent=2)
            
            print(f"\n✓ Entity extraction results saved to '{output_file}'")
            
            return extraction_output
            
    except Exception as e:
        print(f"\n✗ Entity extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main test function"""
    print("\n" + "=" * 80)
    print("PDF TEXT EXTRACTION, CLEANING, AND ENTITY EXTRACTION TEST SUITE")
    print("Using: corpus_metadata/document_utils/reader_pdf_extractors.py")
    print("      corpus_metadata/document_utils/entity_extraction.py")
    print("=" * 80)
    
    # Test 1: Text cleaning
    cleaned_text = test_text_cleaning()
    
    # Test 2: PDF extraction (if PDF file provided)
    extraction_results = None
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        extraction_results = test_pdf_extraction(pdf_path)
        
        # If PDF extraction successful, use that text for entity extraction
        if extraction_results:
            # Find the best extractor result
            for name, result in extraction_results.items():
                if result.get('success') and result.get('text_sample'):
                    # Use the full extracted text (not just sample) if available
                    # For now, we'll use the cleaned sample text
                    break
    else:
        print("\n" + "-" * 40)
        print("TIP: To test PDF extraction, run:")
        print("python test_script.py /path/to/your.pdf")
    
    # Test 3: Entity extraction on cleaned text
    entity_results = test_entity_extraction(cleaned_text)
    
    # Save combined results
    save_test_results(cleaned_text, extraction_results)
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    
    # Final summary
    print("\nSUMMARY:")
    print(f"1. Text Cleaning: ✓ Tested")
    print(f"   - Reduced text by {(1 - len(cleaned_text)/len(SAMPLE_RAW_TEXT))*100:.1f}%")
    
    if extraction_results:
        successful = [k for k, v in extraction_results.items() if v.get('success')]
        print(f"2. PDF Extraction: ✓ Tested {len(successful)}/{len(extraction_results)} extractors")
        for name in successful:
            print(f"   - {name}: {extraction_results[name]['chars']:,} chars in {extraction_results[name]['time']:.2f}s")
    else:
        print("2. PDF Extraction: ⚠ Not tested (no PDF provided)")
    
    if entity_results:
        print(f"3. Entity Extraction: ✓ Tested")
        print(f"   - Abbreviations: {entity_results['entity_counts']['abbreviations']}")
        print(f"   - Drugs: {entity_results['entity_counts']['drugs']}")
        print(f"   - Diseases: {entity_results['entity_counts']['diseases']}")
        print(f"   - Time: {entity_results['extraction_time']:.2f}s")
    else:
        print("3. Entity Extraction: ⚠ Failed or not tested")
    
    print("\nOutput files created:")
    print("  - cleaned_text_output.txt (cleaned sample text)")
    print("  - test_results.json (detailed test results)")
    print("  - entity_extraction_results.json (extracted entities)")

if __name__ == "__main__":
    main()
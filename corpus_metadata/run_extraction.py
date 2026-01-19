#!/usr/bin/env python3
# corpus_metadata/run_extraction.py
"""
Selective extraction runner for the corpus_metadata pipeline.

Run specific extractors independently or in combination.

Usage:
    # Run drug extraction only
    python run_extraction.py document.pdf --drugs

    # Run disease extraction only
    python run_extraction.py document.pdf --diseases

    # Run multiple extractors
    python run_extraction.py document.pdf --drugs --diseases --abbreviations

    # Use a preset
    python run_extraction.py document.pdf --preset standard
    python run_extraction.py document.pdf --preset entities_only

    # Run all extractors
    python run_extraction.py document.pdf --all

    # With options
    python run_extraction.py document.pdf --drugs --no-llm --output-dir ./results

Available presets:
    drugs_only        - Drug detection only
    diseases_only     - Disease detection only
    abbreviations_only - Abbreviation detection only
    feasibility_only  - Feasibility extraction only
    entities_only     - Drugs + Diseases + Abbreviations
    clinical_entities - Drugs + Diseases
    metadata_only     - Authors + Citations + Document metadata
    standard          - Drugs + Diseases + Abbreviations + Feasibility
    all               - Everything
    minimal           - Just abbreviations (fastest)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from G_config.extraction_config import (
    ExtractionConfig,
    ExtractionPreset,
)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="Run selective entity extraction on PDF documents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input files
    parser.add_argument(
        "input",
        nargs="+",
        help="PDF file(s) or directory to process",
    )

    # Preset selection
    parser.add_argument(
        "--preset", "-p",
        type=str,
        choices=[p.value for p in ExtractionPreset],
        help="Use a predefined extraction preset",
    )

    # Individual extractor flags
    extractor_group = parser.add_argument_group("Extractors")
    extractor_group.add_argument(
        "--drugs", "-d",
        action="store_true",
        help="Enable drug extraction",
    )
    extractor_group.add_argument(
        "--diseases", "-D",
        action="store_true",
        help="Enable disease extraction",
    )
    extractor_group.add_argument(
        "--abbreviations", "-a",
        action="store_true",
        help="Enable abbreviation extraction",
    )
    extractor_group.add_argument(
        "--feasibility", "-f",
        action="store_true",
        help="Enable feasibility extraction",
    )
    extractor_group.add_argument(
        "--pharma",
        action="store_true",
        help="Enable pharma company extraction",
    )
    extractor_group.add_argument(
        "--authors",
        action="store_true",
        help="Enable author extraction",
    )
    extractor_group.add_argument(
        "--citations",
        action="store_true",
        help="Enable citation extraction",
    )
    extractor_group.add_argument(
        "--metadata",
        action="store_true",
        help="Enable document metadata extraction",
    )
    extractor_group.add_argument(
        "--tables",
        action="store_true",
        help="Enable table extraction",
    )
    extractor_group.add_argument(
        "--all",
        action="store_true",
        help="Enable all extractors",
    )

    # Processing options
    options_group = parser.add_argument_group("Processing Options")
    options_group.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM validation for abbreviations",
    )
    options_group.add_argument(
        "--no-llm-feasibility",
        action="store_true",
        help="Use pattern-based feasibility extraction instead of LLM",
    )
    options_group.add_argument(
        "--vlm-tables",
        action="store_true",
        help="Use vision model for table extraction",
    )
    options_group.add_argument(
        "--skip-normalization",
        action="store_true",
        help="Skip entity normalization/enrichment",
    )
    options_group.add_argument(
        "--max-pages",
        type=int,
        help="Maximum number of pages to process",
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output-dir", "-o",
        type=Path,
        help="Output directory for results",
    )
    output_group.add_argument(
        "--combined",
        action="store_true",
        help="Export all results to a single combined JSON file",
    )
    output_group.add_argument(
        "--no-export",
        action="store_true",
        help="Don't export JSON files (just print summary)",
    )
    output_group.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output",
    )
    output_group.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    return parser


def build_config(args: argparse.Namespace) -> ExtractionConfig:
    """Build ExtractionConfig from parsed arguments."""
    # Start with preset if specified
    if args.preset:
        preset = ExtractionPreset(args.preset)
        config = ExtractionConfig.from_preset(preset)
    elif args.all:
        config = ExtractionConfig.from_preset(ExtractionPreset.ALL)
    else:
        # Check if any individual flags are set
        has_flags = any([
            args.drugs, args.diseases, args.abbreviations,
            args.feasibility, args.pharma, args.authors,
            args.citations, args.metadata, args.tables,
        ])

        if has_flags:
            # Build from individual flags
            config = ExtractionConfig(
                drugs=args.drugs,
                diseases=args.diseases,
                abbreviations=args.abbreviations,
                feasibility=args.feasibility,
                pharma_companies=args.pharma,
                authors=args.authors,
                citations=args.citations,
                document_metadata=args.metadata,
                tables=args.tables,
            )
        else:
            # Default to standard preset
            config = ExtractionConfig.from_preset(ExtractionPreset.STANDARD)

    # Apply processing options
    if args.no_llm:
        config.use_llm_validation = False
    if args.no_llm_feasibility:
        config.use_llm_feasibility = False
    if args.vlm_tables:
        config.use_vlm_tables = True
    if args.skip_normalization:
        config.skip_normalization = True
    if args.max_pages:
        config.max_pages = args.max_pages

    # Apply output options
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.combined:
        config.export_combined = True
    if args.no_export:
        config.export_json = False

    return config


def collect_pdf_files(inputs: List[str]) -> List[Path]:
    """Collect PDF files from input paths."""
    pdf_files = []

    for input_path in inputs:
        path = Path(input_path)

        if path.is_file() and path.suffix.lower() == ".pdf":
            pdf_files.append(path)
        elif path.is_dir():
            pdf_files.extend(path.glob("**/*.pdf"))
        else:
            print(f"Warning: Skipping invalid path: {input_path}", file=sys.stderr)

    return sorted(set(pdf_files))


def run_extraction(
    pdf_path: Path,
    config: ExtractionConfig,
    verbose: bool = False,
    quiet: bool = False,
) -> Dict[str, Any]:
    """
    Run extraction on a single PDF with the given configuration.

    Args:
        pdf_path: Path to PDF file
        config: Extraction configuration
        verbose: Enable verbose output
        quiet: Minimal output

    Returns:
        Dictionary with extraction results
    """
    from orchestrator import Orchestrator

    if not quiet:
        print(f"\n{'='*60}")
        print(f"Processing: {pdf_path.name}")
        print(f"Extractors: {config}")
        print(f"{'='*60}")

    start_time = time.time()

    # Initialize orchestrator with config
    orchestrator = Orchestrator(
        skip_validation=not config.use_llm_validation,
    )

    # Override orchestrator settings based on config
    if not config.use_llm_feasibility:
        orchestrator.llm_feasibility_extractor = None
    if not config.use_vlm_tables:
        orchestrator.vlm_table_extractor = None

    results = {
        "input_file": str(pdf_path),
        "config": config.to_dict(),
        "timestamp": datetime.now().isoformat(),
        "extractors_run": [],
        "results": {},
    }

    try:
        # Parse PDF first (always needed)
        if verbose:
            print("\nParsing PDF...")
        doc = orchestrator._parse_pdf(pdf_path)
        full_text = "\n".join(
            block.text for block in doc.iter_linear_blocks()
        )

        # Run enabled extractors
        if config.abbreviations:
            if not quiet:
                print("\n[Abbreviations] Extracting...")
            abbrev_results, _ = orchestrator._generate_candidates(doc)
            results["results"]["abbreviations"] = len(abbrev_results)
            results["extractors_run"].append("abbreviations")
            if verbose:
                print(f"  Found {len(abbrev_results)} candidates")

        if config.diseases:
            if not quiet:
                print("\n[Diseases] Extracting...")
            disease_results = orchestrator._process_diseases(doc, pdf_path)
            results["results"]["diseases"] = len(disease_results)
            results["extractors_run"].append("diseases")
            if verbose:
                print(f"  Found {len(disease_results)} diseases")

        if config.drugs:
            if not quiet:
                print("\n[Drugs] Extracting...")
            drug_results = orchestrator._process_drugs(doc, pdf_path)
            results["results"]["drugs"] = len(drug_results)
            results["extractors_run"].append("drugs")
            if verbose:
                print(f"  Found {len(drug_results)} drugs")

        if config.feasibility:
            if not quiet:
                print("\n[Feasibility] Extracting...")
            feas_results = orchestrator._process_feasibility(doc, pdf_path, full_text)
            results["results"]["feasibility"] = len(feas_results)
            results["extractors_run"].append("feasibility")
            if verbose:
                print(f"  Found {len(feas_results)} feasibility items")

        if config.pharma_companies:
            if not quiet:
                print("\n[Pharma] Extracting...")
            pharma_results = orchestrator._process_pharma(doc, pdf_path)
            results["results"]["pharma_companies"] = len(pharma_results)
            results["extractors_run"].append("pharma_companies")

        if config.authors:
            if not quiet:
                print("\n[Authors] Extracting...")
            author_results = orchestrator._process_authors(doc, pdf_path, full_text)
            results["results"]["authors"] = len(author_results)
            results["extractors_run"].append("authors")

        if config.citations:
            if not quiet:
                print("\n[Citations] Extracting...")
            citation_results = orchestrator._process_citations(doc, pdf_path, full_text)
            results["results"]["citations"] = len(citation_results)
            results["extractors_run"].append("citations")

        if config.document_metadata:
            if not quiet:
                print("\n[Document Metadata] Extracting...")
            metadata_results = orchestrator._process_document_metadata(
                doc, pdf_path, full_text[:5000]
            )
            results["results"]["document_metadata"] = 1 if metadata_results else 0
            results["extractors_run"].append("document_metadata")

        results["success"] = True

    except Exception as e:
        results["success"] = False
        results["error"] = str(e)
        if not quiet:
            print(f"\nError: {e}", file=sys.stderr)

    elapsed = time.time() - start_time
    results["elapsed_seconds"] = round(elapsed, 2)

    if not quiet:
        print(f"\n{'─'*40}")
        print(f"Completed in {elapsed:.2f}s")
        print(f"Results: {results['results']}")

    return results


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Collect PDF files
    pdf_files = collect_pdf_files(args.input)
    if not pdf_files:
        print("Error: No PDF files found", file=sys.stderr)
        sys.exit(1)

    # Build configuration
    config = build_config(args)

    if not args.quiet:
        print(f"Configuration: {config}")
        print(f"Files to process: {len(pdf_files)}")

    # Process each file
    all_results = []
    for pdf_path in pdf_files:
        result = run_extraction(
            pdf_path,
            config,
            verbose=args.verbose,
            quiet=args.quiet,
        )
        all_results.append(result)

    # Export combined results if requested
    if config.export_combined and config.export_json:
        output_dir = config.output_dir or Path(".")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"extraction_results_{timestamp}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        if not args.quiet:
            print(f"\nCombined results exported to: {output_file}")

    # Print summary
    if not args.quiet and len(pdf_files) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        successful = sum(1 for r in all_results if r.get("success"))
        print(f"Processed: {len(pdf_files)} files")
        print(f"Successful: {successful}")
        print(f"Failed: {len(pdf_files) - successful}")

    # Exit with error code if any failed
    if not all(r.get("success") for r in all_results):
        sys.exit(1)


if __name__ == "__main__":
    main()

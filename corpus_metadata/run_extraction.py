#!/usr/bin/env python3
# corpus_metadata/run_extraction.py
"""
Extraction runner for the corpus_metadata pipeline.

Runs extraction based on configuration in G_config/config.yaml.
The extraction_pipeline section controls which extractors run.

Usage:
    # Process a single PDF using config.yaml settings
    python run_extraction.py document.pdf

    # Process multiple PDFs
    python run_extraction.py doc1.pdf doc2.pdf

    # Process a directory
    python run_extraction.py /path/to/pdfs/

Configuration (in G_config/config.yaml):
    extraction_pipeline:
      preset: null  # Or: drugs_only, diseases_only, entities_only, etc.
      extractors:
        drugs: true
        diseases: true
        abbreviations: true
        feasibility: true
        ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from G_config.extraction_config import ExtractionConfig


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="Run extraction on PDF documents using config.yaml settings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input files
    parser.add_argument(
        "input",
        nargs="+",
        help="PDF file(s) or directory to process",
    )

    # Output options
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        help="Output directory for results (overrides config.yaml)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output",
    )

    return parser


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


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Collect PDF files
    pdf_files = collect_pdf_files(args.input)
    if not pdf_files:
        print("Error: No PDF files found", file=sys.stderr)
        sys.exit(1)

    # Load configuration from config.yaml
    config = ExtractionConfig.from_yaml()

    # Override output directory if specified
    if args.output_dir:
        config.output_dir = args.output_dir

    if not args.quiet:
        print(f"Configuration: {config}")
        print(f"Enabled extractors: {config.enabled_extractors}")
        print(f"Files to process: {len(pdf_files)}")

    # Import orchestrator here to avoid slow import on --help
    from orchestrator import Orchestrator

    orchestrator = Orchestrator(
        skip_validation=not config.use_llm_validation,
        output_dir=str(config.output_dir) if config.output_dir else None,
    )

    # Process each file
    successful = 0
    failed = 0

    for pdf_path in pdf_files:
        try:
            orchestrator.process_pdf(pdf_path)
            successful += 1
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}", file=sys.stderr)
            failed += 1

    # Print summary
    if not args.quiet and len(pdf_files) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Processed: {len(pdf_files)} files")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")

    # Exit with error code if any failed
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

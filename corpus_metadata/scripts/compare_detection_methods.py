#!/usr/bin/env python3
"""
Compare VLM-assisted detection vs heuristic detection.

Runs both detection methods on a PDF and compares:
- Detection count (tables/figures found)
- Bounding box quality (IoU overlap)
- Visual output for manual inspection
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Load .env file if present
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                value = value.strip().strip('"').strip("'")
                os.environ.setdefault(key, value)

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import fitz
from B_parsing.B13_visual_detector import detect_all_visuals, DetectorConfig
from B_parsing.B17_vlm_detector import (
    detect_visuals_vlm_document,
    compare_detections,
    VLMDetectedVisual,
)


def run_comparison(pdf_path: str, output_dir: str = None):
    """Run both detection methods and compare."""

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}")
        return

    output_dir = Path(output_dir) if output_dir else pdf_path.parent / "detection_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("VISUAL DETECTION COMPARISON")
    print("=" * 60)
    print(f"PDF: {pdf_path.name}")
    print()

    # 1. Run heuristic detection
    print("1. HEURISTIC DETECTION (Docling + PyMuPDF)")
    print("-" * 40)
    start = time.time()

    heuristic_result = detect_all_visuals(str(pdf_path), DetectorConfig())
    heuristic_time = time.time() - start

    # Convert to list of dicts for comparison
    heuristic_visuals = []
    for candidate in heuristic_result.candidates:
        heuristic_visuals.append({
            "page_num": candidate.page_num,
            "bbox_pts": candidate.bbox_pts,
            "visual_type": candidate.docling_type,
            "source": getattr(candidate, "source", "heuristic"),
        })

    print(f"  Tables detected: {heuristic_result.tables_detected}")
    print(f"  Figures detected: {heuristic_result.figures_detected}")
    print(f"  Time: {heuristic_time:.1f}s")
    print()

    # 2. Run VLM detection
    print("2. VLM DETECTION (Claude Vision)")
    print("-" * 40)
    start = time.time()

    vlm_result = detect_visuals_vlm_document(str(pdf_path))
    vlm_time = time.time() - start

    print(f"  Tables detected: {vlm_result['tables_detected']}")
    print(f"  Figures detected: {vlm_result['figures_detected']}")
    print(f"  Time: {vlm_time:.1f}s")
    print(f"  Tokens used: {vlm_result['total_tokens']}")
    print()

    # 3. Compare detections
    print("3. COMPARISON")
    print("-" * 40)

    comparison = compare_detections(
        vlm_result["visuals"],
        heuristic_visuals,
        iou_threshold=0.3,  # Lower threshold for partial matches
    )

    print(f"  Matched visuals (IoU >= 0.3): {comparison['matched_count']}")
    print(f"  VLM-only detections: {comparison['vlm_only_count']}")
    print(f"  Heuristic-only detections: {comparison['heuristic_only_count']}")
    print(f"  Average IoU of matches: {comparison['average_iou']:.2f}")
    print()

    # 4. Detailed per-visual comparison
    print("4. DETAILED RESULTS")
    print("-" * 40)

    # Group by page
    pages = set()
    for v in vlm_result["visuals"]:
        pages.add(v.page_num)
    for v in heuristic_visuals:
        pages.add(v["page_num"])

    for page_num in sorted(pages):
        vlm_on_page = [v for v in vlm_result["visuals"] if v.page_num == page_num]
        heur_on_page = [v for v in heuristic_visuals if v["page_num"] == page_num]

        print(f"\n  Page {page_num}:")
        print(f"    VLM found: {len(vlm_on_page)} | Heuristic found: {len(heur_on_page)}")

        for v in vlm_on_page:
            label = v.label or v.visual_type
            bbox = f"({v.bbox_pts[0]:.0f},{v.bbox_pts[1]:.0f},{v.bbox_pts[2]:.0f},{v.bbox_pts[3]:.0f})"
            print(f"      VLM: {label} {bbox}")

        for v in heur_on_page:
            vtype = v.get("visual_type", "unknown")
            bbox = v.get("bbox_pts", (0,0,0,0))
            bbox_str = f"({bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f})"
            print(f"      Heuristic: {vtype} {bbox_str}")

    # 5. Render comparison images
    print("\n5. GENERATING COMPARISON IMAGES")
    print("-" * 40)

    doc = fitz.open(str(pdf_path))
    try:
        for page_num in sorted(pages):
            page = doc[page_num - 1]

            # Render page
            zoom = 2.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)

            # Draw bboxes - VLM in green, heuristic in red
            shape = page.new_shape()

            # VLM detections - green
            for v in vlm_result["visuals"]:
                if v.page_num == page_num:
                    rect = fitz.Rect(v.bbox_pts)
                    shape.draw_rect(rect)
                    shape.finish(color=(0, 0.8, 0), width=2)  # Green

            # Heuristic detections - red
            for v in heuristic_visuals:
                if v["page_num"] == page_num:
                    rect = fitz.Rect(v["bbox_pts"])
                    shape.draw_rect(rect)
                    shape.finish(color=(0.8, 0, 0), width=2)  # Red

            shape.commit()

            # Render with annotations
            pix = page.get_pixmap(matrix=mat)
            img_path = output_dir / f"comparison_page{page_num}.png"
            pix.save(str(img_path))
            print(f"  Saved: {img_path.name}")

    finally:
        doc.close()

    # 6. Save results JSON
    results = {
        "pdf_path": str(pdf_path),
        "heuristic": {
            "tables": heuristic_result.tables_detected,
            "figures": heuristic_result.figures_detected,
            "time_seconds": heuristic_time,
            "visuals": [
                {
                    "page_num": v["page_num"],
                    "bbox_pts": list(v["bbox_pts"]),
                    "type": v.get("visual_type"),
                }
                for v in heuristic_visuals
            ],
        },
        "vlm": {
            "tables": vlm_result["tables_detected"],
            "figures": vlm_result["figures_detected"],
            "time_seconds": vlm_time,
            "tokens_used": vlm_result["total_tokens"],
            "visuals": [
                {
                    "page_num": v.page_num,
                    "bbox_pts": list(v.bbox_pts),
                    "type": v.visual_type,
                    "label": v.label,
                    "caption_snippet": v.caption_snippet,
                    "confidence": v.confidence,
                }
                for v in vlm_result["visuals"]
            ],
        },
        "comparison": {
            "matched": comparison["matched_count"],
            "vlm_only": comparison["vlm_only_count"],
            "heuristic_only": comparison["heuristic_only_count"],
            "average_iou": comparison["average_iou"],
        },
    }

    results_path = output_dir / "comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {results_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Heuristic: {heuristic_result.tables_detected} tables, {heuristic_result.figures_detected} figures in {heuristic_time:.1f}s")
    print(f"  VLM:       {vlm_result['tables_detected']} tables, {vlm_result['figures_detected']} figures in {vlm_time:.1f}s")
    print(f"  Agreement: {comparison['matched_count']} matched, {comparison['vlm_only_count']} VLM-only, {comparison['heuristic_only_count']} heuristic-only")
    print(f"  Output:    {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare visual detection methods")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--output", "-o", help="Output directory for comparison results")

    args = parser.parse_args()
    run_comparison(args.pdf_path, args.output)

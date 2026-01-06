# corpus_metadata/corpus_abbreviations/D_validation/D03_validation_logger.py
"""
Logger for validation results.
Writes structured logs to corpus_log directory.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from A_core.A01_domain_models import (
    Candidate,
    ExtractedEntity,
    ValidationStatus,
)


class ValidationLogger:
    """
    Logs validation results to corpus_log directory.
    
    Creates one log file per run with all validation results.
    """

    def __init__(
        self,
        log_dir: str = "corpus_log",
        run_id: Optional[str] = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Generate run ID if not provided
        self.run_id = run_id or datetime.utcnow().strftime("VAL_%Y%m%d_%H%M%S")

        # Log file path
        self.log_file = self.log_dir / f"{self.run_id}.jsonl"

        # In-memory buffer for batch writing
        self._buffer: List[Dict[str, Any]] = []

        # Stats
        self.stats = {
            "total": 0,
            "validated": 0,
            "rejected": 0,
            "ambiguous": 0,
            "errors": 0,
        }

    def log_validation(
        self,
        candidate: Candidate,
        entity: ExtractedEntity,
        llm_response: Optional[Union[Dict[str, Any], str]] = None,
        elapsed_ms: Optional[float] = None,
    ) -> None:
        """
        Log a single validation result.
        """
        self.stats["total"] += 1

        if entity.status == ValidationStatus.VALIDATED:
            self.stats["validated"] += 1
        elif entity.status == ValidationStatus.REJECTED:
            self.stats["rejected"] += 1
        elif entity.status == ValidationStatus.AMBIGUOUS:
            self.stats["ambiguous"] += 1

        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "run_id": self.run_id,
            "candidate_id": str(candidate.id),
            "entity_id": str(entity.id),
            "doc_id": candidate.doc_id,
            "short_form": candidate.short_form,
            "long_form": candidate.long_form,
            "field_type": candidate.field_type.value,
            "generator_type": candidate.generator_type.value,
            "initial_confidence": candidate.initial_confidence,
            "validation": {
                "status": entity.status.value,
                "confidence_score": entity.confidence_score,
                "rejection_reason": entity.rejection_reason,
                "corrected_long_form": entity.long_form if entity.long_form != candidate.long_form else None,
                "validation_flags": entity.validation_flags,
            },
            "context_snippet": (candidate.context_text or "")[:300],
            "page_num": candidate.context_location.page_num if candidate.context_location else None,
            "llm_response": llm_response,
            "elapsed_ms": elapsed_ms,
        }

        self._buffer.append(record)

        # Write to file (append mode) with pretty formatting
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str, indent=2) + "\n")
            f.write("---\n")  # Visual separator between records

    def log_error(
        self,
        candidate: Candidate,
        error: str,
        error_type: str = "validation_error",
    ) -> None:
        """
        Log a validation error.
        """
        self.stats["errors"] += 1
        self.stats["total"] += 1

        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "run_id": self.run_id,
            "candidate_id": str(candidate.id),
            "doc_id": candidate.doc_id,
            "short_form": candidate.short_form,
            "long_form": candidate.long_form,
            "error": error,
            "error_type": error_type,
        }

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str, indent=2) + "\n")
            f.write("---\n")  # Visual separator between records

    def write_summary(self) -> Path:
        """
        Write summary statistics to a separate file.
        Returns path to summary file.
        """
        summary_file = self.log_dir / f"{self.run_id}_summary.json"

        summary = {
            "run_id": self.run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "log_file": str(self.log_file),
            "stats": self.stats,
            "validation_rate": (
                self.stats["validated"] / self.stats["total"]
                if self.stats["total"] > 0
                else 0.0
            ),
            "rejection_rate": (
                self.stats["rejected"] / self.stats["total"]
                if self.stats["total"] > 0
                else 0.0
            ),
        }

        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        return summary_file

    def print_summary(self) -> None:
        """
        Print summary to console.
        """
        total = self.stats["total"]
        print(f"\n{'='*50}")
        print(f"VALIDATION SUMMARY - {self.run_id}")
        print(f"{'='*50}")
        print(f"Total candidates:  {total}")
        print(f"  [OK] Validated:     {self.stats['validated']} ({self.stats['validated']/total*100:.1f}%)" if total else "  [OK] Validated:     0")
        print(f"  [X] Rejected:      {self.stats['rejected']} ({self.stats['rejected']/total*100:.1f}%)" if total else "  [X] Rejected:      0")
        print(f"  ? Ambiguous:     {self.stats['ambiguous']} ({self.stats['ambiguous']/total*100:.1f}%)" if total else "  ? Ambiguous:     0")
        print(f"  [WARN] Errors:        {self.stats['errors']}")
        print(f"\nLog file: {self.log_file}")
        print(f"{'='*50}\n")
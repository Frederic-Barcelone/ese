# corpus_metadata/orchestrator_utils.py
"""
Utility classes and functions for the orchestrator.

Extracted from orchestrator.py to reduce file size.
Provides:
- StageTimer: Pipeline stage timing and summary
- Warning suppression setup
"""
from __future__ import annotations

import os
import re
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


# =============================================================================
# WARNING SUPPRESSION (must be configured before library imports)
# =============================================================================

def setup_warning_suppression() -> None:
    """Configure warning suppression for the pipeline.

    Suppresses various warnings from ML/NLP libraries to keep output clean.
    Should be called before importing heavy libraries like spacy, transformers.
    """
    # Suppress at environment level for subprocesses
    os.environ["PYTHONWARNINGS"] = (
        "ignore::UserWarning,ignore::FutureWarning,ignore::DeprecationWarning"
    )
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    _WARNING_FILTERS = [
        (UserWarning, r".*W036.*"),
        (UserWarning, r".*matcher.*does not have any patterns.*"),
        (UserWarning, r".*InconsistentVersionWarning.*"),
        (UserWarning, r".*max_size.*deprecated.*"),
        (FutureWarning, r".*max_size.*deprecated.*"),
        (DeprecationWarning, r".*max_size.*"),
    ]

    for _cat, _pat in _WARNING_FILTERS:
        warnings.filterwarnings("ignore", category=_cat, message=_pat)

    warnings.filterwarnings("ignore", module=r"sklearn\.base")
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"spacy\.language")
    warnings.filterwarnings("ignore", category=UserWarning, module=r"transformers")
    warnings.filterwarnings("ignore", message=r".*Could not get FontBBox.*")  # pdfminer.six


@dataclass
class StageTimer:
    """Track timing for pipeline stages."""

    timings: Dict[str, float] = field(default_factory=dict)
    _start_times: Dict[str, float] = field(default_factory=dict)

    def start(self, stage: str) -> None:
        """Start timing a stage."""
        self._start_times[stage] = time.time()

    def stop(self, stage: str) -> float:
        """Stop timing a stage and return elapsed time."""
        if stage not in self._start_times:
            return 0.0
        elapsed = time.time() - self._start_times[stage]
        self.timings[stage] = elapsed
        return elapsed

    def get(self, stage: str) -> float:
        """Get elapsed time for a stage."""
        return self.timings.get(stage, 0.0)

    def total(self) -> float:
        """Get total time across all stages."""
        return sum(self.timings.values())

    def print_summary(self) -> None:
        """Print timing summary."""
        if not self.timings:
            return

        total = self.total()
        print(f"\n{'─' * 50}")
        print("⏱  TIMING SUMMARY")
        print(f"{'─' * 50}")

        # Sort by execution order (approximate by order added)
        for stage, elapsed in self.timings.items():
            pct = (elapsed / total * 100) if total > 0 else 0
            bar_len = int(pct / 5)  # 20 chars max
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"  {stage:<25} {elapsed:>6.1f}s  {bar} {pct:>5.1f}%")

        print(f"{'─' * 50}")
        print(f"  {'TOTAL':<25} {total:>6.1f}s")
        print(f"{'─' * 50}")


class TeeWriter:
    """Wraps sys.stdout to duplicate output to a log file (ANSI-stripped)."""

    _ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

    def __init__(self, log_path: Path) -> None:
        self._original = sys.stdout
        self._file = open(log_path, "w", encoding="utf-8")  # noqa: SIM115

    def write(self, text: str) -> int:
        self._original.write(text)
        self._file.write(self._ANSI_RE.sub("", text))
        return len(text)

    def flush(self) -> None:
        self._original.flush()
        self._file.flush()

    def isatty(self) -> bool:
        return self._original.isatty()

    def close(self) -> None:
        self._file.close()

    def __getattr__(self, name: str) -> object:
        return getattr(self._original, name)


def activate_tee(log_dir: Path) -> TeeWriter:
    """Activate stdout tee to duplicate console output to a timestamped log file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tee = TeeWriter(log_dir / f"pipeline_run_{ts}.log")
    sys.stdout = tee
    return tee


def deactivate_tee(tee: Optional[TeeWriter]) -> None:
    """Restore original stdout and close the tee log file."""
    if tee is not None:
        sys.stdout = tee._original
        tee.close()


__all__ = [
    "StageTimer",
    "TeeWriter",
    "activate_tee",
    "deactivate_tee",
    "setup_warning_suppression",
]

# corpus_metadata/Z_utils/Z11_console_output.py
"""
Console output formatting utilities with ANSI colors.

Provides:
- ANSI color codes for terminal output
- Step printing with consistent formatting
- Progress indicators
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional


# =============================================================================
# ANSI COLOR CODES
# =============================================================================


@dataclass(frozen=True)
class Colors:
    """ANSI color codes for terminal output."""

    # Reset
    RESET = "\033[0m"

    # Regular colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Styles
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"

    @classmethod
    def supports_color(cls) -> bool:
        """Check if the terminal supports ANSI colors."""
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


# Singleton instance
C = Colors()

# Check color support once at module load
_COLOR_ENABLED = Colors.supports_color()


def _c(color: str, text: str) -> str:
    """Apply color to text if colors are enabled."""
    if _COLOR_ENABLED:
        return f"{color}{text}{C.RESET}"
    return text


# =============================================================================
# STEP FORMATTING
# =============================================================================


class StepPrinter:
    """
    Consistent step printing with colors.

    Usage:
        printer = StepPrinter(total_steps=16)
        printer.step("Parsing PDF...")
        printer.detail("Pages: 12")
        printer.time(88.2)
        printer.skip("Disease detection", "disabled in config")
    """

    def __init__(self, total_steps: int = 16):
        self.total_steps = total_steps
        self.current_step = 0

    def step(self, description: str, step_num: Optional[int] = None) -> None:
        """Print a step header."""
        if step_num is not None:
            self.current_step = step_num
        else:
            self.current_step += 1

        step_str = f"[{self.current_step}/{self.total_steps}]"
        colored_step = _c(C.BRIGHT_CYAN, step_str)
        colored_desc = _c(C.BOLD, description)
        print(f"\n{colored_step} {colored_desc}")

    def detail(self, text: str, indent: int = 2) -> None:
        """Print a detail line."""
        prefix = " " * indent
        print(f"{prefix}{text}")

    def detail_highlight(self, label: str, value: str, indent: int = 2) -> None:
        """Print a detail line with highlighted value."""
        prefix = " " * indent
        colored_value = _c(C.BRIGHT_WHITE, str(value))
        print(f"{prefix}{label}: {colored_value}")

    def time(self, seconds: float, indent: int = 2) -> None:
        """Print timing information."""
        prefix = " " * indent
        time_str = f"{seconds:.1f}s"
        colored_time = _c(C.BRIGHT_GREEN, time_str)
        print(f"{prefix}{_c(C.DIM, '⏱')}  {colored_time}")

    def skip(self, name: str, reason: str = "disabled in config") -> None:
        """Print a skipped step."""
        self.current_step += 1
        step_str = f"[{self.current_step}/{self.total_steps}]"
        colored_step = _c(C.BRIGHT_BLACK, step_str)
        colored_name = _c(C.DIM, name)
        colored_reason = _c(C.BRIGHT_BLACK, f"SKIPPED ({reason})")
        print(f"\n{colored_step} {colored_name} {colored_reason}")

    def success(self, text: str, indent: int = 2) -> None:
        """Print a success message."""
        prefix = " " * indent
        colored_check = _c(C.BRIGHT_GREEN, "✓")
        print(f"{prefix}{colored_check} {text}")

    def warning(self, text: str, indent: int = 2) -> None:
        """Print a warning message."""
        prefix = " " * indent
        colored_warn = _c(C.BRIGHT_YELLOW, "[WARN]")
        print(f"{prefix}{colored_warn} {text}")

    def error(self, text: str, indent: int = 2) -> None:
        """Print an error message."""
        prefix = " " * indent
        colored_err = _c(C.BRIGHT_RED, "[ERROR]")
        print(f"{prefix}{colored_err} {text}")

    def header(self, text: str, char: str = "=", width: int = 60) -> None:
        """Print a section header."""
        line = char * width
        colored_line = _c(C.BRIGHT_CYAN, line)
        colored_text = _c(C.BOLD + C.BRIGHT_WHITE, text)
        print(f"\n{colored_line}")
        print(colored_text)
        print(colored_line)

    def subheader(self, text: str, char: str = "-", width: int = 60) -> None:
        """Print a subsection header."""
        line = char * width
        colored_line = _c(C.DIM, line)
        colored_text = _c(C.BRIGHT_WHITE, text)
        print(f"\n{colored_line}")
        print(colored_text)
        print(colored_line)

    def result(self, label: str, count: int, indent: int = 4) -> None:
        """Print a result count."""
        prefix = " " * indent
        colored_count = _c(C.BRIGHT_YELLOW, str(count))
        print(f"{prefix}+ {label}: {colored_count}")


# Global printer instance
_printer: Optional[StepPrinter] = None


def get_printer(total_steps: int = 16) -> StepPrinter:
    """Get or create the global step printer."""
    global _printer
    if _printer is None or _printer.total_steps != total_steps:
        _printer = StepPrinter(total_steps)
    return _printer


def reset_printer() -> None:
    """Reset the global printer."""
    global _printer
    _printer = None


__all__ = [
    "Colors",
    "C",
    "StepPrinter",
    "get_printer",
    "reset_printer",
]

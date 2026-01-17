# corpus_metadata/corpus_metadata/C_generators/C01_strategy_abbrev.py
"""
The Syntax Matcher - Schwartz-Hearst for abbreviation extraction.

Target: Explicit definitions of short forms in running text.

Strategies:
  1. Schwartz-Hearst: "Tumor Necrosis Factor (TNF)" -> LF (SF)
  2. Reverse explicit: "TNF (Tumor Necrosis Factor)" -> SF (LF)
  3. Implicit phrasing: "TNF, defined as Tumor Necrosis Factor"

References:
  - Schwartz & Hearst (2003) "A Simple Algorithm for Identifying
    Abbreviation Definitions in Biomedical Text"
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from A_core.A01_domain_models import (
    Candidate,
    Coordinate,
    FieldType,
    GeneratorType,
    ProvenanceMetadata,
)
from A_core.A02_interfaces import BaseCandidateGenerator
from A_core.A03_provenance import (
    generate_run_id,
    get_git_revision_hash,
)
from B_parsing.B02_doc_graph import (
    ContentRole,
    DocumentGraph,
)


# ----------------------------
# Helpers (pure functions)
# ----------------------------

_ABBREV_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-\+/().]{0,14}$")

# Pattern to detect author initial contexts
# Matches patterns like "John Smith1,2, Jane Doe3" or "A. Smith, B. Jones"
_AUTHOR_LIST_PATTERN = re.compile(
    r"(?:[A-Z][a-z]+\s+[A-Z][a-z]+\s*[0-9,\s]*,?\s*){2,}",  # Multiple "First Last" names
    re.MULTILINE,
)

# Pattern for initials in author names (e.g., "J.H." or "JH" after a name)
_AUTHOR_INITIAL_PATTERN = re.compile(
    r"[A-Z][a-z]+\s+(?:[A-Z]\.?\s*)+[,\s0-9]"  # Name followed by initials
)


def _is_likely_author_initial(sf: str, context: str) -> bool:
    """
    Check if a 2-letter uppercase abbreviation is likely an author initial.

    Author initials appear in contexts like:
    - "John Smith1,2, BH contributed to..." (where BH = initials)
    - Author contribution sections
    - Name lists with superscript affiliations

    Returns True if the SF is likely an author initial (should be filtered).
    """
    sf = (sf or "").strip()
    ctx = (context or "").strip()

    if not sf or not ctx:
        return False

    # Only check 2-letter all-uppercase abbreviations (typical author initials)
    if len(sf) != 2 or not sf.isupper():
        return False

    # Common author initial patterns - 2 uppercase letters that look like initials
    # These are very common in academic papers and rarely actual abbreviations
    COMMON_AUTHOR_INITIALS = {
        # Very common initial combinations
        "BH", "JH", "JM", "DB", "LS", "LH", "RB", "KH", "MH", "PH",
        "DJ", "MJ", "RJ", "SJ", "JL", "ML", "PL", "JW", "MW", "RW",
        "JA", "MA", "RA", "SA", "JB", "MB", "RB", "SB", "JC", "MC",
        "RC", "SC", "JD", "MD", "RD", "SD", "JE", "ME", "RE", "SE",
        "JF", "MF", "RF", "SF", "JG", "MG", "RG", "SG", "JK", "MK",
        "RK", "SK", "JN", "MN", "RN", "SN", "JP", "MP", "RP", "SP",
        "JR", "MR", "JT", "MT", "RT", "ST", "JV", "MV", "RV", "SV",
        # Authors in the AAV guidelines doc
        "RP", "FL", "LG", "CB", "MB", "LD", "DH", "DM", "PH", "CJ",
        "JLU", "AM", "PM", "JMN", "AT", "JV", "LB", "FB", "DW", "RW",
        "TH", "AS", "JS", "CS", "PS", "KS", "NS", "BS", "WS", "MS",
    }

    # Check if it's a common author initial pattern
    if sf in COMMON_AUTHOR_INITIALS:
        # Look for author context clues
        ctx_lower = ctx.lower()
        author_clues = [
            "contributed", "author", "wrote", "drafted", "reviewed",
            "approved", "manuscript", "acknowledgement", "funding",
            "conflict", "interest", "affiliation", "department",
            "university", "hospital", "medical center", "school of",
            "corresponding", "equal contribution",
        ]
        if any(clue in ctx_lower for clue in author_clues):
            return True

        # Check for author list patterns (names with superscript numbers)
        # e.g., "John Smith1,2, Jane Doe3, BH"
        if re.search(r"[A-Z][a-z]+\s+[A-Z][a-z]+\s*[0-9,]+", ctx):
            return True

        # Check if surrounded by other 2-letter initials (author list)
        # e.g., "BH, JH, MK contributed..."
        initials_nearby = re.findall(r"\b[A-Z]{2}\b", ctx)
        if len(initials_nearby) >= 3:  # Multiple 2-letter sequences = likely author list
            return True

    return False


def _clean_ws(s: str) -> str:
    return " ".join((s or "").split()).strip()


def _looks_like_short_form(sf: str, min_len: int = 2, max_len: int = 10) -> bool:
    sf = (sf or "").strip()
    if len(sf) < min_len or len(sf) > max_len:
        return False
    if sf.isdigit():
        return False

    # Handle space-separated abbreviations (e.g., "CC BY", "IL 2", "Type I")
    if " " in sf:
        # Allow only if it's 2-3 short uppercase tokens (like "CC BY", "IL 2")
        parts = sf.split()
        if len(parts) > 3:
            return False
        # Each part should be short and mostly uppercase/alphanumeric
        for part in parts:
            if len(part) > 6:  # Each token max 6 chars
                return False
            if not any(ch.isupper() for ch in part) and not part.isdigit():
                return False
        # Total length check (already done above, but be explicit)
        if len(sf) > max_len + 3:  # Allow slightly longer for spaces
            return False
        return True

    if not _ABBREV_TOKEN_RE.match(sf):
        return False
    # Must contain at least one uppercase letter (strong pharma heuristic)
    if not any(ch.isupper() for ch in sf):
        return False
    return True


def _truncate_at_breaks(text: str) -> str:
    """
    For implicit LF captures: cut at punctuation / obvious clause starters.

    Improved to reduce false positives by stopping at:
    - Punctuation marks
    - Relative clause starters (which, that, who, etc.)
    - Verb phrases that indicate narrative rather than definition
    - Trailing connectors and articles
    """
    t = (text or "").strip()

    # Stop at punctuation
    t = re.split(r"[.;:\n\r\)\]]", t, maxsplit=1)[0].strip()

    # Stop at relative clause starters
    t = re.split(
        r"\b(which|that|who|where|when|while|whose)\b",
        t,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()

    # Stop at verb phrases that indicate narrative/commentary rather than definition
    # e.g., "enzyme was evaluated" -> "enzyme"
    # e.g., "receptor is a subtype" -> "receptor"
    t = re.split(
        r"\b(was|were|is|are|has|have|had|being|been|can|could|would|should|may|might)\s+",
        t,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()

    # Stop at common non-definition patterns
    # e.g., "protein also known as" -> handled by implicit patterns, not here
    t = re.split(
        r"\b(also|previously|formerly|sometimes|commonly|often)\b",
        t,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()

    # Drop trailing connector words and articles
    t = re.sub(
        r"\b(and|or|as|by|the|a|an|of|in|for|to)\s*$", "", t, flags=re.IGNORECASE
    ).strip()

    # Drop trailing punctuation that might have been left
    t = re.sub(r"[,\-]\s*$", "", t).strip()

    return _clean_ws(t)


def _context_window(text: str, start: int, end: int, window: int = 240) -> str:
    """
    Create a short context window around a match.
    """
    if not text:
        return ""
    s = max(0, start - window)
    e = min(len(text), end + window)
    return _clean_ws(text[s:e])


def _looks_like_measurement(text: str) -> bool:
    """
    Check if text looks like a measurement value rather than a definition.
    Filters false positives like "11.06 mg/L (UACR-I)".
    """
    if not text:
        return False

    text = text.strip()

    # Define measurement units (including compound units like mg/L, g/dL)
    units = r"(?:mg|g|L|mL|μg|ng|IU|%|°C|mmol|μmol|pg|kg|mm|cm|m|s|min|h|d|Hz|kDa|Da)"
    compound_unit = units + r"(?:/" + units + r")?"

    # Reject if ends with a measurement pattern: "NUMBER UNIT"
    # e.g., "11.06 mg/L", "76.79 mg/L", "100 %", "1.3 × 10^9/L"
    measurement_end_pattern = re.compile(
        r"\d+(?:\.\d+)?\s*(?:×\s*10[\^]?\d+)?/?\s*" + compound_unit + r"\s*$",
        re.IGNORECASE,
    )
    if measurement_end_pattern.search(text):
        return True

    # Reject if mostly numeric content (more digits than letters)
    letters = sum(1 for c in text if c.isalpha())
    digits = sum(1 for c in text if c.isdigit())
    if digits > 0 and digits >= letters:
        return True

    # Reject if looks like a numeric list/sequence
    if re.match(r"^[\d.,×\s\-/]+$", text):
        return True

    return False


def _space_sf_extract(short_form: str, preceding_text: str) -> Optional[str]:
    """
    Extract long form for space-separated abbreviations like "CC BY".

    Strategy: Match each letter in the SF to word initials in the preceding text.

    E.g., "CC BY" with preceding "Creative Commons Attribution":
    - C -> Creative, C -> Commons, B -> (no match), Y -> (no match)
    - But we find 2 matches for CC, which is good enough
    - Result: "Creative Commons Attribution" (take words that cover the matches)
    """
    sf = (short_form or "").strip()
    txt = (preceding_text or "").strip()

    if not sf or not txt or " " not in sf:
        return None

    # Get SF letters (uppercase, no spaces)
    sf_letters = [c.upper() for c in sf if c.isalpha()]
    if len(sf_letters) < 2:
        return None

    # Extract words from preceding text (filter out empty/punctuation-only)
    words = [w for w in txt.split() if w and any(c.isalpha() for c in w)]
    if len(words) < 2:
        return None

    # Find the best starting position by matching SF letters to word initials
    # Work backwards from the end of the text
    best_start = -1
    best_matches = 0

    for start_idx in range(len(words)):
        # Try to match SF letters starting from this word
        matches = 0
        sf_idx = 0
        for word_idx in range(start_idx, len(words)):
            if sf_idx >= len(sf_letters):
                break
            word_initial = words[word_idx][0].upper() if words[word_idx] else ""
            if word_initial == sf_letters[sf_idx]:
                matches += 1
                sf_idx += 1

        # Prefer matches that start with the first SF letter
        if matches > best_matches:
            # Verify first letter alignment
            word_initial = words[start_idx][0].upper() if words[start_idx] else ""
            if word_initial == sf_letters[0]:
                best_matches = matches
                best_start = start_idx

    # Need at least 2 letter matches for space-SF
    if best_matches < 2 or best_start < 0:
        return None

    # Extract from best_start to end
    lf = " ".join(words[best_start:])

    # Basic validation
    if 5 <= len(lf) <= 80:
        return lf

    return None


def _word_initial_extract(short_form: str, preceding_text: str) -> Optional[str]:
    """
    Word-initial letter matching for acronyms.
    Matches SF letters to the first letter of each word in preceding text.

    Works well for: "Acanthosis nigricans (AN)" where A=Acanthosis, N=nigricans
    """
    sf = re.sub(r"[^A-Za-z0-9]", "", short_form or "")
    if len(sf) < 2:
        return None

    txt = (preceding_text or "").rstrip()
    if not txt:
        return None

    # Extract words (alphanumeric sequences)
    words = re.findall(r"[A-Za-z][A-Za-z0-9]*", txt)
    if len(words) < len(sf):
        return None

    # Try to match SF letters to word-initial letters (backwards from end of words)
    # Look for consecutive words whose initials spell the SF
    sf_lower = sf.lower()

    # Sliding window: try to find len(sf) consecutive words matching SF
    for start_idx in range(len(words) - len(sf), -1, -1):
        candidate_words = words[start_idx : start_idx + len(sf)]
        initials = "".join(w[0].lower() for w in candidate_words)

        if initials == sf_lower:
            # Found a match! Build the long form
            lf = " ".join(candidate_words)

            # Validate: first letter must match
            if lf[0].lower() != sf[0].lower():
                continue

            # Avoid absurdly long LFs
            if len(lf) > 120:
                continue

            # Reject measurements
            if _looks_like_measurement(lf):
                continue

            return lf

    return None


def _schwartz_hearst_extract(short_form: str, preceding_text: str) -> Optional[str]:
    """
    Schwartz-Hearst-ish backward character alignment.
    Returns a best-effort long form (LF) extracted from preceding_text, or None.

    Works well for: "... Tumor Necrosis Factor (TNF)"

    Now tries word-initial matching first (better for disease acronyms),
    then falls back to classic character alignment.
    """
    # Try word-initial matching first (works better for disease acronyms)
    lf = _word_initial_extract(short_form, preceding_text)
    if lf:
        return lf

    # Fall back to classic Schwartz-Hearst character alignment
    sf = re.sub(r"[^A-Za-z0-9]", "", short_form or "")
    if len(sf) < 2:
        return None

    # Limit lookback window (characters)
    txt = (preceding_text or "").rstrip()
    if not txt:
        return None

    window_size = min(len(txt), len(sf) * 18 + 30)
    txt = txt[-window_size:]

    i = len(sf) - 1
    j = len(txt) - 1

    # Align SF chars backwards into txt backwards
    while i >= 0:
        c = sf[i].lower()
        while j >= 0 and txt[j].lower() != c:
            j -= 1
        if j < 0:
            return None
        i -= 1
        j -= 1

    # j is now before the earliest matched character; pick LF start at word boundary
    start = txt.rfind(" ", 0, j + 1) + 1
    lf = _clean_ws(txt[start:])

    if not lf:
        return None

    # Strong safety: first letter alignment (prevents many false positives)
    if lf[0].lower() != short_form[0].lower():
        return None

    # Avoid absurdly long LFs
    if len(lf) > 120:
        return None

    # Reject measurement values (e.g., "11.06 mg/L" is not a definition)
    if _looks_like_measurement(lf):
        return None

    return lf


def _extract_preceding_name(short_form: str, preceding_text: str) -> Optional[str]:
    """
    Extract a drug/compound name immediately before parentheses.

    Handles patterns like "iptacopan (LNP023)" where:
    - The SF is an alphanumeric code (LNP023, NCT04817618)
    - The LF is a proper name/word directly before the parentheses

    This is different from Schwartz-Hearst because the code letters
    don't need to appear in the name.
    """
    sf = (short_form or "").strip()
    txt = (preceding_text or "").rstrip()

    if not sf or not txt:
        return None

    # SF should look like an alphanumeric code (contains letters AND digits)
    has_letter = any(c.isalpha() for c in sf)
    has_digit = any(c.isdigit() for c in sf)
    is_compound_id = has_letter and has_digit and len(sf) >= 4

    if not is_compound_id:
        return None

    # Extract the last word before parentheses
    # Match a word that's either:
    # - A proper noun (capitalized) like "Iptacopan"
    # - A lowercase word like "iptacopan"
    # But NOT all-caps (that would be another abbreviation)
    word_match = re.search(r"([A-Za-z][a-z]{2,20})\s*$", txt)
    if not word_match:
        return None

    candidate_lf = word_match.group(1)

    # Skip if the candidate is too short or too long
    if len(candidate_lf) < 3 or len(candidate_lf) > 25:
        return None

    # Skip if it looks like a common word (articles, prepositions)
    skip_words = {
        "the", "and", "for", "with", "from", "that", "this",
        "which", "where", "when", "while", "being", "both"
    }
    if candidate_lf.lower() in skip_words:
        return None

    return candidate_lf


def _validate_sf_in_lf(short_form: str, long_form: str) -> bool:
    """
    Lightweight validation that SF letters appear in LF (in order, not necessarily contiguous).
    This is used for:
      - validating implicit captures
      - validating reverse explicit pattern: "SF (Long Form)"
    """
    sf = re.sub(r"[^A-Za-z0-9]", "", short_form or "")
    lf = re.sub(r"[^A-Za-z0-9 ]", " ", long_form or "")
    lf = _clean_ws(lf)

    if not sf or not lf:
        return False

    if lf[0].lower() != short_form[0].lower():
        return False

    # In-order scan across LF
    sf_idx = 0
    for ch in lf:
        if ch.lower() == sf[sf_idx].lower():
            sf_idx += 1
            if sf_idx == len(sf):
                return True

    return False


# ----------------------------
# Generator 1: Explicit + Implicit (syntax)
# ----------------------------


class AbbrevSyntaxCandidateGenerator(BaseCandidateGenerator):
    """
    Abbreviation extraction from running text.

    Strategy A (Explicit):  Long Form (SF)
    Strategy B (Explicit):  SF (Long Form)
    Strategy C (Implicit):  SF, defined as/stands for/... Long Form
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        self.min_sf_length = int(self.config.get("min_sf_length", 2))
        self.max_sf_length = int(self.config.get("max_sf_length", 10))

        # Capture any parentheses content; we decide SF vs LF by heuristics
        self.parens_any = re.compile(r"\(([^)]+)\)")

        # Implicit phrasing patterns (case-insensitive phrases, SF stays strict-ish)
        # LF capture is non-greedy and stops before punctuation via lookahead.
        # Use lookahead instead of \b for end boundary to handle hyphenated abbreviations (e.g., APPEAR-C3G)
        self.implicit_patterns = [
            re.compile(
                r"(?<![A-Za-z0-9])([A-Z][A-Za-z0-9\-\+/().]{1,14})(?=[,\s])\s*,?\s*(?:defined\s+as)\s+(.{5,160}?)(?=[.;:\n\r\)\]]|$)",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?<![A-Za-z0-9])([A-Z][A-Za-z0-9\-\+/().]{1,14})(?=[,\s])\s*,?\s*(?:stands\s+for)\s+(.{5,160}?)(?=[.;:\n\r\)\]]|$)",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?<![A-Za-z0-9])([A-Z][A-Za-z0-9\-\+/().]{1,14})(?=[,\s])\s*,?\s*(?:abbreviated\s+as)\s+(.{5,160}?)(?=[.;:\n\r\)\]]|$)",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?<![A-Za-z0-9])([A-Z][A-Za-z0-9\-\+/().]{1,14})(?=[,\s])\s*,?\s*(?:also\s+known\s+as)\s+(.{5,160}?)(?=[.;:\n\r\)\]]|$)",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?<![A-Za-z0-9])([A-Z][A-Za-z0-9\-\+/().]{1,14})(?=[,\s])\s*,?\s*(?:short\s+for)\s+(.{5,160}?)(?=[.;:\n\r\)\]]|$)",
                re.IGNORECASE,
            ),
        ]

        # Explicit inline definition patterns: "SF=long form" or "SF: long form"
        # Captures patterns like "RR=relative reduction" or "UPCR: urine protein-creatinine ratio"
        self.inline_definition_patterns = [
            # SF=long form (e.g., "RR=relative reduction")
            # Uses greedy match for LF, stopping at period, comma, or uppercase letter
            re.compile(
                r"(?<![A-Za-z0-9])([A-Z][A-Za-z0-9\-]{1,10})\s*[=:]\s*([a-z][a-z\s\-/]{3,60})(?=[.,;)\]A-Z]|$)",
            ),
        ]

        self.context_window_chars = int(self.config.get("context_window_chars", 400))
        self.carryover_chars = int(
            self.config.get("carryover_chars", 140)
        )  # helps cross-block edge cases
        self.max_candidates_per_block = int(
            self.config.get("max_candidates_per_block", 200)
        )

        # Provenance defaults (prefer orchestrator to pass these in)
        self.pipeline_version = str(
            self.config.get("pipeline_version") or get_git_revision_hash()
        )
        self.run_id = str(self.config.get("run_id") or generate_run_id("ABBR"))
        self.doc_fingerprint_default = str(
            self.config.get("doc_fingerprint") or "unknown-doc-fingerprint"
        )

    @property
    def generator_type(self) -> GeneratorType:
        return GeneratorType.SYNTAX_PATTERN

    def extract(self, doc_structure: DocumentGraph) -> List[Candidate]:
        doc = doc_structure
        out: List[Candidate] = []
        seen: Set[Tuple[str, str, str]] = set()

        prev_tail = ""

        for block in doc.iter_linear_blocks(skip_header_footer=True):
            # Avoid extracting inside section headers; keep those for context elsewhere
            if block.role == ContentRole.SECTION_HEADER:
                prev_tail = _clean_ws(block.text)[-self.carryover_chars :]
                continue

            text_block = _clean_ws(block.text)
            if not text_block:
                prev_tail = ""
                continue

            combined = (
                (prev_tail + " " + text_block).strip() if prev_tail else text_block
            )

            added_this_block = 0

            # -------------------------
            # Strategy A/B: Parentheses explicit
            # -------------------------
            for m in self.parens_any.finditer(combined):
                if added_this_block >= self.max_candidates_per_block:
                    break

                inside = _clean_ws(m.group(1))
                if not inside:
                    continue

                # Prefer the content inside parens as SF if it looks like SF
                if _looks_like_short_form(
                    inside, self.min_sf_length, self.max_sf_length
                ):
                    sf = inside
                    preceding = combined[: m.start()]

                    # Skip author initials (e.g., "BH", "JM" in author lists)
                    if _is_likely_author_initial(sf, combined):
                        continue

                    # Try Schwartz-Hearst first (works for single-token SFs)
                    lf = _schwartz_hearst_extract(sf, preceding)

                    # Fallback for space-separated SFs like "CC BY"
                    if not lf and " " in sf:
                        lf = _space_sf_extract(sf, preceding)

                    # HIGH PRIORITY FIX: Handle compound IDs like "iptacopan (LNP023)"
                    # where the code doesn't spell out the drug name
                    if not lf:
                        lf = _extract_preceding_name(sf, preceding)

                    if not lf:
                        continue

                    key = (doc.doc_id, sf.upper(), lf.lower())
                    if key in seen:
                        continue
                    seen.add(key)

                    method = "explicit_lf_sf" if " " not in sf else "explicit_space_sf"
                    out.append(
                        self._make_candidate(
                            doc,
                            block,
                            sf,
                            lf,
                            combined,
                            m.start(),
                            m.end(),
                            method=method,
                            confidence=0.95 if " " in sf else 0.98,
                        )
                    )
                    added_this_block += 1
                    continue

                # Otherwise, treat as possible LF and look for a preceding SF token
                lf_candidate = inside
                preceding = combined[: m.start()].rstrip()

                # Grab last token before "(" as SF
                prev_token_match = re.search(
                    r"([A-Za-z0-9\-\+/().]{2,15})\s*$", preceding
                )
                if not prev_token_match:
                    continue
                sf_candidate = prev_token_match.group(1)

                if not _looks_like_short_form(
                    sf_candidate, self.min_sf_length, self.max_sf_length
                ):
                    continue

                # Skip author initials (e.g., "BH", "JM" in author lists)
                if _is_likely_author_initial(sf_candidate, combined):
                    continue

                lf_clean = _truncate_at_breaks(lf_candidate)
                if not lf_clean:
                    continue

                if not _validate_sf_in_lf(sf_candidate, lf_clean):
                    continue

                key = (doc.doc_id, sf_candidate.upper(), lf_clean.lower())
                if key in seen:
                    continue
                seen.add(key)

                out.append(
                    self._make_candidate(
                        doc,
                        block,
                        sf_candidate,
                        lf_clean,
                        combined,
                        m.start(),
                        m.end(),
                        method="explicit_sf_lf",
                        confidence=0.96,
                    )
                )
                added_this_block += 1

            # -------------------------
            # Strategy C: Implicit phrasing
            # -------------------------
            for pat in self.implicit_patterns:
                if added_this_block >= self.max_candidates_per_block:
                    break

                for m in pat.finditer(
                    text_block
                ):  # use block text for implicit; avoids cross-block weirdness
                    if added_this_block >= self.max_candidates_per_block:
                        break

                    sf = _clean_ws(m.group(1))
                    raw_lf = _clean_ws(m.group(2))

                    if not _looks_like_short_form(
                        sf, self.min_sf_length, self.max_sf_length
                    ):
                        continue

                    # Skip author initials (e.g., "BH", "JM" in author lists)
                    if _is_likely_author_initial(sf, text_block):
                        continue

                    lf = _truncate_at_breaks(raw_lf)
                    if not lf:
                        continue

                    # validate implicit LF (prevents greedy junk)
                    if not _validate_sf_in_lf(sf, lf):
                        continue

                    key = (doc.doc_id, sf.upper(), lf.lower())
                    if key in seen:
                        continue
                    seen.add(key)

                    out.append(
                        self._make_candidate(
                            doc,
                            block,
                            sf,
                            lf,
                            text_block,
                            m.start(),
                            m.end(),
                            method="implicit_phrasing",
                            confidence=0.90,
                        )
                    )
                    added_this_block += 1

            # -------------------------
            # Strategy D: Inline definitions (SF=LF or SF: LF)
            # HIGH PRIORITY: Catches "RR=relative reduction" patterns
            # -------------------------
            for pat in self.inline_definition_patterns:
                if added_this_block >= self.max_candidates_per_block:
                    break

                for m in pat.finditer(text_block):
                    if added_this_block >= self.max_candidates_per_block:
                        break

                    sf = _clean_ws(m.group(1))
                    lf = _clean_ws(m.group(2))

                    if not _looks_like_short_form(
                        sf, self.min_sf_length, self.max_sf_length
                    ):
                        continue

                    # Skip author initials (e.g., "BH", "JM" in author lists)
                    if _is_likely_author_initial(sf, text_block):
                        continue

                    # LF should be lowercase phrase (not another abbreviation)
                    if not lf or len(lf) < 4:
                        continue

                    # Skip if LF is all uppercase (likely another abbreviation)
                    if lf.isupper():
                        continue

                    # Skip if LF contains common noise patterns
                    noise_patterns = [
                        "sponsored by", "workshop", "scientific", "facts",
                        "uncertainties", "study", "trial", "published",
                    ]
                    lf_lower = lf.lower()
                    if any(noise in lf_lower for noise in noise_patterns):
                        continue

                    # Skip if LF is too long (likely noise)
                    if len(lf) > 40:
                        continue

                    key = (doc.doc_id, sf.upper(), lf.lower())
                    if key in seen:
                        continue
                    seen.add(key)

                    out.append(
                        self._make_candidate(
                            doc,
                            block,
                            sf,
                            lf,
                            text_block,
                            m.start(),
                            m.end(),
                            method="inline_definition",
                            confidence=0.97,  # High confidence - explicit definition
                        )
                    )
                    added_this_block += 1

            # update tail for cross-block continuity
            prev_tail = (
                combined[-self.carryover_chars :] if self.carryover_chars > 0 else ""
            )

        return out

    def _make_candidate(
        self,
        doc: DocumentGraph,
        block,
        sf: str,
        lf: str,
        context_source_text: str,
        match_start: int,
        match_end: int,
        method: str,
        confidence: float,
    ) -> Candidate:
        ctx = _context_window(
            context_source_text,
            match_start,
            match_end,
            window=self.context_window_chars,
        )

        loc = Coordinate(
            page_num=int(block.page_num),
            block_id=str(block.id),
            bbox=block.bbox,
        )

        prov = ProvenanceMetadata(
            pipeline_version=self.pipeline_version,
            run_id=self.run_id,
            doc_fingerprint=str(
                self.config.get("doc_fingerprint") or self.doc_fingerprint_default
            ),
            generator_name=self.generator_type,
            # keep method here (cheap + no schema changes)
            rule_version=f"abbrev_syntax::{method}",
        )

        return Candidate(
            doc_id=doc.doc_id,
            field_type=FieldType.DEFINITION_PAIR,
            generator_type=self.generator_type,
            short_form=_clean_ws(sf),
            long_form=_clean_ws(lf),
            context_text=ctx if ctx else context_source_text,
            context_location=loc,
            initial_confidence=float(max(0.0, min(1.0, confidence))),
            provenance=prov,
        )


# Re-export for backward compatibility

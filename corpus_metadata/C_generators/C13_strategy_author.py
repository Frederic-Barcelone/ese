"""
Author and investigator mention detection using pattern matching.

This module detects author and investigator names in clinical documents using
regex patterns. Identifies names with academic credentials, role-prefixed names,
and author blocks with affiliations for provenance tracking.

Key Components:
    - AuthorDetector: Main detector for author/investigator mentions
    - Pattern categories:
        - Names with credentials (e.g., "John Smith, MD, PhD")
        - Role-prefixed names (e.g., "Principal Investigator: Jane Doe")
        - Author blocks with superscript affiliations
    - AuthorRoleType: Enum for investigator roles

Example:
    >>> from C_generators.C13_strategy_author import AuthorDetector
    >>> detector = AuthorDetector(config={})
    >>> candidates = detector.detect(doc_graph, "doc_123", "fingerprint")
    >>> for c in candidates:
    ...     print(f"{c.name} ({c.role}): {c.affiliation}")
    John Smith (PRINCIPAL_INVESTIGATOR): Harvard Medical School

Dependencies:
    - A_core.A01_domain_models: Coordinate, EvidenceSpan, ValidationStatus
    - A_core.A03_provenance: Provenance tracking utilities
    - A_core.A10_author_models: AuthorCandidate, AuthorRoleType, ExtractedAuthor
    - B_parsing.B02_doc_graph: DocumentGraph
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

from A_core.A01_domain_models import Coordinate, EvidenceSpan, ValidationStatus
from A_core.A03_provenance import generate_run_id, get_git_revision_hash
from A_core.A10_author_models import (
    AuthorCandidate,
    AuthorExportDocument,
    AuthorExportEntry,
    AuthorGeneratorType,
    AuthorProvenanceMetadata,
    AuthorRoleType,
    ExtractedAuthor,
)
from A_core.A23_doc_graph_models import DocumentGraph


class AuthorDetector:
    """
    Detect author/investigator mentions using regex patterns.

    Detects names with academic credentials, role-prefixed names,
    and author blocks in clinical documents.
    """

    # Name with credentials pattern (e.g., "John Smith, MD, PhD")
    NAME_WITH_CREDENTIALS = re.compile(
        r"([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+)"  # Name
        r"(?:,\s*(?:MD|PhD|MPH|DO|PharmD|RN|MSN|DrPH|ScD|MBBS|MBChB|FRCPC?|FACP))+",
        re.MULTILINE,
    )

    # Role-prefixed name patterns
    ROLE_PATTERNS: List[Tuple[re.Pattern, AuthorRoleType]] = [
        (
            re.compile(
                r"Principal\s+[Ii]nvestigator[s]?[:\s]+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+)",
                re.MULTILINE,
            ),
            AuthorRoleType.PRINCIPAL_INVESTIGATOR,
        ),
        (
            re.compile(
                r"Corresponding\s+[Aa]uthor[s]?[:\s]+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+)",
                re.MULTILINE,
            ),
            AuthorRoleType.CORRESPONDING_AUTHOR,
        ),
        (
            re.compile(
                r"Co-?[Ii]nvestigator[s]?[:\s]+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+)",
                re.MULTILINE,
            ),
            AuthorRoleType.CO_INVESTIGATOR,
        ),
        (
            re.compile(
                r"Study\s+[Cc]hair[:\s]+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+)",
                re.MULTILINE,
            ),
            AuthorRoleType.STUDY_CHAIR,
        ),
        (
            re.compile(
                r"Steering\s+[Cc]ommittee[:\s]+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+)",
                re.MULTILINE,
            ),
            AuthorRoleType.STEERING_COMMITTEE,
        ),
    ]

    # Email pattern for affiliation extraction
    EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")

    # ORCID pattern
    ORCID_PATTERN = re.compile(r"(?:ORCID[:\s]*)?(\d{4}-\d{4}-\d{4}-\d{3}[\dX])")

    # Author byline pattern - matches names in comma-separated lists
    # Handles: "David Kavanagh*, Andrew S Bomback*, Marina Vivarelli"
    # Also handles names with middle initials: "John A. Smith" or "John A Smith"
    AUTHOR_BYLINE_NAME = re.compile(
        r"([A-Z][a-z]+(?:\s+[A-Z](?:\.|(?=\s)))?(?:\s+[A-Z][a-z]+)+)\*?",
        re.UNICODE,
    )

    # Pattern to detect "on behalf of" investigator groups
    ON_BEHALF_PATTERN = re.compile(
        r"on\s+behalf\s+of\s+(?:the\s+)?([A-Z][A-Za-z0-9\-]+(?:\s+[A-Za-z]+)*)\s+(?:investigators?|group|consortium|committee|study\s+group)",
        re.IGNORECASE,
    )

    # Pattern to find author list sections (near beginning of document)
    AUTHOR_LIST_MARKERS = [
        r"^[A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+\*?,",  # Starts with name, comma
        r"Authors?:",
        r"Contributors?:",
        r"Writing\s+(?:group|committee)",
    ]

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        llm_client: Optional[Any] = None,
    ):
        self.config = config or {}
        self.llm_client = llm_client
        self.run_id = str(self.config.get("run_id") or generate_run_id("AUTHOR"))
        self.pipeline_version = (
            self.config.get("pipeline_version") or get_git_revision_hash()
        )

    def detect(
        self,
        doc_graph: DocumentGraph,
        doc_id: str,
        doc_fingerprint: str,
        full_text: Optional[str] = None,
    ) -> List[AuthorCandidate]:
        """
        Detect author mentions in document.

        Args:
            doc_graph: Document graph
            doc_id: Document identifier
            doc_fingerprint: Document fingerprint
            full_text: Optional pre-built full text

        Returns:
            List of AuthorCandidate objects
        """
        candidates: List[AuthorCandidate] = []
        seen_names: Set[str] = set()

        # Build full text if not provided
        if not full_text:
            full_text = " ".join(
                block.text
                for block in doc_graph.iter_linear_blocks()
                if block.text
            )

        if not full_text:
            return candidates

        # LLM-based extraction (replaces broken regex byline detection)
        if self.llm_client:
            llm_authors = self._extract_authors_llm(full_text, doc_id, doc_fingerprint)
            for c in llm_authors:
                normalized = self._normalize_name(c.full_name)
                if normalized not in seen_names:
                    seen_names.add(normalized)
                    candidates.append(c)
        else:
            # Fallback: regex byline detection (no LLM available)
            candidates.extend(
                self._detect_author_byline(full_text, doc_id, doc_fingerprint, seen_names)
            )

        # Detect role-prefixed names (higher confidence)
        for pattern, role in self.ROLE_PATTERNS:
            for match in pattern.finditer(full_text):
                name = match.group(1).strip()
                normalized_name = self._normalize_name(name)

                if normalized_name in seen_names:
                    continue
                if not self._is_valid_name(name):
                    continue

                seen_names.add(normalized_name)

                context_start = max(0, match.start() - 100)
                context_end = min(len(full_text), match.end() + 100)
                context_text = full_text[context_start:context_end]

                # Extract email if present in context
                email_match = self.EMAIL_PATTERN.search(context_text)
                email = email_match.group(0) if email_match else None

                # Extract ORCID if present
                orcid_match = self.ORCID_PATTERN.search(context_text)
                orcid = orcid_match.group(1) if orcid_match else None

                provenance = AuthorProvenanceMetadata(
                    pipeline_version=self.pipeline_version,
                    run_id=self.run_id,
                    doc_fingerprint=doc_fingerprint,
                    generator_name=AuthorGeneratorType.REGEX_PATTERN,
                )

                candidate = AuthorCandidate(
                    doc_id=doc_id,
                    full_name=name,
                    role=role,
                    email=email,
                    orcid=orcid,
                    generator_type=AuthorGeneratorType.REGEX_PATTERN,
                    context_text=context_text,
                    context_location=Coordinate(page_num=1, block_id="unknown"),
                    initial_confidence=0.95,
                    provenance=provenance,
                )
                candidates.append(candidate)

        # Detect names with credentials (lower confidence, general authors)
        for match in self.NAME_WITH_CREDENTIALS.finditer(full_text):
            name = match.group(1).strip()
            normalized_name = self._normalize_name(name)

            if normalized_name in seen_names:
                continue
            if not self._is_valid_name(name):
                continue

            seen_names.add(normalized_name)

            context_start = max(0, match.start() - 100)
            context_end = min(len(full_text), match.end() + 100)
            context_text = full_text[context_start:context_end]

            # Try to determine role from context
            role = self._infer_role_from_context(context_text)

            # Extract email if present
            email_match = self.EMAIL_PATTERN.search(context_text)
            email = email_match.group(0) if email_match else None

            # Extract ORCID if present
            orcid_match = self.ORCID_PATTERN.search(context_text)
            orcid = orcid_match.group(1) if orcid_match else None

            provenance = AuthorProvenanceMetadata(
                pipeline_version=self.pipeline_version,
                run_id=self.run_id,
                doc_fingerprint=doc_fingerprint,
                generator_name=AuthorGeneratorType.HEADER_PATTERN,
            )

            candidate = AuthorCandidate(
                doc_id=doc_id,
                full_name=name,
                role=role,
                email=email,
                orcid=orcid,
                generator_type=AuthorGeneratorType.HEADER_PATTERN,
                context_text=context_text,
                context_location=Coordinate(page_num=1, block_id="unknown"),
                initial_confidence=0.85,
                provenance=provenance,
            )
            candidates.append(candidate)

        return candidates

    def _extract_authors_llm(
        self,
        full_text: str,
        doc_id: str,
        doc_fingerprint: str,
    ) -> List[AuthorCandidate]:
        """
        Extract authors using LLM from the first ~2000 chars of text.

        Sends the beginning of the document to Claude Haiku to extract
        person names that are document authors/investigators.
        """
        candidates: List[AuthorCandidate] = []
        if not self.llm_client:
            return candidates

        # Use first ~2000 chars where authors typically appear
        snippet = full_text[:2000]

        system_prompt = (
            "You are an expert at extracting author names from biomedical documents. "
            "Extract ONLY real person names that are authors or investigators of this document. "
            "Do NOT extract institution names, department names, addresses, or place names. "
            "Return a JSON array of objects with keys: \"full_name\" (string) and \"role\" (string). "
            "Role should be one of: author, principal_investigator, corresponding_author, co_investigator, unknown. "
            "If no authors are found, return an empty array []."
        )
        user_prompt = f"Extract authors from this document excerpt:\n\n{snippet}"

        try:
            response = self.llm_client.complete_json_any(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
                max_tokens=1000,
                call_type="author_extraction",
            )

            if not isinstance(response, list):
                response = [response] if isinstance(response, dict) else []

            for item in response:
                if not isinstance(item, dict):
                    continue
                name = (item.get("full_name") or "").strip()
                if not name or not self._is_valid_name(name):
                    continue

                role_str = (item.get("role") or "author").lower().strip()
                role_map = {
                    "principal_investigator": AuthorRoleType.PRINCIPAL_INVESTIGATOR,
                    "corresponding_author": AuthorRoleType.CORRESPONDING_AUTHOR,
                    "co_investigator": AuthorRoleType.CO_INVESTIGATOR,
                    "study_chair": AuthorRoleType.STUDY_CHAIR,
                    "steering_committee": AuthorRoleType.STEERING_COMMITTEE,
                }
                role = role_map.get(role_str, AuthorRoleType.AUTHOR)

                provenance = AuthorProvenanceMetadata(
                    pipeline_version=self.pipeline_version,
                    run_id=self.run_id,
                    doc_fingerprint=doc_fingerprint,
                    generator_name=AuthorGeneratorType.HEADER_PATTERN,
                )

                candidate = AuthorCandidate(
                    doc_id=doc_id,
                    full_name=name,
                    role=role,
                    generator_type=AuthorGeneratorType.HEADER_PATTERN,
                    context_text=snippet[:200],
                    context_location=Coordinate(page_num=1, block_id="llm_extraction"),
                    initial_confidence=0.90,
                    provenance=provenance,
                )
                candidates.append(candidate)

        except Exception as e:
            logger.warning("LLM author extraction failed: %s", e)

        return candidates

    def _detect_author_byline(
        self,
        full_text: str,
        doc_id: str,
        doc_fingerprint: str,
        seen_names: Set[str],
    ) -> List[AuthorCandidate]:
        """
        Detect author names in journal-style bylines.

        Looks for comma-separated lists of names near the beginning of the document,
        typically following the title. Handles asterisks for corresponding authors.
        """
        candidates: List[AuthorCandidate] = []

        # Focus on the first ~3000 chars where author lists typically appear
        search_text = full_text[:3000]

        # Look for "on behalf of" patterns first to identify investigator groups
        on_behalf_match = self.ON_BEHALF_PATTERN.search(full_text[:5000])
        investigator_group = on_behalf_match.group(1) if on_behalf_match else None

        # Find potential author list sections
        # Look for patterns like "Name1, Name2, Name3" or "Name1*, Name2*, Name3"
        # Usually appears after a title (ends with newline) and before Abstract/Introduction

        # Split into potential segments
        lines = search_text.split("\n")

        author_block = None
        author_block_start = None

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Check if this line looks like an author list
            # Must have multiple commas and names
            comma_count = line.count(",")
            if comma_count >= 2:
                # Count potential names (capitalized words)
                potential_names = re.findall(
                    r"[A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+", line
                )
                if len(potential_names) >= 3:
                    author_block = line
                    author_block_start = i
                    break

            # Also check for "Authors:" style
            if re.match(r"^Authors?:", line, re.IGNORECASE):
                # Author list might be on this line or next
                author_block = line
                author_block_start = i
                break

        # If we found an author block, look for continuation lines
        if author_block and author_block_start is not None:
            for j in range(author_block_start + 1, min(author_block_start + 5, len(lines))):
                next_line = lines[j].strip()
                if not next_line:
                    continue

                # Check if this line continues the author list
                # It should have commas and capitalized names, but not be a new section
                if any(
                    kw in next_line.lower()
                    for kw in ["abstract", "summary", "background", "introduction", "methods"]
                ):
                    break

                # If it has author-like content (names with commas), add it
                potential_names = re.findall(
                    r"[A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+", next_line
                )
                if len(potential_names) >= 1 and "," in next_line:
                    author_block += " " + next_line
                elif "on behalf of" in next_line.lower():
                    author_block += " " + next_line
                    break  # This is typically the end
                else:
                    break  # Not an author line

        if not author_block:
            return candidates

        # Parse names from the author block
        # Remove common suffixes/markers
        author_block = re.sub(r"\d+,?", "", author_block)  # Remove affiliation numbers
        author_block = re.sub(r"[\†\‡§¶#''\u2019\u2018\"\"?\u201C\u201D]+", "", author_block)  # Remove symbols and superscript artifacts
        author_block = re.sub(r"\(.*?\)", "", author_block)  # Remove parentheticals

        # Handle cases where title and authors are joined without line breaks
        # Try to split on common title/section patterns that precede author lists
        title_pattern = r"(?:Hypertension|Research|Article|Study|Report|Analysis|Review|Investigation)\s+"
        if re.search(title_pattern, author_block, re.IGNORECASE):
            # Split at the title word and check each part for author-like content
            parts = re.split(title_pattern, author_block, flags=re.IGNORECASE)
            # Find the part that contains author names with credentials
            for part in parts:
                part_stripped = part.strip()
                if not part_stripped:
                    continue
                # Check if this part has author-like content (names with credentials)
                # Look for pattern: Name + credential (MM, PhD, MD, etc.)
                if re.search(r"^[A-Z][a-z]+\s+[A-Z][a-z]+.*?(?:MM|MD|PhD|MPH|PharmD|DO)", part_stripped):
                    # Also verify it has multiple names (at least 2-3 comma or semicolon separated)
                    if part_stripped.count(";") >= 2 or part_stripped.count(",") >= 3:
                        author_block = part_stripped
                        break

        # Split "and" between last two authors (e.g., "Capuano and Antigny")
        author_block = re.sub(r"\band\b", ",", author_block)

        # Find names by splitting on commas or semicolons and parsing each segment
        # This handles complex names with multiple middle initials like "Edwin K S Wong"
        # Also handles journal-style semicolon separation like "Name1, PhD; Name2, MD"
        segments = re.split(r"[;,]\s*", author_block)

        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue

            # Check for corresponding author marker
            is_corresponding = "*" in segment
            segment = segment.replace("*", "").strip()

            # Remove ORCID symbols (® or similar icons that appear after names)
            segment = re.sub(r"[®©™\u00AE\u00A9\u2122\uFFFD]", "", segment).strip()

            # Strip trailing OCR artifacts (superscript markers misread as punctuation)
            segment = segment.rstrip("''\u2019\u2018\"\"?\u201C\u201D.!").strip()

            # Remove trailing credentials (MM, PhD, MD, etc.) from segment
            # This handles "Lihua Guan, PhD" -> "Lihua Guan"
            # Or "MM; Lihua Guan" -> "Lihua Guan" (when credentials are prefix from previous)
            credentials_pattern = r"(?:^|\s)(?:MM|MD|PhD|MPH|DO|PharmD|RN|MSN|DrPH|ScD|MBBS|MBChB|FRCPC?|FACP|MS|MA|MSc|BSc|BA|DNP|DPT)\s*$"
            segment = re.sub(credentials_pattern, "", segment, flags=re.IGNORECASE).strip()

            # Also remove leading credentials (from semicolon split artifacts)
            # Handles cases like "PhD; Weiping Xie" where "PhD" is leftover from previous name
            leading_cred_pattern = r"^(?:MM|MD|PhD|MPH|DO|PharmD|RN|MSN|DrPH|ScD|MBBS|MBChB|FRCPC?|FACP|MS|MA|MSc|BSc|BA|DNP|DPT)(?:\s*[;,]?\s*)"
            segment = re.sub(leading_cred_pattern, "", segment, flags=re.IGNORECASE).strip()

            # Skip if it's a group name or too short
            if len(segment) < 5:
                continue
            if any(
                kw in segment.lower()
                for kw in ["behalf", "investigator", "group", "committee", "abstract"]
            ):
                continue

            # Try to parse as a name
            # Pattern: First [Middle...] Last
            # Where Middle can be initials (K S) or full names
            name_match = re.match(
                r"^([A-Z][a-z]+(?:[-'][A-Z][a-z]+)?)"  # First name
                r"((?:\s+[A-Z](?:\.|\s|$))*)"  # Middle initials (K S or K. S.)
                r"(?:\s+([A-Z][a-z]+(?:[-'][A-Z][a-z]+)?))?$",  # Last name
                segment,
            )

            if name_match:
                first = name_match.group(1)
                middle = (name_match.group(2) or "").strip()
                last = name_match.group(3)

                if not last:
                    # Maybe the "middle" is actually the last name
                    # e.g., "Yaqin Wang" where Wang is captured as middle
                    parts = segment.split()
                    if len(parts) >= 2:
                        first = parts[0]
                        middle = " ".join(parts[1:-1]) if len(parts) > 2 else ""
                        last = parts[-1]

                if not first or not last:
                    continue

                # Build full name
                if middle:
                    full_name = f"{first} {middle} {last}"
                else:
                    full_name = f"{first} {last}"
            else:
                # Fallback: just split by spaces
                parts = segment.split()
                if len(parts) < 2:
                    continue
                # Assume first is first name, last is last name, middle is everything else
                first = parts[0]
                last = parts[-1]
                middle = " ".join(parts[1:-1]) if len(parts) > 2 else ""

                # Validate first and last are capitalized names
                if not (
                    first[0].isupper()
                    and last[0].isupper()
                    and (len(last) > 1 and last[1:].islower())
                ):
                    continue

                if middle:
                    full_name = f"{first} {middle} {last}"
                else:
                    full_name = f"{first} {last}"

            # Clean up extra spaces
            full_name = " ".join(full_name.split())

            normalized = self._normalize_name(full_name)
            if normalized in seen_names:
                continue
            if not self._is_valid_name(full_name):
                continue

            seen_names.add(normalized)

            # Determine role
            role = (
                AuthorRoleType.CORRESPONDING_AUTHOR
                if is_corresponding
                else AuthorRoleType.AUTHOR
            )

            provenance = AuthorProvenanceMetadata(
                pipeline_version=self.pipeline_version,
                run_id=self.run_id,
                doc_fingerprint=doc_fingerprint,
                generator_name=AuthorGeneratorType.HEADER_PATTERN,
            )

            candidate = AuthorCandidate(
                doc_id=doc_id,
                full_name=full_name,
                role=role,
                generator_type=AuthorGeneratorType.HEADER_PATTERN,
                context_text=author_block[:200],
                context_location=Coordinate(page_num=1, block_id="author_byline"),
                initial_confidence=0.90 if is_corresponding else 0.85,
                provenance=provenance,
            )
            candidates.append(candidate)

        # If we found an investigator group, add it as a note
        if investigator_group and candidates:
            # Mark first author as part of investigator group
            for c in candidates:
                if c.affiliation is None:
                    c.affiliation = f"on behalf of {investigator_group} investigators"
                    break

        return candidates

    def _normalize_name(self, name: str) -> str:
        """Normalize name for deduplication."""
        return " ".join(name.lower().split())

    def _is_valid_name(self, name: str) -> bool:
        """Check if name appears to be a valid person name."""
        parts = name.split()
        if len(parts) < 2:
            return False

        # Filter out common false positives
        invalid_names = {
            "study design",
            "data analysis",
            "statistical analysis",
            "new england",
            "journal medicine",
            "clinical trial",
        }
        if self._normalize_name(name) in invalid_names:
            return False

        # Name should have reasonable length
        if len(name) < 5 or len(name) > 60:
            return False

        return True

    def _infer_role_from_context(self, context: str) -> AuthorRoleType:
        """Infer author role from surrounding context."""
        context_lower = context.lower()

        if "principal investigator" in context_lower:
            return AuthorRoleType.PRINCIPAL_INVESTIGATOR
        if "corresponding author" in context_lower:
            return AuthorRoleType.CORRESPONDING_AUTHOR
        if "co-investigator" in context_lower or "coinvestigator" in context_lower:
            return AuthorRoleType.CO_INVESTIGATOR
        if "steering committee" in context_lower:
            return AuthorRoleType.STEERING_COMMITTEE
        if "study chair" in context_lower:
            return AuthorRoleType.STUDY_CHAIR
        if "data safety" in context_lower or "dsmb" in context_lower:
            return AuthorRoleType.DATA_SAFETY_BOARD

        return AuthorRoleType.AUTHOR

    def validate_candidates(
        self, candidates: List[AuthorCandidate]
    ) -> List[ExtractedAuthor]:
        """
        Validate author candidates.

        Since these are pattern matches, we auto-validate with the initial confidence.
        """
        validated: List[ExtractedAuthor] = []

        for candidate in candidates:
            evidence = EvidenceSpan(
                text=candidate.full_name,
                location=candidate.context_location,
                scope_ref="author_detection",
                start_char_offset=0,
                end_char_offset=len(candidate.full_name),
            )

            extracted = ExtractedAuthor(
                candidate_id=candidate.id,
                doc_id=candidate.doc_id,
                full_name=candidate.full_name,
                role=candidate.role,
                affiliation=candidate.affiliation,
                email=candidate.email,
                orcid=candidate.orcid,
                primary_evidence=evidence,
                status=ValidationStatus.VALIDATED,
                confidence_score=candidate.initial_confidence,
                validation_flags=["pattern_match"],
                provenance=candidate.provenance,
            )
            validated.append(extracted)

        return validated

    def export_to_json(
        self,
        validated: List[ExtractedAuthor],
        doc_id: str,
        doc_path: Optional[str] = None,
    ) -> AuthorExportDocument:
        """
        Create export document for authors.
        """
        entries: List[AuthorExportEntry] = []
        unique_names: Set[str] = set()

        for author in validated:
            unique_names.add(self._normalize_name(author.full_name))

            entry = AuthorExportEntry(
                full_name=author.full_name,
                role=author.role.value,
                affiliation=author.affiliation,
                email=author.email,
                orcid=author.orcid,
                confidence=author.confidence_score,
                context=author.primary_evidence.text,
                page=author.primary_evidence.location.page_num,
            )
            entries.append(entry)

        return AuthorExportDocument(
            run_id=self.run_id,
            timestamp=self.config.get("timestamp", ""),
            document=doc_id,
            document_path=doc_path,
            pipeline_version=self.pipeline_version,
            total_detected=len(validated),
            unique_authors=len(unique_names),
            authors=entries,
        )

    def print_summary(self) -> None:
        """Print loading summary."""
        logger.info("Author detector: regex-based detection initialized")

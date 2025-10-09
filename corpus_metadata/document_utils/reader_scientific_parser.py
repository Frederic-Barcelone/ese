#
# /Users/kxbb828/Library/CloudStorage/OneDrive-AZCollaboration/Desktop/WIP-ESE/17_Corpus/corpus_metadata/document_utils/reader_scientific_parser.py
#
"""
Scientific Parser Module
=======================

Specialized parser for extracting structure from scientific papers.
"""

import logging
import re
from typing import Dict, Optional, List, Tuple, Any

logger = logging.getLogger(__name__)


class ScientificPaperParser:
    """
    Parser for extracting structured information from scientific papers.

    Identifies and extracts common sections like abstract, introduction,
    methods, results, discussion, and references.
    """

    def __init__(self):
        """Initialize the scientific paper parser."""
        self.section_patterns = self._build_section_patterns()
        self.reference_patterns = self._build_reference_patterns()

    def _build_section_patterns(self) -> Dict[str, re.Pattern]:
        """
        Build regex patterns for common scientific paper sections.

        Returns:
            Dictionary mapping section names to compiled regex patterns
        """
        aliases = {
            'abstract': [r'abstract', r'summary'],
            'keywords': [r'keywords?'],
            'introduction': [r'introduction', r'background'],
            'methods': [
                r'methods?',
                r'materials?\s*(?:&|and)\s*methods?',
                r'methodology',
                r'experimental\s+procedures'
            ],
            'results': [r'results?', r'findings', r'data\s+analysis'],
            'discussion': [r'discussion', r'results\s+and\s+discussion'],
            'conclusion': [r'conclusions?', r'summary', r'discussion\s+and\s+conclusion'],
            'references': [r'references?', r'bibliography', r'works?\s+cited']
        }

        # Flatten all aliases to build lookahead for section boundaries
        all_headers = sum(aliases.values(), [])

        patterns: Dict[str, re.Pattern] = {}
        for section, words in aliases.items():
            # Build header alternation, ensuring full-line match (multiline mode)
            header_regex = r'^(?:' + r'|'.join(words) + r')[\.\:\s]*$'
            # Build lookahead that stops at any other recognized section header or end of text
            boundary_regex = r'(?=\n^(?:' + r'|'.join(all_headers) + r')[\.\:\s]*$|\Z)'
            # Capture the body after the header line
            full_regex = header_regex + r'\s*(.*?)' + boundary_regex
            patterns[section] = re.compile(
                full_regex,
                re.IGNORECASE | re.MULTILINE | re.DOTALL
            )

        return patterns

    def _build_reference_patterns(self) -> List[re.Pattern]:
        """
        Build regex patterns for detecting references.

        Returns:
            List of compiled regex patterns for different reference formats
        """
        return [
            # [1] Author, Year format
            re.compile(r'^\[\d+\]\s+[A-Z][a-z]+', re.MULTILINE),
            # 1. Author format
            re.compile(r'^\d+\.\s+[A-Z][a-z]+', re.MULTILINE),
            # Author (Year) format
            re.compile(r'[A-Z][a-z]+\s+\(\d{4}\)'),
            # DOI pattern (plain or URL)
            re.compile(
                r'(?:doi:\s*|https?://(?:dx\.)?doi\.org/)(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)',
                re.IGNORECASE
            ),
            # arXiv pattern
            re.compile(r'arXiv:\s*\d+\.\d+', re.IGNORECASE),
            # PubMed pattern
            re.compile(r'PMID:\s*\d+', re.IGNORECASE)
        ]

    def extract_sections(self, text: str) -> Dict[str, Any]:
        """
        Extract common sections from scientific paper text.

        Args:
            text: Full text of the paper

        Returns:
            Dictionary mapping section names to extracted text and metadata
        """
        sections: Dict[str, Any] = {}

        # Normalize text for better matching
        text = self._normalize_text(text)

        # Extract each section using its compiled regex
        for section_name, pattern in self.section_patterns.items():
            match = pattern.search(text)
            if match:
                section_text = match.group(1).strip()
                section_text = self._clean_section_text(section_text)
                if len(section_text) > 50:  # Only include substantial sections
                    sections[section_name] = section_text[:2000]
                    logger.debug(f"Extracted {section_name} section ({len(section_text)} chars)")

        # Extract additional metadata
        metadata = self._extract_paper_metadata(text, sections)
        if metadata:
            sections.update(metadata)

        return sections

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for better section matching.

        Args:
            text: Raw text

        Returns:
            Normalized text
        """
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Convert LaTeX section commands to plain headers
        text = re.sub(r'\\section\*?\{([^}]+)\}', r'\1\n', text)
        text = re.sub(r'\\subsection\*?\{([^}]+)\}', r'\1\n', text)

        # Remove hyphenation at line breaks
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)

        # Split into paragraphs, preserve blank lines between paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        cleaned_paragraphs = [re.sub(r'[ \t]*\n[ \t]*', ' ', p) for p in paragraphs]
        text = '\n\n'.join(cleaned_paragraphs)

        return text

    def _clean_section_text(self, text: str) -> str:
        """
        Clean extracted section text.

        Args:
            text: Raw section text

        Returns:
            Cleaned text
        """
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text)

        # Remove standalone figure or table captions
        text = re.sub(r'^(Figure|Table|Fig\.)\s+\d+.*?$', '', text, flags=re.MULTILINE)

        # Remove standalone page numbers
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)

        return text.strip()

    def _extract_paper_metadata(self, text: str, sections: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract additional metadata from the paper.

        Args:
            text: Full paper text
            sections: Already extracted sections

        Returns:
            Dictionary with additional metadata
        """
        metadata: Dict[str, Any] = {}

        # Extract title: look for the first standalone line before "Abstract"
        title = self._extract_title(text)
        if title:
            metadata['title'] = title

        # Extract authors
        authors = self._extract_authors(text[:1000])
        if authors:
            metadata['authors'] = authors

        # Extract DOI
        doi = self._extract_doi(text)
        if doi:
            metadata['doi'] = doi

        # Extract publication year
        year = self._extract_year(text)
        if year:
            metadata['year'] = year

        # Count references if references section was found
        if 'references' in sections:
            ref_count = self._count_references(sections['references'])
            metadata['reference_count'] = ref_count

        # Detect paper type
        paper_type = self._detect_paper_type(text, sections)
        if paper_type:
            metadata['paper_type'] = paper_type

        return metadata

    def _extract_title(self, text: str) -> Optional[str]:
        """
        Extract title, assuming it appears before the abstract and is a standalone line.

        Args:
            text: Full normalized text

        Returns:
            Title string or None
        """
        # Look for a line of Title Case text before the word "Abstract"
        # Split first 500 chars into lines
        snippet = text[:500]
        lines = snippet.split('\n')
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Skip empty lines or lines that contain digits or too many words
            if not stripped or len(stripped) > 200 or len(stripped.split()) < 2:
                continue
            # If next non-empty line contains "abstract", treat this as title
            for j in range(i + 1, len(lines)):
                nxt = lines[j].strip().lower()
                if nxt.startswith('abstract') or nxt.startswith('summary'):
                    # Validate title: reasonable length and capitalization
                    if 10 < len(stripped) < 200:
                        return stripped
            # Stop scanning after encountering "abstract" without a valid title above
            if stripped.lower().startswith('abstract') or stripped.lower().startswith('summary'):
                break
        return None

    def _extract_authors(self, text: str) -> Optional[List[str]]:
        """
        Extract author names from paper header.

        Args:
            text: Text from paper header

        Returns:
            List of author names or None
        """
        # Common author patterns
        patterns = [
            # e.g., "John A. Doe, Jane B. Smith and Bob C. Brown"
            re.compile(r'([A-Z][a-z]+(?:\s+[A-Z]\.?(?:[A-Z]\.?)?\s*)?(?:\s+[A-Z][a-z]+)+)(?:\s*,\s*|\s+and\s+)', re.MULTILINE),
            # e.g., "John Doe, Jane Smith"
            re.compile(r'([A-Z][a-z]+\s+[A-Z][a-z]+)(?:\s*,\s*|\s+and\s+)', re.MULTILINE),
        ]

        for pattern in patterns:
            matches = pattern.findall(text)
            if matches:
                # Clean up author names (remove trailing whitespace, digits, or footnote markers)
                authors = [
                    re.sub(r'[\d\*]+$', '', name.strip())
                    for name in matches
                    if len(name.strip()) > 5
                ]
                if authors:
                    return authors[:10]  # Limit to first 10 authors

        return None

    def _extract_doi(self, text: str) -> Optional[str]:
        """
        Extract DOI from text (either plain "doi:" or URL form).

        Args:
            text: Full text

        Returns:
            DOI string or None
        """
        doi_pattern = re.compile(
            r'(?:doi:\s*|https?://(?:dx\.)?doi\.org/)(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)',
            re.IGNORECASE
        )
        match = doi_pattern.search(text)
        if match:
            return match.group(1).strip()
        return None

    def _extract_year(self, text: str) -> Optional[str]:
        """
        Extract publication year from text, prioritizing lines with "Received", "Accepted", or "©".

        Args:
            text: Full text

        Returns:
            Year string or None
        """
        # First look for lines like "Received: Month DD, YYYY" or "Accepted: Month DD, YYYY" or "© YYYY"
        pub_match = re.search(r'(?:Received|Accepted|©)\s.*?(\b(19|20)\d{2}\b)', text, re.IGNORECASE)
        if pub_match:
            return pub_match.group(1)

        # Fallback: first occurrence of a 4-digit year between 1900 and 2099
        year_match = re.search(r'\b(19|20)\d{2}\b', text)
        if year_match:
            return year_match.group(0)

        return None

    def _count_references(self, ref_text: str) -> int:
        """
        Count number of references in reference section.

        Args:
            ref_text: Text from references section

        Returns:
            Estimated number of references
        """
        counts = []

        # Count bracketed numeric references, e.g., "[1]"
        bracket_refs = len(re.findall(r'^\[\d+\]', ref_text, re.MULTILINE))
        if bracket_refs:
            counts.append(bracket_refs)

        # Count numbered references, e.g., "1."
        numbered_refs = len(re.findall(r'^\d+\.', ref_text, re.MULTILINE))
        if numbered_refs:
            counts.append(numbered_refs)

        # As fallback, count year appearances (each entry often has one year)
        year_refs = len(re.findall(r'\b(19|20)\d{2}\b', ref_text))
        if year_refs:
            counts.append(year_refs)

        # If none found, attempt splitting on double newlines (paragraphs)
        if not counts:
            entries = re.split(r'\n\s*\n', ref_text.strip())
            if len(entries) > 1:
                counts.append(len(entries))

        return max(counts) if counts else 0

    def _detect_paper_type(self, text: str, sections: Dict[str, Any]) -> Optional[str]:
        """
        Detect the type of scientific paper.

        Args:
            text: Full paper text
            sections: Extracted sections

        Returns:
            Paper type or None
        """
        text_lower = text.lower()

        # Detect systematic review or meta-analysis
        if any(term in text_lower[:2000] for term in ['systematic review', 'literature review', 'meta-analysis']):
            return 'review'

        # Detect case study or case report
        if 'case study' in text_lower[:2000] or 'case report' in text_lower[:2000]:
            return 'case_study'

        # Detect experimental studies: look for randomized/trial/participants in methods
        if 'methods' in sections:
            methods_text = sections['methods'].lower()
            if any(word in methods_text for word in ['randomized', 'controlled', 'trial', 'participants']):
                return 'experimental'

        # Detect theoretical papers by counting occurrences of "theorem", "proof", "lemma"
        theory_count = (
            text_lower.count('theorem')
            + text_lower.count('proof')
            + text_lower.count('lemma')
        )
        if theory_count > 3:
            return 'theoretical'

        # Default to research if standard structure present
        if all(sec in sections for sec in ['abstract', 'introduction', 'conclusion']):
            return 'research'

        return None

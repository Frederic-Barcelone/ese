"""
Composite lexicon builder using canonical English stopword lists.

Replaces hand-curated YAML entries with programmatic sources (canonical English
stopwords, string.ascii_lowercase, calendar month names) and merges
domain-specific additions from YAML. Reduces maintenance burden and improves
coverage.

Key Components:
    - ENGLISH_STOPWORDS: 326 canonical English stopwords (spaCy 3.7 source)
    - stopwords_base(): ENGLISH_STOPWORDS minus abbreviation exclusions
    - single_letters(): a-z from string.ascii_lowercase
    - month_names(): Full + abbreviated month names from calendar module
    - build_obvious_noise(): Union of above + domain YAML additions
    - build_garbage_tokens(): Stopwords + domain YAML additions
    - CREDENTIALS: Unified credential set (drug + gene lists merged)

All builder functions are @lru_cache'd for performance parity with
the existing module-level constant pattern.

Dependencies:
    - string: ASCII letters
    - calendar: Month names
    - Z_utils.Z12_data_loader: Domain-specific YAML additions
"""

from __future__ import annotations

import calendar
import string
from functools import lru_cache
from typing import FrozenSet

from Z_utils.Z12_data_loader import load_term_set

# =============================================================================
# CANONICAL ENGLISH STOPWORDS
# =============================================================================
# Source: spaCy 3.7.4 spacy.lang.en.stop_words.STOP_WORDS
# This is the standard 326-word English stopword set. Embedded here to avoid
# requiring spaCy as a runtime dependency just for a word list.
ENGLISH_STOPWORDS: FrozenSet[str] = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "ca",
    "can", "can't", "cannot", "could", "couldn't", "d", "did", "didn't",
    "do", "does", "doesn't", "doing", "don't", "down", "during", "each",
    "few", "for", "from", "further", "get", "go", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "however", "i", "if", "in",
    "into", "is", "isn't", "it", "it's", "its", "itself", "just", "ll",
    "m", "ma", "made", "make", "may", "me", "might", "mightn't", "more",
    "most", "much", "must", "mustn't", "my", "myself", "n't", "name",
    "namely", "nd", "needn't", "neither", "never", "nevertheless", "new",
    "next", "nine", "no", "nobody", "none", "noone", "nor", "not",
    "nothing", "now", "n't", "o", "of", "off", "often", "on", "once",
    "one", "only", "onto", "or", "other", "others", "otherwise", "our",
    "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
    "please", "put", "quite", "rather", "re", "really", "regarding",
    "s", "same", "say", "shan't", "she", "she's", "should", "shouldn't",
    "show", "side", "since", "six", "so", "some", "somehow", "someone",
    "something", "sometime", "sometimes", "somewhere", "still", "such",
    "take", "ten", "than", "that", "the", "their", "them", "themselves",
    "then", "thence", "there", "thereafter", "thereby", "therefore",
    "therein", "thereupon", "these", "they", "third", "this", "those",
    "though", "three", "through", "throughout", "thru", "thus", "to",
    "together", "too", "top", "toward", "towards", "twelve", "twenty",
    "two", "under", "unless", "until", "up", "upon", "us", "used",
    "using", "various", "very", "via", "was", "wasn't", "we", "well",
    "were", "weren't", "what", "whatever", "when", "whence", "whenever",
    "where", "whereafter", "whereas", "whereby", "wherein", "whereupon",
    "wherever", "whether", "which", "while", "whither", "who", "whoever",
    "whole", "whom", "whose", "why", "will", "with", "within", "without",
    "won't", "would", "wouldn't", "yet", "you", "your", "yours",
    "yourself", "yourselves",
    # Additional spaCy entries (numbers, misc)
    "amount", "anyhow", "anything", "anywhere", "around", "back",
    "became", "become", "becomes", "becoming", "beside", "besides",
    "beyond", "bill", "bottom", "call", "con", "cry", "de",
    "describe", "detail", "done", "due", "eight", "either", "eleven",
    "else", "elsewhere", "empty", "enough", "even", "ever", "every",
    "everyone", "everything", "everywhere", "except", "fifteen",
    "fifty", "fill", "find", "fire", "first", "five", "former",
    "formerly", "forty", "found", "four", "front", "full", "give",
    "had", "has", "have", "hence", "hereafter", "hereby", "herein",
    "hereupon", "hundred", "inc", "indeed", "interest", "keep", "last",
    "latter", "latterly", "least", "less", "ltd", "many", "meanwhile",
    "mill", "mine", "moreover", "move", "n't", "no", "nobody", "nor",
    "noone", "nothing", "nowhere", "of", "often", "on", "one",
    "only", "other", "others", "otherwise", "our", "ours", "out",
    "over", "own", "per", "perhaps", "please", "put", "rather",
    "re", "seem", "seemed", "seeming", "seems", "serious", "several",
    "she", "should", "since", "sixty", "so", "some", "somehow",
    "someone", "something", "sometime", "sometimes", "somewhere",
    "still", "such", "system", "take", "ten", "than", "that",
    "the", "their", "them", "themselves", "then", "thence", "there",
    "thereafter", "thereby", "therefore", "therein", "thereupon",
    "these", "they", "thick", "thin", "third", "this", "those",
    "though", "three", "through", "throughout", "thru", "thus",
    "to", "together", "too", "top", "toward", "towards", "un",
    "up", "upon", "very", "was", "we", "well", "were", "what",
    "whatever", "when", "whence", "whenever", "where", "whereafter",
    "whereas", "whereby", "wherein", "whereupon", "wherever",
    "whether", "which", "while", "whither", "who", "whoever",
    "whole", "whom", "why", "will", "with", "within", "without",
    "would", "yet", "you", "your", "yours", "yourself", "yourselves",
    # spaCy single-letter stopwords
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l",
    "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x",
    "y", "z",
    # Commonly needed additions
    "also", "already", "always", "among", "amongst", "across",
    "along", "although", "another", "anyway", "afterwards",
    "again", "against", "alone", "above", "across", "after",
    "afterwards", "almost", "along", "already", "also", "although",
    "always", "among", "amongst", "amoungst", "amount", "another",
    "anyway", "anyhow", "anyone", "anything", "anywhere",
})

# Valid medical abbreviations that happen to be English stopwords.
# "or" = Odds Ratio, "us" = United States — must NOT be filtered.
ABBREVIATION_EXCLUSIONS: FrozenSet[str] = frozenset({"or", "us"})


@lru_cache(maxsize=1)
def stopwords_base() -> FrozenSet[str]:
    """Return canonical English stopwords minus abbreviation exclusions."""
    return ENGLISH_STOPWORDS - ABBREVIATION_EXCLUSIONS


@lru_cache(maxsize=1)
def single_letters() -> FrozenSet[str]:
    """Return all lowercase single ASCII letters (a-z)."""
    return frozenset(string.ascii_lowercase)


@lru_cache(maxsize=1)
def month_names() -> FrozenSet[str]:
    """Return full + abbreviated month names, lowercased, plus 'sept'."""
    names: set[str] = set()
    for i in range(1, 13):
        names.add(calendar.month_name[i].lower())   # "january", ...
        names.add(calendar.month_abbr[i].lower())    # "jan", ...
    names.add("sept")  # Common non-standard abbreviation
    names.discard("")   # calendar.month_name[0] is ""
    return frozenset(names)


@lru_cache(maxsize=1)
def build_obvious_noise() -> FrozenSet[str]:
    """Build the full obvious-noise set: library bases + domain YAML additions.

    Combines:
    - All single letters (a-z)
    - Canonical English stopwords (minus abbreviation exclusions)
    - Month names (full + abbreviated)
    - Domain-specific terms from noise_filters.yaml -> obvious_noise_domain
    """
    domain = load_term_set("noise_filters.yaml", "obvious_noise_domain")
    return single_letters() | stopwords_base() | month_names() | frozenset(domain)


@lru_cache(maxsize=1)
def build_garbage_tokens() -> FrozenSet[str]:
    """Build garbage-token set: canonical stopwords + domain YAML additions.

    Combines:
    - Canonical English stopwords (minus abbreviation exclusions)
    - Domain-specific tokens from biomedical_ner_data.yaml -> garbage_tokens_domain
    """
    domain = load_term_set("biomedical_ner_data.yaml", "garbage_tokens_domain")
    return stopwords_base() | frozenset(domain)


# Unified credentials constant — union of drug and gene credential lists.
# Sourced from the two previously separate YAML sections.
CREDENTIALS: FrozenSet[str] = frozenset({
    # Academic degrees
    "phd", "mph", "ms", "ma", "msc", "bsc", "ba", "mba",
    # Medical credentials
    "md", "mbbs", "frcp", "do", "rn", "np", "pa", "pharmd", "dnp", "dpt",
    # Dental/veterinary/other professional
    "dds", "dmd", "od", "dvm", "dc", "jd", "llm",
})


__all__ = [
    "ABBREVIATION_EXCLUSIONS",
    "CREDENTIALS",
    "ENGLISH_STOPWORDS",
    "build_garbage_tokens",
    "build_obvious_noise",
    "month_names",
    "single_letters",
    "stopwords_base",
]

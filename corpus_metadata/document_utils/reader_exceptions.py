#
# /Users/kxbb828/Library/CloudStorage/OneDrive-AZCollaboration/Desktop/WIP-ESE/17_Corpus/corpus_metadata/document_utils/reader_exceptions.py
#

"""
Reader Exceptions Module
=======================

Custom exceptions for the document reader system.
"""


class DocumentReaderError(Exception):
    """Base exception for all document reader errors."""
    pass


class FileNotFoundError(DocumentReaderError):
    """Raised when a file cannot be found."""
    pass


class UnsupportedFileTypeError(DocumentReaderError):
    """Raised when attempting to read an unsupported file type."""
    pass


class ExtractionError(DocumentReaderError):
    """Raised when content extraction fails."""
    pass


class OCRError(DocumentReaderError):
    """Raised when OCR processing fails."""
    pass


class ValidationError(DocumentReaderError):
    """Raised when file validation fails."""
    pass


class CacheError(DocumentReaderError):
    """Raised when cache operations fail."""
    pass


class ConfigurationError(DocumentReaderError):
    """Raised when configuration is invalid."""
    pass


class TimeoutError(DocumentReaderError):
    """Raised when an operation times out."""
    pass


class CorruptedFileError(DocumentReaderError):
    """Raised when a file appears to be corrupted."""
    pass
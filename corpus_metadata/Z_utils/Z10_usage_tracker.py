# corpus_metadata/Z_utils/Z10_usage_tracker.py
"""
Usage Tracker - SQLite database to track lexicon and data source usage.

Tracks which lexicons and data sources are used per document to help
identify which are valuable and which can be removed.

Usage:
    tracker = UsageTracker("usage_stats.db")

    # Start processing a document
    tracker.start_document("doc123", "my_paper.pdf")

    # Log lexicon matches
    tracker.log_lexicon_usage("doc123", "MONDO", matches=15)
    tracker.log_lexicon_usage("doc123", "RxNorm", matches=3)

    # Log data source queries
    tracker.log_datasource_usage("doc123", "PubTator3", queries=5, results=12)

    # Finish document
    tracker.finish_document("doc123")

    # Get statistics
    stats = tracker.get_lexicon_stats()
    print(stats)
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class UsageTracker:
    """SQLite-based tracker for lexicon and data source usage."""

    def __init__(self, db_path: str | Path = "usage_stats.db"):
        """
        Initialize the usage tracker.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    started_at TIMESTAMP NOT NULL,
                    finished_at TIMESTAMP,
                    status TEXT DEFAULT 'processing'
                )
            """)

            # Lexicon usage table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS lexicon_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL,
                    lexicon_name TEXT NOT NULL,
                    matches_count INTEGER DEFAULT 0,
                    candidates_generated INTEGER DEFAULT 0,
                    validated_count INTEGER DEFAULT 0,
                    logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents(document_id),
                    UNIQUE(document_id, lexicon_name)
                )
            """)

            # Data source usage table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS datasource_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL,
                    datasource_name TEXT NOT NULL,
                    queries_count INTEGER DEFAULT 0,
                    results_count INTEGER DEFAULT 0,
                    errors_count INTEGER DEFAULT 0,
                    logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents(document_id),
                    UNIQUE(document_id, datasource_name)
                )
            """)

            # Create indexes for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_lexicon_document
                ON lexicon_usage(document_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_lexicon_name
                ON lexicon_usage(lexicon_name)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_datasource_document
                ON datasource_usage(document_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_datasource_name
                ON datasource_usage(datasource_name)
            """)

            conn.commit()
            logger.info(f"Usage tracker database initialized: {self.db_path}")

        finally:
            conn.close()

    def start_document(self, document_id: str, filename: str) -> None:
        """
        Register a document as being processed.

        Args:
            document_id: Unique identifier for the document
            filename: Original PDF filename
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO documents (document_id, filename, started_at, status)
                VALUES (?, ?, ?, 'processing')
            """, (document_id, filename, datetime.now()))
            conn.commit()
        finally:
            conn.close()

    def finish_document(self, document_id: str, status: str = "completed") -> None:
        """
        Mark a document as finished processing.

        Args:
            document_id: Document identifier
            status: Final status ('completed', 'failed', 'partial')
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE documents
                SET finished_at = ?, status = ?
                WHERE document_id = ?
            """, (datetime.now(), status, document_id))
            conn.commit()
        finally:
            conn.close()

    def log_lexicon_usage(
        self,
        document_id: str,
        lexicon_name: str,
        matches: int = 0,
        candidates: int = 0,
        validated: int = 0,
    ) -> None:
        """
        Log lexicon usage for a document.

        Args:
            document_id: Document identifier
            lexicon_name: Name of the lexicon (e.g., 'MONDO', 'RxNorm')
            matches: Number of raw matches found
            candidates: Number of candidates generated
            validated: Number of validated entities
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO lexicon_usage
                    (document_id, lexicon_name, matches_count, candidates_generated, validated_count)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(document_id, lexicon_name) DO UPDATE SET
                    matches_count = matches_count + excluded.matches_count,
                    candidates_generated = candidates_generated + excluded.candidates_generated,
                    validated_count = validated_count + excluded.validated_count,
                    logged_at = CURRENT_TIMESTAMP
            """, (document_id, lexicon_name, matches, candidates, validated))
            conn.commit()
        finally:
            conn.close()

    def log_datasource_usage(
        self,
        document_id: str,
        datasource_name: str,
        queries: int = 0,
        results: int = 0,
        errors: int = 0,
    ) -> None:
        """
        Log data source usage for a document.

        Args:
            document_id: Document identifier
            datasource_name: Name of the data source (e.g., 'PubTator3', 'ClinicalTrials.gov')
            queries: Number of API queries made
            results: Number of results returned
            errors: Number of errors encountered
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO datasource_usage
                    (document_id, datasource_name, queries_count, results_count, errors_count)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(document_id, datasource_name) DO UPDATE SET
                    queries_count = queries_count + excluded.queries_count,
                    results_count = results_count + excluded.results_count,
                    errors_count = errors_count + excluded.errors_count,
                    logged_at = CURRENT_TIMESTAMP
            """, (document_id, datasource_name, queries, results, errors))
            conn.commit()
        finally:
            conn.close()

    def get_lexicon_stats(self) -> List[Dict[str, Any]]:
        """
        Get aggregated statistics for all lexicons.

        Returns:
            List of dicts with lexicon stats sorted by total matches
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    lexicon_name,
                    COUNT(DISTINCT document_id) as documents_used,
                    SUM(matches_count) as total_matches,
                    SUM(candidates_generated) as total_candidates,
                    SUM(validated_count) as total_validated,
                    AVG(matches_count) as avg_matches_per_doc
                FROM lexicon_usage
                GROUP BY lexicon_name
                ORDER BY total_matches DESC
            """)
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_datasource_stats(self) -> List[Dict[str, Any]]:
        """
        Get aggregated statistics for all data sources.

        Returns:
            List of dicts with data source stats sorted by total queries
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    datasource_name,
                    COUNT(DISTINCT document_id) as documents_used,
                    SUM(queries_count) as total_queries,
                    SUM(results_count) as total_results,
                    SUM(errors_count) as total_errors,
                    CASE WHEN SUM(queries_count) > 0
                        THEN ROUND(SUM(results_count) * 1.0 / SUM(queries_count), 2)
                        ELSE 0
                    END as avg_results_per_query
                FROM datasource_usage
                GROUP BY datasource_name
                ORDER BY total_queries DESC
            """)
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_document_usage(self, document_id: str) -> Dict[str, Any]:
        """
        Get usage details for a specific document.

        Args:
            document_id: Document identifier

        Returns:
            Dict with document info, lexicon usage, and datasource usage
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Get document info
            cursor.execute("""
                SELECT * FROM documents WHERE document_id = ?
            """, (document_id,))
            doc_row = cursor.fetchone()
            if not doc_row:
                return {}

            # Get lexicon usage
            cursor.execute("""
                SELECT lexicon_name, matches_count, candidates_generated, validated_count
                FROM lexicon_usage WHERE document_id = ?
            """, (document_id,))
            lexicons = [dict(row) for row in cursor.fetchall()]

            # Get datasource usage
            cursor.execute("""
                SELECT datasource_name, queries_count, results_count, errors_count
                FROM datasource_usage WHERE document_id = ?
            """, (document_id,))
            datasources = [dict(row) for row in cursor.fetchall()]

            return {
                "document": dict(doc_row),
                "lexicons": lexicons,
                "datasources": datasources,
            }
        finally:
            conn.close()

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        Get list of all processed documents.

        Returns:
            List of document records
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM documents ORDER BY started_at DESC
            """)
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_unused_lexicons(self, min_documents: int = 5) -> List[str]:
        """
        Find lexicons that have zero matches across multiple documents.

        Args:
            min_documents: Minimum documents processed to consider

        Returns:
            List of lexicon names with zero total matches
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT lexicon_name
                FROM lexicon_usage
                GROUP BY lexicon_name
                HAVING SUM(matches_count) = 0
                   AND COUNT(DISTINCT document_id) >= ?
            """, (min_documents,))
            return [row["lexicon_name"] for row in cursor.fetchall()]
        finally:
            conn.close()

    def print_summary(self) -> None:
        """Print a summary of usage statistics to console."""
        docs = self.get_all_documents()
        lexicon_stats = self.get_lexicon_stats()
        datasource_stats = self.get_datasource_stats()

        print("\n" + "=" * 60)
        print("USAGE TRACKER SUMMARY")
        print("=" * 60)

        print(f"\nDocuments processed: {len(docs)}")
        completed = sum(1 for d in docs if d.get("status") == "completed")
        print(f"  Completed: {completed}")
        print(f"  Failed/Partial: {len(docs) - completed}")

        print("\n" + "-" * 60)
        print("LEXICON USAGE")
        print("-" * 60)
        print(f"{'Lexicon':<25} {'Docs':<6} {'Matches':<10} {'Validated':<10}")
        print("-" * 60)
        for stat in lexicon_stats:
            print(f"{stat['lexicon_name']:<25} {stat['documents_used']:<6} "
                  f"{stat['total_matches']:<10} {stat['total_validated']:<10}")

        print("\n" + "-" * 60)
        print("DATA SOURCE USAGE")
        print("-" * 60)
        print(f"{'Data Source':<25} {'Docs':<6} {'Queries':<10} {'Results':<10} {'Errors':<8}")
        print("-" * 60)
        for stat in datasource_stats:
            print(f"{stat['datasource_name']:<25} {stat['documents_used']:<6} "
                  f"{stat['total_queries']:<10} {stat['total_results']:<10} "
                  f"{stat['errors_count']:<8}")

        print("\n" + "=" * 60)


# Singleton instance for easy access
_tracker: Optional[UsageTracker] = None


def get_tracker(db_path: str | Path = "usage_stats.db") -> UsageTracker:
    """Get or create the global usage tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = UsageTracker(db_path)
    return _tracker


__all__ = [
    "UsageTracker",
    "get_tracker",
]

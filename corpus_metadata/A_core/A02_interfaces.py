from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

# Python 3.9-friendly TypeAlias
from typing_extensions import TypeAlias

from A_core.A01_domain_models import (
    GeneratorType,
    Candidate,
    ExtractedEntity,
)

# Flexible doc model
DocumentModel: TypeAlias = Any


class BaseParser(ABC):
    """
    Interface for ingestion/parsing layer (PDF -> DocumentModel).
    """

    @abstractmethod
    def parse(self, file_path: str) -> DocumentModel:
        raise NotImplementedError


class BaseCandidateGenerator(ABC):
    """
    Interface for any strategy that generates candidates (high recall).
    """

    @property
    @abstractmethod
    def generator_type(self) -> GeneratorType:
        raise NotImplementedError

    @abstractmethod
    def extract(self, doc_structure: DocumentModel) -> List[Candidate]:
        raise NotImplementedError


class BaseVerifier(ABC):
    """
    Interface for the Judge (LLM or deterministic verifier).
    """

    @abstractmethod
    def verify(self, candidate: Candidate) -> ExtractedEntity:
        raise NotImplementedError


class BaseNormalizer(ABC):
    """
    Interface for post-verification standardization.
    """

    @abstractmethod
    def normalize(self, entity: ExtractedEntity) -> ExtractedEntity:
        raise NotImplementedError


class BaseEvaluationMetric(ABC):
    """
    Contract for metrics.
    """

    @abstractmethod
    def compute(
        self,
        predictions: List[ExtractedEntity],
        ground_truth: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        raise NotImplementedError

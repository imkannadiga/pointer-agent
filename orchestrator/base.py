"""Abstract interfaces for the SLM-planner -> VLM-grounder -> SLM-verifier pipeline.

Every concrete implementation (Qwen planner, Florence-2 grounder, ...) plugs in
behind these interfaces so swapping a component is a config change, not an
orchestration-code change.
"""
from abc import ABC, abstractmethod

from PIL import Image


class BasePlanner(ABC):
    """Raw instruction -> structured query."""

    @abstractmethod
    def parse(self, instruction: str) -> dict:
        """Returns {"target_phrase": str, "answer_type": "point"|"bbox"}."""
        raise NotImplementedError


class BaseGrounder(ABC):
    """Image + structured query -> point/bbox prediction."""

    @abstractmethod
    def ground(self, image: Image.Image, query: dict) -> dict:
        """Returns {"point": [x, y], "bbox": [x0, y0, x1, y1]} (both populated)."""
        raise NotImplementedError


class BaseVerifier(ABC):
    """Sanity-checks a grounder prediction against the original instruction."""

    @abstractmethod
    def verify(
        self, instruction: str, query: dict, prediction: dict, image_size: tuple[int, int]
    ) -> tuple[bool, dict]:
        """Returns (ok, prediction) - ok is False if the prediction looks wrong."""
        raise NotImplementedError

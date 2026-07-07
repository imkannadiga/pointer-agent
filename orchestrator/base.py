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
        """Returns {"target_phrase": str, "answer_type": "point"|"bbox",
        "referring_expression": str | None}.

        referring_expression is populated only for tasks that need more than
        a literal on-page phrase to disambiguate (e.g. "just before the word
        X", "between X and Y") - grounders that support it use it verbatim
        as their phrase-grounding prompt; grounders that don't can ignore it
        and fall back to target_phrase. None (the default) preserves the
        original single-phrase contract exactly."""
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

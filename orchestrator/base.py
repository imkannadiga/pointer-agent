"""Abstract interfaces for the SLM-planner -> VLM-grounder -> SLM-verifier pipeline.

Every concrete implementation (Qwen planner, Florence-2 grounder, ...) plugs in
behind these interfaces so swapping a component is a config change, not an
orchestration-code change.
"""
from abc import ABC, abstractmethod

from PIL import Image


class BasePlanner(ABC):
    """Raw instruction -> structured query.

    Fixed output shape regardless of category (see progress.md section 4b):
    the VLM is only ever asked to find `anchor_phrase`; `relation` tells the
    orchestrator how to turn that anchor into the final answer.
    """

    @abstractmethod
    def parse(self, instruction: str) -> dict:
        """Returns {"anchor_phrase": str, "relation": str,
        "relation_params": dict, "answer_type": "point"|"bbox"}.

        `relation == "self"` means the grounder's own bbox for anchor_phrase
        is the final answer. Every other relation is resolved by a
        BaseCharLocator against a crop around the grounded anchor - see
        `orchestrator/char_locator.py` for the concrete relation names
        (line_edge_start/end, before_word/after_word, between_chars,
        char_at_occurrence, sentence_end, line_bbox, paragraph_bbox,
        paragraph_edge_start/end, between_anchors) and what `relation_params`
        each expects."""
        raise NotImplementedError


class BaseGrounder(ABC):
    """Image + anchor phrase -> point/bbox prediction.

    Does exactly one job: locate `query["anchor_phrase"]` in the full image.
    Never asked to represent a relation/boundary/offset itself - that's
    BaseCharLocator's job."""

    @abstractmethod
    def ground(self, image: Image.Image, query: dict) -> dict:
        """Returns {"point": [x, y], "bbox": [x0, y0, x1, y1]} (both populated)."""
        raise NotImplementedError


class BaseCharLocator(ABC):
    """Deterministic (non-model) resolver: an anchor's grounded bbox + a
    relation -> a precise final point/bbox, via OCR on a crop around the
    anchor. Zero model calls, zero training - plain geometry/string logic."""

    @abstractmethod
    def resolve(
        self,
        image: Image.Image,
        anchor_phrase: str,
        anchor_bbox: list[int],
        relation: str,
        relation_params: dict,
        answer_type: str,
    ) -> dict:
        """Returns {"point": [x, y], "bbox": [x0, y0, x1, y1]} in full-image
        coordinates (not crop-local) - see progress.md section 4b's
        coordinate-transform correctness requirement. anchor_phrase is passed
        through (not just anchor_bbox) so implementations can sanity-check
        OCR output against its length before trusting it."""
        raise NotImplementedError


class BaseVerifier(ABC):
    """Sanity-checks a grounder prediction against the original instruction."""

    @abstractmethod
    def verify(
        self, instruction: str, query: dict, prediction: dict, image_size: tuple[int, int]
    ) -> tuple[bool, dict]:
        """Returns (ok, prediction) - ok is False if the prediction looks wrong."""
        raise NotImplementedError

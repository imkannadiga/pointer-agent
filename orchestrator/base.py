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

    def parse_batch(self, instructions: list[str]) -> list[dict]:
        """Batched counterpart of parse() - one query per instruction, same
        order in, same order out. Default implementation just loops (so
        every BasePlanner works with Pipeline.run_batch() even before it's
        optimized); override for a real batched model call, as
        orchestrator/planner_qwen.py does."""
        return [self.parse(instruction) for instruction in instructions]


class BaseGrounder(ABC):
    """Image + anchor phrase -> point/bbox prediction.

    Does exactly one job: locate `query["anchor_phrase"]` in the full image.
    Never asked to represent a relation/boundary/offset itself - that's
    BaseCharLocator's job."""

    @abstractmethod
    def ground(self, image: Image.Image, query: dict) -> dict:
        """Returns {"point": [x, y], "bbox": [x0, y0, x1, y1]} (both populated)."""
        raise NotImplementedError

    def ground_batch(self, images: list[Image.Image], queries: list[dict]) -> list[dict]:
        """Batched counterpart of ground() - same order in, same order out.
        Default implementation just loops; override for a real batched
        generate() call, as orchestrator/grounder_florence2.py does."""
        return [self.ground(image, query) for image, query in zip(images, queries)]


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

    def verify_batch(
        self,
        instructions: list[str],
        queries: list[dict],
        predictions: list[dict],
        image_sizes: list[tuple[int, int]],
    ) -> tuple[list[bool], list[dict]]:
        """Batched counterpart of verify() - returns (oks, predictions), same
        order in as out. Default implementation just loops; override for a
        real batched model call, as orchestrator/verifier_qwen.py does."""
        oks, preds = [], []
        for instruction, query, prediction, image_size in zip(
            instructions, queries, predictions, image_sizes
        ):
            ok, pred = self.verify(instruction, query, prediction, image_size)
            oks.append(ok)
            preds.append(pred)
        return oks, preds

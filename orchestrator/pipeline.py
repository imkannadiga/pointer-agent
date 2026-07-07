"""SLM planner -> VLM grounder -> [BaseCharLocator] -> SLM verifier loop,
with one retry on failure.

Rewired in Sprint 4 (see progress.md section 4b) around the planner's fixed
`{anchor_phrase, relation, relation_params, answer_type}` contract:
`relation == "self"` uses the grounder's own bbox directly; `"between_anchors"`
grounds a second phrase and computes the gap geometrically (no locator
involved); every other relation is handed off to a `BaseCharLocator` to
resolve precisely from a crop around the grounded anchor.
"""
import re

from PIL import Image

from orchestrator.base import BaseCharLocator, BaseGrounder, BasePlanner, BaseVerifier

_PUNCT_RE = re.compile(r"[^\w\s]")


def _normalize_phrase(phrase: str) -> str:
    return _PUNCT_RE.sub("", phrase).strip()


def _gap_between(b1: list, b2: list) -> dict:
    """Geometric gap between two full-image bboxes - same axis convention
    task_builder.py's between_words/blank_line ground truth uses: a
    horizontal gap when the boxes share a line, a vertical gap when they're
    stacked (e.g. two paragraphs either side of a blank line)."""
    y_overlap = min(b1[3], b2[3]) - max(b1[1], b2[1])
    x_overlap = min(b1[2], b2[2]) - max(b1[0], b2[0])
    if y_overlap > 0 and (x_overlap <= 0 or y_overlap >= x_overlap):
        left, right = (b1, b2) if b1[0] < b2[0] else (b2, b1)
        x0, x1 = left[2], right[0]
        if x1 <= x0:
            x0, x1 = min(left[0], right[0]), max(left[2], right[2])
        y0, y1 = min(b1[1], b2[1]), max(b1[3], b2[3])
    else:
        top, bottom = (b1, b2) if b1[1] < b2[1] else (b2, b1)
        y0, y1 = top[3], bottom[1]
        if y1 <= y0:
            y0, y1 = min(top[1], bottom[1]), max(top[3], bottom[3])
        x0, x1 = max(min(b1[0], b2[0]), 0), max(b1[2], b2[2])
    bbox = [round(x0), round(y0), round(x1), round(y1)]
    point = [round((bbox[0] + bbox[2]) / 2), round((bbox[1] + bbox[3]) / 2)]
    return {"bbox": bbox, "point": point}


class Pipeline:
    def __init__(
        self,
        planner: BasePlanner,
        grounder: BaseGrounder,
        verifier: BaseVerifier,
        char_locator: BaseCharLocator | None = None,
    ):
        self.planner = planner
        self.grounder = grounder
        self.verifier = verifier
        self.char_locator = char_locator
        # Exposed for eval/run_eval.py's relation-classification metric -
        # the planner's raw parsed query from the most recent run().
        self.last_query = None

    def run(self, image: Image.Image, instruction: str) -> dict:
        image_size = (image.width, image.height)
        query = self.planner.parse(instruction)
        self.last_query = query
        prediction = self._resolve(image, query)
        ok, prediction = self.verifier.verify(instruction, query, prediction, image_size)
        if ok:
            return prediction

        retry_phrase = _normalize_phrase(query["anchor_phrase"])
        if retry_phrase and retry_phrase != query["anchor_phrase"]:
            retry_query = {**query, "anchor_phrase": retry_phrase}
            retry_prediction = self._resolve(image, retry_query)
            retry_ok, retry_prediction = self.verifier.verify(
                instruction, retry_query, retry_prediction, image_size
            )
            if retry_ok:
                return retry_prediction

        return prediction

    def _resolve(self, image: Image.Image, query: dict) -> dict:
        relation = query.get("relation", "self")
        if relation == "self":
            return self.grounder.ground(image, query)
        if relation == "between_anchors":
            return self._resolve_between_anchors(image, query)

        anchor_prediction = self.grounder.ground(image, query)
        if self.char_locator is None:
            # No locator configured: best-effort fallback to the anchor's
            # own grounded box rather than failing outright.
            return anchor_prediction
        return self.char_locator.resolve(
            image,
            query["anchor_phrase"],
            anchor_prediction["bbox"],
            relation,
            query.get("relation_params") or {},
            query.get("answer_type", "point"),
        )

    def _resolve_between_anchors(self, image: Image.Image, query: dict) -> dict:
        first = self.grounder.ground(image, query)
        second_phrase = (query.get("relation_params") or {}).get("second_anchor_phrase")
        if not second_phrase or not hasattr(self.grounder, "ground_phrase"):
            return first
        second = self.grounder.ground_phrase(image, second_phrase)
        return _gap_between(first["bbox"], second["bbox"])

"""SLM planner -> VLM grounder -> [BaseCharLocator] -> SLM verifier loop,
with one retry on failure.

Rewired in Sprint 4 (see progress.md section 4b) around the planner's fixed
`{anchor_phrase, relation, relation_params, answer_type}` contract:
`relation == "self"` uses the grounder's own bbox directly; `"between_anchors"`
grounds a second phrase and computes the gap geometrically (no locator
involved); every other relation is handed off to a `BaseCharLocator` to
resolve precisely from a crop around the grounded anchor.

Occurrence disambiguation ("the 2nd X"): when the planner emits
`relation_params.n`, the anchor is picked as the nth of the grounder's
candidate boxes in reading order (`orchestrator/instances.py`) - plain
deterministic geometry, per the section-4b principle that ordinal counting
is never asked of a model. Composes with every relation, since n selects
the *anchor*, and the relation then resolves off that anchor as usual.

Batched evaluation (run_batch): the three model-backed stages (planner,
grounder, verifier) each get one batched call for the whole input instead
of one call per row - see the *_batch methods on BasePlanner/BaseGrounder/
BaseVerifier and their real implementations in planner_qwen.py/
grounder_florence2.py/verifier_qwen.py. What can't stay batched in lockstep:
- Different rows resolve through different relation branches after
  grounding (self / between_anchors / char-locator) - handled by
  partitioning the batch per relation after one shared batched grounding
  call (every relation needs the anchor grounded first).
- TesseractCharLocator does OCR, not model inference - there is nothing to
  batch there, so those rows still resolve in a plain per-row loop; this
  is not the expensive stage.
- The verifier's retry-on-failure is inherently conditional per row - it
  becomes a second, smaller batched pass over just the rows that failed
  the first verification, not a second full-batch pass.
"""
import re

from PIL import Image

from orchestrator.base import BaseCharLocator, BaseGrounder, BasePlanner, BaseVerifier
from orchestrator.instances import select_nth

_PUNCT_RE = re.compile(r"[^\w\s]")


def _normalize_phrase(phrase: str) -> str:
    return _PUNCT_RE.sub("", phrase).strip()


def _select_from_candidates(query: dict, prediction: dict) -> dict:
    """Shared by _ground_anchor()/_ground_anchor_batch() so the nth-instance
    selection step can't drift between the single-row and batched paths."""
    n = (query.get("relation_params") or {}).get("n")
    picked = select_nth(prediction.get("candidates") or [], n)
    if picked is None:
        return {"point": prediction["point"], "bbox": prediction["bbox"]}
    point = [round((picked[0] + picked[2]) / 2), round((picked[1] + picked[3]) / 2)]
    return {"point": point, "bbox": picked}


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
        # Batched counterpart of last_query - one query per row, same order
        # as the most recent run_batch() call's inputs.
        self.last_queries = None

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

    def run_batch(self, images: list[Image.Image], instructions: list[str]) -> list[dict]:
        """Batched counterpart of run(): one query per (image, instruction)
        pair, resolved with the same per-row branching + one-retry logic as
        run(), but with each model-backed stage issued as a single batched
        call across the whole input instead of one call per row. Order of
        `images`/`instructions` in is the order of the returned predictions."""
        assert len(images) == len(instructions)
        n = len(images)
        if n == 0:
            self.last_queries = []
            return []

        image_sizes = [(image.width, image.height) for image in images]
        queries = self.planner.parse_batch(instructions)
        self.last_queries = queries

        predictions = self._resolve_batch(images, queries)
        oks, predictions = self.verifier.verify_batch(instructions, queries, predictions, image_sizes)

        retry_idx, retry_images, retry_instructions, retry_queries = [], [], [], []
        for i in range(n):
            if oks[i]:
                continue
            retry_phrase = _normalize_phrase(queries[i]["anchor_phrase"])
            if retry_phrase and retry_phrase != queries[i]["anchor_phrase"]:
                retry_idx.append(i)
                retry_images.append(images[i])
                retry_instructions.append(instructions[i])
                retry_queries.append({**queries[i], "anchor_phrase": retry_phrase})

        if retry_idx:
            retry_predictions = self._resolve_batch(retry_images, retry_queries)
            retry_image_sizes = [image_sizes[i] for i in retry_idx]
            retry_oks, retry_predictions = self.verifier.verify_batch(
                retry_instructions, retry_queries, retry_predictions, retry_image_sizes
            )
            for local_i, i in enumerate(retry_idx):
                if retry_oks[local_i]:
                    predictions[i] = retry_predictions[local_i]

        return predictions

    def _ground_anchor(self, image: Image.Image, query: dict) -> dict:
        """Ground the anchor phrase, then - if the query carries an
        occurrence index (relation_params.n) and the grounder surfaced
        multiple candidate boxes - pick the nth candidate in reading order
        instead of blindly trusting the first match. With zero or one
        candidate there is nothing to disambiguate (a grounder SFT'd on
        occurrence phrases typically returns the one right box already), so
        the grounder's own answer stands."""
        prediction = self.grounder.ground(image, query)
        return _select_from_candidates(query, prediction)

    def _ground_anchor_batch(self, images: list[Image.Image], queries: list[dict]) -> list[dict]:
        """Batched counterpart of _ground_anchor(): one grounder call for
        the whole batch, then the same per-row nth-instance selection over
        each row's own candidates (a pure-Python/geometry step, so there is
        nothing to batch there beyond just looping it)."""
        raw = self.grounder.ground_batch(images, queries)
        return [_select_from_candidates(query, prediction) for query, prediction in zip(queries, raw)]

    def _resolve(self, image: Image.Image, query: dict) -> dict:
        relation = query.get("relation", "self")
        if relation == "self":
            return self._ground_anchor(image, query)
        if relation == "between_anchors":
            return self._resolve_between_anchors(image, query)

        anchor_prediction = self._ground_anchor(image, query)
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

    def _resolve_batch(self, images: list[Image.Image], queries: list[dict]) -> list[dict]:
        """Batched counterpart of _resolve(): every relation needs the
        anchor grounded first, so that happens as one batched call across
        the WHOLE input regardless of relation, then rows are partitioned
        by relation for what comes after:
        - "self": the grounded anchor prediction is already the answer.
        - "between_anchors": needs a second grounder call, batched again
          but only across this relation's own subset of the input.
        - everything else: TesseractCharLocator.resolve() is OCR, not a
          model call, so it stays a plain per-row loop - not the expensive
          stage batching is targeting.
        """
        anchor_predictions = self._ground_anchor_batch(images, queries)
        results: list[dict | None] = [None] * len(queries)
        between_idx = []

        for i, query in enumerate(queries):
            relation = query.get("relation", "self")
            if relation == "self":
                results[i] = anchor_predictions[i]
            elif relation == "between_anchors":
                between_idx.append(i)
            elif self.char_locator is None:
                # No locator configured: best-effort fallback to the
                # anchor's own grounded box, same as the single-row path.
                results[i] = anchor_predictions[i]
            else:
                results[i] = self.char_locator.resolve(
                    images[i],
                    query["anchor_phrase"],
                    anchor_predictions[i]["bbox"],
                    relation,
                    query.get("relation_params") or {},
                    query.get("answer_type", "point"),
                )

        if between_idx:
            self._resolve_between_anchors_batch(images, queries, anchor_predictions, results, between_idx)

        return results

    def _resolve_between_anchors_batch(
        self,
        images: list[Image.Image],
        queries: list[dict],
        anchor_predictions: list[dict],
        results: list[dict | None],
        between_idx: list[int],
    ) -> None:
        """Fills `results` in place for the "between_anchors" subset of a
        batch: a second grounder call for each row's second anchor phrase,
        batched across just this subset, then the same geometric gap
        computation _resolve_between_anchors() uses for a single row."""
        can_batch = hasattr(self.grounder, "ground_phrase_batch")
        second_images, second_phrases, valid_idx = [], [], []
        for i in between_idx:
            phrase = (queries[i].get("relation_params") or {}).get("second_anchor_phrase")
            if phrase and can_batch:
                second_images.append(images[i])
                second_phrases.append(phrase)
                valid_idx.append(i)
            else:
                # Same fallback as the single-row path: no second phrase, or
                # a grounder that doesn't support ground_phrase at all.
                results[i] = anchor_predictions[i]

        if not valid_idx:
            return
        seconds = self.grounder.ground_phrase_batch(second_images, second_phrases)
        for i, second in zip(valid_idx, seconds):
            results[i] = _gap_between(anchor_predictions[i]["bbox"], second["bbox"])

    def _resolve_between_anchors(self, image: Image.Image, query: dict) -> dict:
        first = self._ground_anchor(image, query)
        second_phrase = (query.get("relation_params") or {}).get("second_anchor_phrase")
        if not second_phrase or not hasattr(self.grounder, "ground_phrase"):
            return first
        second = self.grounder.ground_phrase(image, second_phrase)
        return _gap_between(first["bbox"], second["bbox"])

"""Tests for Pipeline.run_batch(): batched evaluation across multiple rows.

Covers what would be silent if wrong:
- the batched path must reproduce run()'s per-row results exactly (a
  regression here would mean the fast path quietly disagrees with the
  known-correct sequential one);
- the batched path must genuinely issue one call per stage, not silently
  fall back to the ABC's default per-row loop (that would "work" but give
  none of the speedup the feature exists for);
- the verifier's retry-on-failure must batch only the rows that actually
  failed, not the whole input a second time.

Stub planner/grounder/verifier are real BasePlanner/BaseGrounder/
BaseVerifier subclasses (not bare duck-typed objects) so they also exercise
the ABCs' default parse_batch/ground_batch/verify_batch fallback contract.
"""
from PIL import Image

from orchestrator.base import BaseCharLocator, BaseGrounder, BasePlanner, BaseVerifier
from orchestrator.instances import format_grounding_phrase
from orchestrator.pipeline import Pipeline

IMG = Image.new("RGB", (500, 500))


class _DictPlanner(BasePlanner):
    """Looks up a fixed query per instruction; counts calls to each entry
    point so tests can assert the batched path was actually used."""

    def __init__(self, queries_by_instruction: dict):
        self.queries_by_instruction = queries_by_instruction
        self.batch_calls = 0
        self.single_calls = 0

    def parse(self, instruction):
        self.single_calls += 1
        return self.queries_by_instruction[instruction]

    def parse_batch(self, instructions):
        self.batch_calls += 1
        return [self.queries_by_instruction[i] for i in instructions]


class _DictGrounder(BaseGrounder):
    """candidates_by_phrase is keyed by the *formatted* grounding phrase
    (format_grounding_phrase(anchor_phrase, n)) - the same shape
    Florence2Grounder builds internally - so a query's anchor_phrase/n
    threads through exactly as it would against the real grounder."""

    def __init__(self, candidates_by_phrase: dict):
        self.candidates_by_phrase = candidates_by_phrase
        self.batch_calls = 0
        self.single_calls = 0
        self.phrase_batch_calls = 0

    def _predict(self, phrase: str) -> dict:
        candidates = self.candidates_by_phrase[phrase]
        bbox = candidates[0]
        point = [round((bbox[0] + bbox[2]) / 2), round((bbox[1] + bbox[3]) / 2)]
        return {"point": point, "bbox": bbox, "candidates": list(candidates)}

    def _phrase_for(self, query: dict) -> str:
        n = (query.get("relation_params") or {}).get("n")
        return format_grounding_phrase(query["anchor_phrase"], n)

    def ground(self, image, query):
        self.single_calls += 1
        return self._predict(self._phrase_for(query))

    def ground_batch(self, images, queries):
        self.batch_calls += 1
        return [self._predict(self._phrase_for(q)) for q in queries]

    def ground_phrase(self, image, phrase):
        self.single_calls += 1
        return self._predict(phrase)

    def ground_phrase_batch(self, images, phrases):
        self.phrase_batch_calls += 1
        return [self._predict(p) for p in phrases]


class _ScriptedVerifier(BaseVerifier):
    """Fails an instruction exactly once (its first check) if it's in
    fail_once, then passes on every subsequent check - lets a test script
    "fails first pass, succeeds on retry" without any real model."""

    def __init__(self, fail_once: frozenset = frozenset()):
        self.fail_once = set(fail_once)
        self.seen = set()
        self.batch_calls = 0
        self.single_calls = 0
        self.batch_sizes = []

    def _ok(self, instruction: str) -> bool:
        if instruction in self.fail_once and instruction not in self.seen:
            self.seen.add(instruction)
            return False
        return True

    def verify(self, instruction, query, prediction, image_size):
        self.single_calls += 1
        return self._ok(instruction), prediction

    def verify_batch(self, instructions, queries, predictions, image_sizes):
        self.batch_calls += 1
        self.batch_sizes.append(len(instructions))
        return [self._ok(i) for i in instructions], predictions


class _OffsetCharLocator(BaseCharLocator):
    """Deterministic, distinguishable-from-the-anchor resolution, standing
    in for TesseractCharLocator's OCR-based relations."""

    def resolve(self, image, anchor_phrase, anchor_bbox, relation, relation_params, answer_type):
        x0, y0, x1, y1 = anchor_bbox
        bbox = [x0 + 1, y0 + 1, x0 + 3, y1]
        point = [round((bbox[0] + bbox[2]) / 2), round((bbox[1] + bbox[3]) / 2)]
        return {"point": point, "bbox": bbox}


# A fixed set of rows spanning every relation branch run_batch() partitions
# on: "self" (with and without an occurrence index), "between_anchors", and
# a char-locator relation ("before_word").
_QUERIES = {
    "find plain": {
        "anchor_phrase": "word", "relation": "self",
        "relation_params": {}, "answer_type": "point",
    },
    "find nth": {
        "anchor_phrase": "word", "relation": "self",
        "relation_params": {"n": 1}, "answer_type": "point",
    },
    "find gap": {
        "anchor_phrase": "left", "relation": "between_anchors",
        "relation_params": {"second_anchor_phrase": "right"}, "answer_type": "point",
    },
    "find caret": {
        "anchor_phrase": "word2", "relation": "before_word",
        "relation_params": {}, "answer_type": "point",
    },
}
_CANDIDATES = {
    "word": [[10, 10, 30, 30]],
    # Reading order sorts [B, A] -> [A, B] (A is above B); n=1 should select
    # A, which is NOT the raw first element - a batched path that ignored n
    # and just kept the grounder's own top box would wrongly return B here.
    'the 1st "word"': [[60, 140, 90, 160], [10, 100, 50, 120]],
    "left": [[10, 100, 50, 120]],
    "right": [[200, 100, 240, 120]],
    "word2": [[400, 400, 450, 420]],
}


def _make_stubs(fail_once=frozenset()):
    planner = _DictPlanner(_QUERIES)
    grounder = _DictGrounder(_CANDIDATES)
    verifier = _ScriptedVerifier(fail_once=fail_once)
    return planner, grounder, verifier


def test_run_batch_matches_run_per_row():
    instructions = list(_QUERIES.keys())
    images = [IMG] * len(instructions)

    expected = []
    for instruction in instructions:
        p, g, v = _make_stubs()
        pipeline = Pipeline(p, g, v, char_locator=_OffsetCharLocator())
        expected.append(pipeline.run(images[0], instruction))

    p, g, v = _make_stubs()
    batch_pipeline = Pipeline(p, g, v, char_locator=_OffsetCharLocator())
    actual = batch_pipeline.run_batch(images, instructions)

    assert actual == expected
    assert batch_pipeline.last_queries == [_QUERIES[i] for i in instructions]


def test_run_batch_issues_one_batched_call_per_stage_not_a_loop():
    instructions = list(_QUERIES.keys())
    images = [IMG] * len(instructions)
    planner, grounder, verifier = _make_stubs()
    pipeline = Pipeline(planner, grounder, verifier, char_locator=_OffsetCharLocator())

    pipeline.run_batch(images, instructions)

    assert planner.batch_calls == 1 and planner.single_calls == 0
    # One batched call grounds every row's anchor regardless of relation,
    # plus exactly one more (batched, not per-row) for "find gap"'s second
    # anchor phrase - never the per-row ground()/ground_phrase() methods.
    assert grounder.batch_calls == 1 and grounder.single_calls == 0
    assert grounder.phrase_batch_calls == 1
    assert verifier.batch_calls == 1 and verifier.single_calls == 0


def test_run_batch_retries_only_failed_rows():
    # "retry me"'s anchor_phrase is punctuated so _normalize_phrase() gives
    # the retry a genuinely different phrase to look up ("bad!" -> "bad").
    queries = {
        "find plain": _QUERIES["find plain"],
        "find nth": _QUERIES["find nth"],
        "retry me": {
            "anchor_phrase": "bad!", "relation": "self",
            "relation_params": {}, "answer_type": "point",
        },
    }
    candidates = dict(_CANDIDATES, **{
        "bad!": [[300, 300, 320, 320]],   # first attempt (verifier rejects it)
        "bad": [[10, 10, 12, 12]],        # retry attempt (verifier accepts it)
    })
    instructions = ["find plain", "find nth", "retry me"]

    planner = _DictPlanner(queries)
    grounder = _DictGrounder(candidates)
    verifier = _ScriptedVerifier(fail_once={"retry me"})
    pipeline = Pipeline(planner, grounder, verifier, char_locator=_OffsetCharLocator())

    predictions = pipeline.run_batch([IMG] * 3, instructions)

    # First verify_batch call covers all 3 rows; the retry call covers only
    # the one row that failed - never the whole batch a second time.
    assert verifier.batch_sizes == [3, 1]
    assert predictions[2]["bbox"] == [10, 10, 12, 12]  # the retry's box, not [300, 300, 320, 320]

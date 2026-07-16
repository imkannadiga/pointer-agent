"""Unit tests for the occurrence ("nth instance") machinery.

Covers the two failure modes that would be silent in production: a
reading-order sort that disagrees between data-gen (ground-truth index) and
pipeline (candidate selection), and nth-selection being applied when there
is nothing to disambiguate.
"""
from PIL import Image

from orchestrator.instances import (
    format_grounding_phrase,
    occurrence_index,
    select_nth,
    sort_reading_order,
)
from orchestrator.pipeline import Pipeline

# Three visual lines with y-jitter well inside half a box height; the
# correct reading order is a1 a2 a3 / b1 b2 / c1.
A1 = [10, 100, 50, 120]
A2 = [60, 103, 90, 123]   # +3px baseline jitter, same line
A3 = [95, 98, 140, 118]
B1 = [12, 140, 44, 160]
B2 = [80, 141, 130, 161]
C1 = [10, 180, 60, 200]
READING_ORDER = [A1, A2, A3, B1, B2, C1]
SHUFFLED = [B2, C1, A2, B1, A3, A1]


def test_reading_order_multiline():
    assert sort_reading_order(SHUFFLED) == READING_ORDER


def test_reading_order_with_key():
    items = [{"box": b, "tag": i} for i, b in enumerate(SHUFFLED)]
    ordered = sort_reading_order(items, bbox_of=lambda it: it["box"])
    assert [it["box"] for it in ordered] == READING_ORDER


def test_occurrence_index_is_identity_based():
    # Two instances with identical geometry must still index distinctly.
    twin_a = [10, 10, 20, 20]
    twin_b = [10, 10, 20, 20]
    assert occurrence_index([twin_a, twin_b], twin_b) in (1, 2)
    assert occurrence_index([A1, B1, C1], B1) == 2


def test_select_nth_picks_reading_order():
    assert select_nth(SHUFFLED, 1) == A1
    assert select_nth(SHUFFLED, 4) == B1


def test_select_nth_clamps_beyond_count():
    assert select_nth(SHUFFLED, 99) == C1


def test_select_nth_declines_when_nothing_to_disambiguate():
    assert select_nth([], 2) is None
    assert select_nth([A1], 2) is None            # single candidate: trust it
    assert select_nth(SHUFFLED, None) is None
    assert select_nth(SHUFFLED, 0) is None
    assert select_nth(SHUFFLED, "not-a-number") is None


def test_format_grounding_phrase():
    assert format_grounding_phrase("Submit") == "Submit"
    assert format_grounding_phrase("Submit", None) == "Submit"
    assert format_grounding_phrase("Submit", 2) == 'the 2nd "Submit"'
    assert format_grounding_phrase("via", 3) == 'the 3rd "via"'
    assert format_grounding_phrase("online", 12) == 'the 12th "online"'
    assert format_grounding_phrase("x", "junk") == "x"


class _StubPlanner:
    def __init__(self, query):
        self.query = query

    def parse(self, instruction):
        return self.query


class _StubGrounder:
    """Returns a fixed candidate list, first candidate as the top box -
    mirroring Florence2Grounder's return shape."""

    def __init__(self, candidates):
        self.candidates = candidates

    def ground(self, image, query):
        bbox = self.candidates[0]
        point = [round((bbox[0] + bbox[2]) / 2), round((bbox[1] + bbox[3]) / 2)]
        return {"point": point, "bbox": bbox, "candidates": list(self.candidates)}


class _StubVerifier:
    def verify(self, instruction, query, prediction, image_size):
        return True, prediction


def _run_pipeline(query, candidates):
    pipeline = Pipeline(_StubPlanner(query), _StubGrounder(candidates), _StubVerifier())
    return pipeline.run(Image.new("RGB", (200, 220)), "irrelevant")


def test_pipeline_selects_nth_candidate():
    query = {
        "anchor_phrase": "word", "relation": "self",
        "relation_params": {"n": 4}, "answer_type": "point",
    }
    prediction = _run_pipeline(query, SHUFFLED)
    assert prediction["bbox"] == B1  # 4th in reading order, not the grounder's top box


def test_pipeline_without_n_keeps_grounder_top_box():
    query = {
        "anchor_phrase": "word", "relation": "self",
        "relation_params": {}, "answer_type": "point",
    }
    prediction = _run_pipeline(query, SHUFFLED)
    assert prediction["bbox"] == SHUFFLED[0]


def test_pipeline_single_candidate_ignores_n():
    # A grounder SFT'd on occurrence phrases may return exactly the right
    # single box - n must not index into it.
    query = {
        "anchor_phrase": "word", "relation": "self",
        "relation_params": {"n": 3}, "answer_type": "point",
    }
    prediction = _run_pipeline(query, [A2])
    assert prediction["bbox"] == A2

"""Prompted (no fine-tuning) SLM planner: instruction -> structured query.

Output contract rewritten in Sprint 4 (see progress.md section 4b):
`{anchor_phrase, relation, relation_params, answer_type}` for every category,
replacing the old `target_phrase`/`referring_expression` pair.
`DEFAULT_SYSTEM_PROMPT` and `PLANNER_OUTPUT_KEYS`/`RELATION_NAMES` are the
single source of truth for this JSON shape - `orchestrator/pipeline.py` and
`orchestrator/char_locator.py` both import from here rather than
hardcoding the contract a second time, so the prompt and the code that
consumes its output can't silently drift apart the way `referring_expression`
was bolted on ad hoc in Sprint 3.
"""
import json
import re

from orchestrator.base import BasePlanner
from orchestrator.slm import QwenSLM

PLANNER_OUTPUT_KEYS = ("anchor_phrase", "relation", "relation_params", "answer_type")

# Every relation task_builder.py can statically assign a category to (see
# data_gen/task_builder.py) - "self" means the grounder's own bbox is final;
# everything else is resolved by orchestrator/char_locator.py.
RELATION_NAMES = (
    "self",
    "line_edge_start", "line_edge_end",
    "paragraph_edge_start", "paragraph_edge_end",
    "before_word", "after_word", "between_chars",
    "char_at_occurrence", "sentence_end",
    "line_bbox", "paragraph_bbox",
    "between_anchors",
)

DEFAULT_SYSTEM_PROMPT = (
    "You parse GUI-grounding instructions into a structured query. "
    "Identify the anchor_phrase: the literal text (or, for field-style "
    "instructions like \"the invoice number\", the semantic description) "
    "that a visual grounder should search the image for. Then pick exactly "
    "one relation describing how the final answer relates to that anchor:\n"
    '  "self" - the anchor\'s own box/location is the answer.\n'
    '  "line_edge_start" / "line_edge_end" - the left/right edge of the '
    "line the anchor word starts/ends.\n"
    '  "paragraph_edge_start" / "paragraph_edge_end" - the left/right edge '
    "of the paragraph's first/last word.\n"
    '  "before_word" / "after_word" - a caret position just before/after '
    "the anchor word.\n"
    '  "between_chars" - a caret position between two specific characters '
    "in the anchor word; fill relation_params with char1, char2.\n"
    '  "char_at_occurrence" - a specific character within the anchor word; '
    "fill relation_params with target_char and occurrence (1-indexed count "
    "of that character within the word, e.g. the 2nd \"e\").\n"
    '  "sentence_end" - the sentence-ending punctuation right after the '
    "anchor word.\n"
    '  "line_bbox" / "paragraph_bbox" - the full box of the line/paragraph '
    "containing the anchor word.\n"
    '  "between_anchors" - the gap between the anchor word and a second '
    "one; fill relation_params with second_anchor_phrase.\n"
    "relation_params is an empty object {} unless the relation above says "
    "otherwise. "
    'Respond with strict JSON only, no other text: '
    '{"anchor_phrase": "...", "relation": "...", "relation_params": {...}, '
    '"answer_type": "point"|"bbox"}. '
    'Use "bbox" only if the instruction asks for a box/boundary/extent/region; '
    'otherwise use "point".'
)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_QUOTED_RE = re.compile(r'"([^"]+)"')
_BBOX_KEYWORDS = ("box", "boundar", "extent", "region", "highlight")

# Keyword -> relation, checked in order; first match wins. Best-effort only -
# real relation classification is what slm/train_sft.py's SFT is for.
_RELATION_KEYWORDS = (
    ("between", "between_anchors"),
    ("just before", "before_word"),
    ("before", "before_word"),
    ("just after", "after_word"),
    ("after", "after_word"),
    ("start of the line", "line_edge_start"),
    ("beginning of the line", "line_edge_start"),
    ("line starts", "line_edge_start"),
    ("end of the line", "line_edge_end"),
    ("line ends", "line_edge_end"),
    ("start of the paragraph", "paragraph_edge_start"),
    ("end of the paragraph", "paragraph_edge_end"),
    ("sentence", "sentence_end"),
)


def _fallback_parse(instruction: str) -> dict:
    quoted = _QUOTED_RE.findall(instruction)
    anchor_phrase = quoted[0] if quoted else instruction.strip()
    lowered = instruction.lower()
    answer_type = "bbox" if any(k in lowered for k in _BBOX_KEYWORDS) else "point"
    relation = "self"
    for keyword, rel in _RELATION_KEYWORDS:
        if keyword in lowered:
            relation = rel
            break
    relation_params = {}
    if relation == "between_anchors" and len(quoted) >= 2:
        relation_params = {"second_anchor_phrase": quoted[1]}
    return {
        "anchor_phrase": anchor_phrase,
        "relation": relation,
        "relation_params": relation_params,
        "answer_type": answer_type,
    }


class QwenPlanner(BasePlanner):
    def __init__(self, slm: QwenSLM):
        self.slm = slm

    def parse(self, instruction: str) -> dict:
        response = self.slm.chat(DEFAULT_SYSTEM_PROMPT, f"Instruction: {instruction}")
        match = _JSON_BLOCK_RE.search(response)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if (
                    "anchor_phrase" in parsed
                    and parsed.get("relation") in RELATION_NAMES
                    and parsed.get("answer_type") in ("point", "bbox")
                ):
                    return {
                        "anchor_phrase": str(parsed["anchor_phrase"]),
                        "relation": parsed["relation"],
                        "relation_params": parsed.get("relation_params") or {},
                        "answer_type": parsed["answer_type"],
                    }
            except json.JSONDecodeError:
                pass
        return _fallback_parse(instruction)

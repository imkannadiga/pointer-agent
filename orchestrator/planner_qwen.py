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
    "Additionally, if the instruction singles out WHICH occurrence of the "
    'anchor text is meant (e.g. "the second Submit", "the 3rd occurrence '
    'of X", "the first \'via\'"), add "n" to relation_params: the 1-based '
    "occurrence number, counting instances in reading order (top-to-bottom, "
    "left-to-right). n combines with any relation. Omit n when the "
    "instruction does not specify an occurrence. "
    'Respond with strict JSON only, no other text: '
    '{"anchor_phrase": "...", "relation": "...", "relation_params": {...}, '
    '"answer_type": "point"|"bbox"}. '
    'Use "bbox" only if the instruction asks for a box/boundary/extent/region; '
    'otherwise use "point".'
)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

# Matches an anchor quoted with any of the four delimiter conventions actually
# observed in real pointerbench-text instructions: straight double quotes,
# straight single quotes, backticks, and guillemets. The old version only
# matched straight double quotes ('"([^"]+)"'), so any instruction using a
# different delimiter (very common - confirmed via dataset_fidelity_check.py
# at ~40% single-quote / ~20-23% backtick / ~23% guillemet on the real set)
# fell through to _fallback_parse's `else` branch and used the ENTIRE
# instruction as anchor_phrase. That's the exact, confirmed cause of
# full-instruction-length anchor_phrase values seen in OCR-sanity-check
# warnings (e.g. a 60-67 char "anchor" that is actually the whole sentence).
# Each alternative has its own capture group; _find_quoted() below flattens
# whichever group actually matched into one ordered list.
_QUOTED_RE = re.compile(r'"([^"]+)"|\'([^\']+)\'|`([^`]+)`|«([^»]+)»')
_BBOX_KEYWORDS = ("box", "boundar", "extent", "region", "highlight")


def _find_quoted(text: str) -> list[str]:
    """Returns quoted spans in the order they appear, regardless of which
    of the four delimiter styles was used. Note: doubled/nested delimiters
    of the same kind (e.g. `««X» Y»`) are a genuinely ambiguous edge case
    for a single-pass regex and are not specially handled here - this is a
    best-effort fallback (only invoked when the trained SLM's JSON output
    itself failed to parse), not the primary parsing path, so it doesn't
    need to be perfect on unusual nested quoting."""
    return [next(g for g in m.groups() if g is not None) for m in _QUOTED_RE.finditer(text)]


# English-only best-effort ordinal extraction for the fallback parser -
# real occurrence parsing (all 6 languages) is what slm/train_sft.py's SFT
# learns from the synthetic occurrence-phrased instructions.
_ORDINAL_WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
_ORDINAL_RE = re.compile(
    r"\b(?:the\s+)?(first|second|third|fourth|fifth|(\d+)(?:st|nd|rd|th))\b",
    re.IGNORECASE,
)


def _fallback_occurrence(lowered: str) -> int | None:
    m = _ORDINAL_RE.search(lowered)
    if not m:
        return None
    return int(m.group(2)) if m.group(2) else _ORDINAL_WORDS[m.group(1)]

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
    quoted = _find_quoted(instruction)
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
    # Strip quoted spans first so an ordinal word that is itself the anchor
    # text (e.g. Point to "first") doesn't read as an occurrence index.
    # Stripping now covers all four delimiter styles, not just straight
    # double quotes, for the same reason _find_quoted replaced the old
    # single-delimiter _QUOTED_RE.findall() call above.
    n = _fallback_occurrence(_QUOTED_RE.sub("", instruction).lower())
    if n:
        relation_params["n"] = n
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
        return self._parse_response(instruction, response)

    def parse_batch(self, instructions: list[str]) -> list[dict]:
        if not instructions:
            return []
        responses = self.slm.chat_batch(
            DEFAULT_SYSTEM_PROMPT, [f"Instruction: {i}" for i in instructions]
        )
        return [
            self._parse_response(instruction, response)
            for instruction, response in zip(instructions, responses)
        ]

    def _parse_response(self, instruction: str, response: str) -> dict:
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
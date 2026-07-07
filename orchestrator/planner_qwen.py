"""Prompted (no fine-tuning) SLM planner: instruction -> structured query."""
import json
import re

from orchestrator.base import BasePlanner
from orchestrator.slm import QwenSLM

SYSTEM_PROMPT = (
    "You parse GUI-grounding instructions into a structured query. "
    "Given an instruction, identify the exact text/word/phrase being referred to "
    "and whether the answer should be a single point or a bounding box. "
    "If the instruction refers to a position relative to some text (e.g. just "
    "before/after a word, or between two words/characters) rather than the text "
    "itself, also fill in referring_expression with a short phrase describing "
    "that position (e.g. \"just before the word 'cores'\"); otherwise leave it null. "
    'Respond with strict JSON only, no other text: '
    '{"target_phrase": "...", "answer_type": "point"|"bbox", "referring_expression": "..."|null}. '
    'Use "bbox" only if the instruction asks for a box/boundary/extent/region; otherwise use "point".'
)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_QUOTED_RE = re.compile(r'"([^"]+)"')
_BBOX_KEYWORDS = ("box", "boundar", "extent", "region", "highlight")


def _fallback_parse(instruction: str) -> dict:
    m = _QUOTED_RE.search(instruction)
    target_phrase = m.group(1) if m else instruction.strip()
    lowered = instruction.lower()
    answer_type = "bbox" if any(k in lowered for k in _BBOX_KEYWORDS) else "point"
    return {"target_phrase": target_phrase, "answer_type": answer_type, "referring_expression": None}


class QwenPlanner(BasePlanner):
    def __init__(self, slm: QwenSLM):
        self.slm = slm

    def parse(self, instruction: str) -> dict:
        response = self.slm.chat(SYSTEM_PROMPT, f"Instruction: {instruction}")
        match = _JSON_BLOCK_RE.search(response)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if "target_phrase" in parsed and parsed.get("answer_type") in ("point", "bbox"):
                    return {
                        "target_phrase": str(parsed["target_phrase"]),
                        "answer_type": parsed["answer_type"],
                        "referring_expression": parsed.get("referring_expression") or None,
                    }
            except json.JSONDecodeError:
                pass
        return _fallback_parse(instruction)

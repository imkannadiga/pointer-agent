"""Text-only SLM verifier: structural sanity check (plain Python, cheap) plus
one semantic confidence check (LM call) against the grounder's matched phrase.

Since the verifier has no vision, it cannot inspect the image directly - it
can only check the prediction's geometry and whether the matched phrase makes
sense as an answer to the instruction.
"""
import json
import re

from orchestrator.base import BaseVerifier
from orchestrator.slm import QwenSLM

SYSTEM_PROMPT = (
    "You check whether a phrase correctly identifies the target of a GUI-grounding "
    "instruction. Respond with strict JSON only: {\"confident\": true|false}."
)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _structurally_ok(prediction: dict, image_size: tuple[int, int]) -> bool:
    w, h = image_size
    bbox = prediction.get("bbox")
    if not bbox or len(bbox) != 4:
        return False
    x0, y0, x1, y1 = bbox
    if not (x1 > x0 and y1 > y0):
        return False
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return False
    point = prediction.get("point")
    if not point or len(point) != 2:
        return False
    px, py = point
    return 0 <= px <= w and 0 <= py <= h


class QwenVerifier(BaseVerifier):
    def __init__(self, slm: QwenSLM):
        self.slm = slm

    def _semantically_confident(self, instruction: str, query: dict) -> bool:
        user = (
            f'Instruction: {instruction}\n'
            f'Matched phrase: "{query["target_phrase"]}"\n'
            "Does the matched phrase plausibly identify what the instruction is asking for?"
        )
        response = self.slm.chat(SYSTEM_PROMPT, user, max_new_tokens=20)
        match = _JSON_BLOCK_RE.search(response)
        if match:
            try:
                parsed = json.loads(match.group(0))
                return bool(parsed.get("confident", True))
            except json.JSONDecodeError:
                pass
        return True  # fail open: don't burn a retry on unparsable verifier output

    def verify(
        self, instruction: str, query: dict, prediction: dict, image_size: tuple[int, int]
    ) -> tuple[bool, dict]:
        if not _structurally_ok(prediction, image_size):
            return False, prediction
        return self._semantically_confident(instruction, query), prediction

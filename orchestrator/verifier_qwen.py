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

    @staticmethod
    def _confidence_user_prompt(instruction: str, query: dict) -> str:
        return (
            f'Instruction: {instruction}\n'
            f'Matched phrase: "{query["anchor_phrase"]}"\n'
            "Does the matched phrase plausibly identify what the instruction is asking for?"
        )

    def _parse_confidence(self, response: str) -> bool:
        match = _JSON_BLOCK_RE.search(response)
        if match:
            try:
                parsed = json.loads(match.group(0))
                return bool(parsed.get("confident", True))
            except json.JSONDecodeError:
                pass
        return True  # fail open: don't burn a retry on unparsable verifier output

    def _semantically_confident(self, instruction: str, query: dict) -> bool:
        user = self._confidence_user_prompt(instruction, query)
        response = self.slm.chat(SYSTEM_PROMPT, user, max_new_tokens=20)
        return self._parse_confidence(response)

    def verify(
        self, instruction: str, query: dict, prediction: dict, image_size: tuple[int, int]
    ) -> tuple[bool, dict]:
        if not _structurally_ok(prediction, image_size):
            return False, prediction
        return self._semantically_confident(instruction, query), prediction

    def verify_batch(
        self,
        instructions: list[str],
        queries: list[dict],
        predictions: list[dict],
        image_sizes: list[tuple[int, int]],
    ) -> tuple[list[bool], list[dict]]:
        """Batched counterpart of verify(): the structural check is plain
        Python (no model call), so it stays a per-row loop; only the rows
        that pass it go into a single batched semantic-confidence LM call,
        so a batch full of already-structurally-invalid predictions never
        wastes a forward pass asking the SLM about them."""
        n = len(instructions)
        oks = [False] * n
        semantic_idx = [i for i in range(n) if _structurally_ok(predictions[i], image_sizes[i])]

        if semantic_idx:
            users = [self._confidence_user_prompt(instructions[i], queries[i]) for i in semantic_idx]
            responses = self.slm.chat_batch(SYSTEM_PROMPT, users, max_new_tokens=20)
            for i, response in zip(semantic_idx, responses):
                oks[i] = self._parse_confidence(response)

        return oks, predictions

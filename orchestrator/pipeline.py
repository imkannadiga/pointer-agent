"""SLM planner -> VLM grounder -> SLM verifier loop, with one retry on failure."""
import re

from PIL import Image

from orchestrator.base import BaseGrounder, BasePlanner, BaseVerifier

_PUNCT_RE = re.compile(r"[^\w\s]")


def _normalize_phrase(phrase: str) -> str:
    return _PUNCT_RE.sub("", phrase).strip()


class Pipeline:
    def __init__(self, planner: BasePlanner, grounder: BaseGrounder, verifier: BaseVerifier):
        self.planner = planner
        self.grounder = grounder
        self.verifier = verifier

    def run(self, image: Image.Image, instruction: str) -> dict:
        image_size = (image.width, image.height)
        query = self.planner.parse(instruction)
        prediction = self.grounder.ground(image, query)
        ok, prediction = self.verifier.verify(instruction, query, prediction, image_size)
        if ok:
            return prediction

        retry_phrase = _normalize_phrase(query["target_phrase"])
        if retry_phrase and retry_phrase != query["target_phrase"]:
            retry_query = {**query, "target_phrase": retry_phrase}
            retry_prediction = self.grounder.ground(image, retry_query)
            retry_ok, retry_prediction = self.verifier.verify(
                instruction, retry_query, retry_prediction, image_size
            )
            if retry_ok:
                return retry_prediction

        return prediction

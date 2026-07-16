"""Off-the-shelf (no fine-tuning) VLM grounder using Florence-2's native
phrase-grounding task head."""
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from orchestrator.base import BaseGrounder
from orchestrator.instances import format_grounding_phrase

TASK_PROMPT = "<CAPTION_TO_PHRASE_GROUNDING>"


class Florence2Grounder(BaseGrounder):
    def __init__(
        self,
        model_name: str = "microsoft/Florence-2-base",
        device: str = "cpu",
        adapter_path: str | None = None,
    ):
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.to(device)
        self.model.eval()
        self.device = device

    def ground(self, image: Image.Image, query: dict) -> dict:
        # When the query carries an occurrence index (relation_params.n),
        # the phrase handed to Florence-2 keeps that context ('the 2nd
        # "Submit"') instead of truncating it to the bare word - same
        # enriched shape vlm/dataset.py trains on. The deterministic nth
        # selection over the returned candidates happens in pipeline.py.
        n = (query.get("relation_params") or {}).get("n")
        return self.ground_phrase(image, format_grounding_phrase(query["anchor_phrase"], n))

    def ground_phrase(self, image: Image.Image, phrase: str) -> dict:
        """Grounds a raw phrase directly - used for the "self" relation and
        for between_anchors' second anchor, which isn't a full query dict."""
        prompt = TASK_PROMPT + phrase
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=256,
                num_beams=1,
                do_sample=False,
                # Florence-2's checkpoint ships early_stopping=True (a
                # beam-search-era default from its decoder config) alongside
                # num_beams=3; without this override that stale default
                # leaks through against our num_beams=1 greedy decoding and
                # transformers warns about the mismatch on every call.
                early_stopping=False,
            )
        text_out = self.processor.batch_decode(out, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            text_out, task=TASK_PROMPT, image_size=(image.width, image.height)
        )
        bboxes = parsed.get(TASK_PROMPT, {}).get("bboxes", [])
        candidates = [[round(v) for v in b] for b in bboxes]
        if not candidates:
            # No match at all: fall back to the full image so downstream code
            # always has a well-formed (if wrong) prediction to work with.
            bbox = [0, 0, image.width, image.height]
        else:
            bbox = candidates[0]
        point = [round((bbox[0] + bbox[2]) / 2), round((bbox[1] + bbox[3]) / 2)]
        # candidates carries every box Florence-2 returned for the phrase
        # (empty on total miss) so pipeline.py can resolve "the nth X"
        # deterministically instead of trusting the first match.
        return {"point": point, "bbox": bbox, "candidates": candidates}

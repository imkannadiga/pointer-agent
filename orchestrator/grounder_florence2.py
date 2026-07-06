"""Off-the-shelf (no fine-tuning) VLM grounder using Florence-2's native
phrase-grounding task head."""
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from orchestrator.base import BaseGrounder

TASK_PROMPT = "<CAPTION_TO_PHRASE_GROUNDING>"


class Florence2Grounder(BaseGrounder):
    def __init__(self, model_name: str = "microsoft/Florence-2-base", device: str = "cpu"):
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        )
        self.model.to(device)
        self.model.eval()
        self.device = device

    def ground(self, image: Image.Image, query: dict) -> dict:
        prompt = TASK_PROMPT + query["target_phrase"]
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=256,
                num_beams=1,
                do_sample=False,
            )
        text_out = self.processor.batch_decode(out, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            text_out, task=TASK_PROMPT, image_size=(image.width, image.height)
        )
        bboxes = parsed.get(TASK_PROMPT, {}).get("bboxes", [])
        if not bboxes:
            # No match at all: fall back to the full image so downstream code
            # always has a well-formed (if wrong) prediction to work with.
            bbox = [0, 0, image.width, image.height]
        else:
            bbox = [round(v) for v in bboxes[0]]
        point = [round((bbox[0] + bbox[2]) / 2), round((bbox[1] + bbox[3]) / 2)]
        return {"point": point, "bbox": bbox}

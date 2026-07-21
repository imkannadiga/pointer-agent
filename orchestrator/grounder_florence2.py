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

    def ground_batch(self, images: list[Image.Image], queries: list[dict]) -> list[dict]:
        phrases = [
            format_grounding_phrase(q["anchor_phrase"], (q.get("relation_params") or {}).get("n"))
            for q in queries
        ]
        return self.ground_phrase_batch(images, phrases)

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

    def ground_phrase_batch(self, images: list[Image.Image], phrases: list[str]) -> list[dict]:
        """Batched counterpart of ground_phrase(): one padded generate()
        call for the whole batch instead of one call per (image, phrase)
        pair. Florence-2's language model is encoder-decoder (BART-style),
        not decoder-only like QwenSLM - so, unlike chat_batch's left-padding
        requirement, padding side doesn't affect correctness here: the
        decoder always starts fresh from BOS regardless of how the
        encoder's input got padded, as long as attention_mask marks the pad
        positions (which processor(..., padding=True) already sets - the
        same call shape vlm/dataset.py's training collate_fn uses, just for
        generation instead of teacher forcing). post_process_generation has
        no batched form in the library, so decoding each row's boxes stays
        a per-item Python loop; only the expensive model call is batched.
        """
        if not images:
            return []
        prompts = [TASK_PROMPT + p for p in phrases]
        inputs = self.processor(
            text=prompts, images=images, return_tensors="pt", padding=True
        ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                # Deliberately NOT passing attention_mask, even though the
                # processor computed a real one for this padded batch: found
                # by hitting a shape-mismatch crash and reading Florence-2's
                # own remote code (modeling_florence2.py) to root-cause it.
                # Florence2ForConditionalGeneration.forward()/generate() both
                # discard whatever attention_mask they're given and rebuild
                # their own once pixel_values is set (_merge_input_ids_with_
                # image_features concatenates an all-ones image-token mask
                # with an all-ones text-token mask - never the caller's real
                # per-token mask). forward() overwrites its local variable
                # with that recomputed one before use, so it's shape-safe
                # (silently not padding-aware); generate() does the same
                # recomputation but then ALSO forwards any attention_mask
                # from **kwargs verbatim into the inner language_model.
                # generate() call, unrescaled - so passing ours (sized to
                # the raw text length) crashes once it meets inputs_embeds
                # sized to the fused image+text length. Omitting it here
                # reproduces exactly the same "attends across text padding"
                # imprecision vlm/train_sft.py's batched training forward()
                # pass already has (harmless there because label masking is
                # a separate, correct mechanism) - a pre-existing Florence-2
                # HF-port limitation, not a new correctness regression.
                max_new_tokens=256,
                num_beams=1,
                do_sample=False,
                early_stopping=False,
            )
        texts = self.processor.batch_decode(out, skip_special_tokens=False)
        results = []
        for text_out, image in zip(texts, images):
            parsed = self.processor.post_process_generation(
                text_out, task=TASK_PROMPT, image_size=(image.width, image.height)
            )
            bboxes = parsed.get(TASK_PROMPT, {}).get("bboxes", [])
            candidates = [[round(v) for v in b] for b in bboxes]
            bbox = candidates[0] if candidates else [0, 0, image.width, image.height]
            point = [round((bbox[0] + bbox[2]) / 2), round((bbox[1] + bbox[3]) / 2)]
            results.append({"point": point, "bbox": bbox, "candidates": candidates})
        return results

"""Converts synthetic metadata rows into Florence-2
`<CAPTION_TO_PHRASE_GROUNDING>` SFT pairs (Phase 1).

Targets stay in Florence-2's own location-token vocabulary (`<loc_N>`,
N in [0, 999) per axis, quantized the same way its own post-processing code
dequantizes: `bin = floor(coord / dim * 1000)`) rather than raw decimal
pixels - see vlm/train_sft.py's module docstring / progress.md for why this
generalizes better than a from-scratch JSON-coordinate output head. The VLM
only ever grounds `anchor_phrase` (Sprint 4 architecture refinement - see
progress.md section 4b): it's never asked to represent a relation/boundary
itself, so every row's target is anchor_phrase -> its own bbox, regardless
of category.
"""
import json
import os

from PIL import Image
from torch.utils.data import Dataset

from orchestrator.grounder_florence2 import TASK_PROMPT
from orchestrator.instances import format_grounding_phrase

LOC_BINS = 1000


def _quantize(coord: float, size: float, bins: int = LOC_BINS) -> int:
    return min(max(int(coord / size * bins), 0), bins - 1)


def bbox_to_location_tokens(bbox: list, image_size: tuple) -> str:
    w, h = image_size
    x0, y0, x1, y1 = bbox
    bx0, by0 = _quantize(x0, w), _quantize(y0, h)
    bx1, by1 = _quantize(x1, w), _quantize(y1, h)
    return f"<loc_{bx0}><loc_{by0}><loc_{bx1}><loc_{by1}>"


class GroundingSFTDataset(Dataset):
    def __init__(self, metadata_path: str, max_samples: int | None = None):
        images_dir = os.path.join(os.path.dirname(metadata_path), "images")
        with open(metadata_path) as f:
            rows = [json.loads(line) for line in f]
        if max_samples:
            rows = rows[:max_samples]
        self.rows = rows
        self.images_dir = images_dir

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        image = Image.open(os.path.join(self.images_dir, row["file_name"])).convert("RGB")
        # Occurrence rows (relation_params.n) train on the same enriched
        # referring expression ('the 2nd "X"') the grounder is handed at
        # inference (orchestrator/grounder_florence2.py builds it from the
        # same formatter), teaching Florence-2 to use the ordinal context
        # to pick the right instance - never a phrase shape it won't see.
        n = (row.get("relation_params") or {}).get("n")
        phrase = format_grounding_phrase(row["anchor_phrase"], n)
        prompt = TASK_PROMPT + phrase
        # Trains on anchor_bbox (the anchor's own box), not row["bbox"] (the
        # task's final answer) - for relational categories these differ (e.g.
        # caret_before_word's bbox is a caret sliver outside the word; the
        # VLM must only ever learn to find the word itself, never the caret).
        target_text = phrase + bbox_to_location_tokens(row["anchor_bbox"], tuple(row["image_size"]))
        return {"image": image, "prompt": prompt, "target_text": target_text}


def make_collate_fn(processor):
    def collate(batch: list) -> dict:
        images = [b["image"] for b in batch]
        prompts = [b["prompt"] for b in batch]
        targets = [b["target_text"] for b in batch]

        inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
        labels = processor.tokenizer(
            text=targets, return_tensors="pt", padding=True, return_token_type_ids=False
        ).input_ids
        labels[labels == processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs

    return collate

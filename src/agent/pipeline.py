"""Zoom-and-verify inference agent: coarse box, zoomed refinement, projection.

Coordinate spaces used below:

* **frame** — the fixed benchmark frame (default 1024x768); every public
  input and output of the agent lives here.
* **crop** — pixels of the margin-expanded macro crop, before re-letterboxing.
* **zoom** — the crop letterboxed back up to frame size; what ``micro_step``
  actually sees.

``project_coordinates`` undoes zoom -> crop (inverse letterbox) and then
crop -> frame (translation by the crop origin). It takes a
:class:`CropContext` rather than the raw macro box because the macro box
alone under-determines the mapping: the crop was margin-expanded, clamped
to the frame and rounded to integer pixels, and the exact letterbox
scale/padding of that final crop is what must be inverted.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import torch

from src.data.dataset import DEFAULT_PROMPT_TEMPLATE, build_messages, tensor_to_pil
from src.data.transforms import (
    LetterboxParams,
    compute_letterbox,
    letterbox_image,
    unletterbox_boxes,
)
from src.utils.rewards import parse_box

logger = logging.getLogger(__name__)

#: Below this macro-box extent (frame px) zooming adds nothing but blur.
_MIN_ZOOM_EXTENT = 4.0


@dataclass(frozen=True)
class CropContext:
    """Geometry linking one zoomed crop back to the full frame."""

    crop_box: tuple[int, int, int, int]  # margin-expanded, clamped, frame px
    letterbox: LetterboxParams           # crop size -> frame size


class ZoomAndVerifyAgent:
    """Two-pass grounding: macro box on the full frame, micro box on a zoom.

    Inference-only wrapper around a chat VLM. ``image`` arguments are float
    ``(3, H, W)`` tensors in ``[0, 1]`` already letterboxed to
    ``target_size`` — the same frame the model was trained on. Generation is
    greedy for deterministic evaluation.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        target_size: tuple[int, int] = (1024, 768),
        margin: float = 0.15,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        max_new_tokens: int = 48,
    ) -> None:
        self.model = model
        self.processor = processor
        self.target_size = (int(target_size[0]), int(target_size[1]))
        self.margin = float(margin)
        self.prompt_template = prompt_template
        self.max_new_tokens = int(max_new_tokens)

    # ------------------------------------------------------------------ model

    @torch.inference_mode()
    def _generate_box(self, image: torch.Tensor, query: str) -> torch.Tensor | None:
        """One grounding call: image + query -> parsed frame-space box or None."""
        text = self.processor.apply_chat_template(
            build_messages(query, self.prompt_template),
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text], images=[tensor_to_pil(image)], return_tensors="pt"
        ).to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]
        generated = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
        )
        completion = self.processor.batch_decode(
            generated[:, prompt_len:], skip_special_tokens=True
        )[0]
        box = parse_box(completion)
        if box is None:
            logger.warning("no parseable box in completion: %r", completion)
        return box

    def macro_step(self, image: torch.Tensor, query: str) -> torch.Tensor | None:
        """Coarse prediction on the full letterboxed frame."""
        self._check_frame(image)
        return self._generate_box(image, query)

    def micro_step(self, cropped_image: torch.Tensor, query: str) -> torch.Tensor | None:
        """Refined prediction on a zoomed crop (output is in zoom coords)."""
        self._check_frame(cropped_image)
        return self._generate_box(cropped_image, query)

    # ------------------------------------------------------------------- math

    def crop_and_pad(
        self,
        image: torch.Tensor,
        macro_bbox: torch.Tensor,
        margin: float | None = None,
    ) -> tuple[torch.Tensor, CropContext]:
        """Crop ``macro_bbox`` (expanded by ``margin``) and re-letterbox it.

        The box is expanded by ``margin`` times its width/height on each
        side, clamped to the frame, and rounded outward to integer pixels;
        the crop is then letterboxed (uniform scale + centred zero-padding,
        never stretched) back to ``target_size``.

        Returns:
            ``(zoom_image, ctx)`` — the frame-sized zoom tensor and the
            geometry needed by :func:`project_coordinates` to come back.
        """
        self._check_frame(image)
        margin = self.margin if margin is None else float(margin)
        frame_w, frame_h = self.target_size

        x0, y0, x1, y1 = (float(v) for v in macro_bbox.tolist())
        pad_w, pad_h = (x1 - x0) * margin, (y1 - y0) * margin
        cx0 = max(0, math.floor(x0 - pad_w))
        cy0 = max(0, math.floor(y0 - pad_h))
        cx1 = min(frame_w, math.ceil(x1 + pad_w))
        cy1 = min(frame_h, math.ceil(y1 + pad_h))
        if cx1 - cx0 < 1 or cy1 - cy0 < 1:
            raise ValueError(f"degenerate crop {cx0, cy0, cx1, cy1} from {macro_bbox}")

        crop = image[..., cy0:cy1, cx0:cx1]
        params = compute_letterbox((cx1 - cx0, cy1 - cy0), self.target_size)
        zoom = letterbox_image(crop, params)
        ctx = CropContext(crop_box=(cx0, cy0, cx1, cy1), letterbox=params)
        logger.debug(
            "crop_and_pad: macro=%s margin=%.2f -> crop=%s scale=%.4f pad=(%d, %d)",
            [round(v, 1) for v in (x0, y0, x1, y1)], margin,
            ctx.crop_box, params.scale, params.pad_x, params.pad_y,
        )
        return zoom, ctx

    def project_coordinates(
        self, micro_bbox: torch.Tensor, ctx: CropContext
    ) -> torch.Tensor:
        """Project a zoom-space box back into absolute frame coordinates.

        Inverse letterbox (undo scale and padding, clamp into the crop),
        then translate by the crop's frame-space origin.
        """
        local = unletterbox_boxes(micro_bbox, ctx.letterbox)
        offset = local.new_tensor(
            [ctx.crop_box[0], ctx.crop_box[1], ctx.crop_box[0], ctx.crop_box[1]]
        )
        projected = local + offset
        logger.debug(
            "project: zoom=%s -> crop-local=%s -> frame=%s",
            [round(v, 1) for v in micro_bbox.tolist()],
            [round(v, 1) for v in local.tolist()],
            [round(v, 1) for v in projected.tolist()],
        )
        return projected

    # ---------------------------------------------------------------- forward

    def forward(self, image: torch.Tensor, query: str) -> torch.Tensor | None:
        """Full loop: macro -> crop-and-zoom -> micro -> project.

        Falls back to the macro box when the micro pass fails; returns
        ``None`` only when even the macro pass produces no parseable box.
        """
        macro = self.macro_step(image, query)
        if macro is None:
            return None

        frame_w, frame_h = self.target_size
        macro = macro.clamp(
            min=macro.new_tensor([0.0, 0.0, 0.0, 0.0]),
            max=macro.new_tensor([frame_w, frame_h, frame_w, frame_h]),
        )
        if (
            macro[2] - macro[0] < _MIN_ZOOM_EXTENT
            or macro[3] - macro[1] < _MIN_ZOOM_EXTENT
        ):
            logger.debug("macro box %s too small to zoom; returning as-is",
                         macro.tolist())
            return macro

        zoom, ctx = self.crop_and_pad(image, macro)
        micro = self.micro_step(zoom, query)
        if micro is None:
            logger.warning("micro pass failed; falling back to macro box")
            return macro
        return self.project_coordinates(micro, ctx)

    __call__ = forward

    # ---------------------------------------------------------------- helpers

    def _check_frame(self, image: torch.Tensor) -> None:
        h, w = image.shape[-2:]
        if (w, h) != self.target_size:
            raise ValueError(
                f"expected a {self.target_size} letterboxed frame, got {w}x{h}"
            )

"""Letterboxing and bounding-box coordinate transforms for GUI grounding.

Coordinate convention, used everywhere in this repo:

* Boxes are ``[x0, y0, x1, y1]`` with ``x0 < x1`` and ``y0 < y1``.
* Origin is the top-left corner of the image; units are absolute pixels
  unless a function explicitly says otherwise.

The letterbox maps an arbitrary-size screenshot into the fixed 1024x768
PointerBench frame without distortion: one uniform scale factor, then
centred zero-padding on the short axis. Boxes go through the identical
affine map (``x' = x * scale + pad_x``), and :func:`unletterbox_boxes`
inverts it exactly, so predictions made in the padded frame project back
to original-screenshot pixels with sub-pixel error only from the integer
rounding of the pad offsets (<= 0.5 px).

This module also holds the pandas-level metadata preprocessing for
OS-Atlas: flattening raw annotation records into one row per element and
filtering out degenerate elements before any PyTorch ``Dataset`` is built.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import pandas as pd
import torch
import torch.nn.functional as F

#: Columns of the normalized metadata frame produced by `records_to_frame`.
#: Box columns are absolute pixels in the *original* screenshot.
FRAME_COLUMNS = ("image_path", "query", "x0", "y0", "x1", "y1", "img_w", "img_h")


@dataclass(frozen=True)
class LetterboxParams:
    """Affine parameters of one letterbox: original frame -> target frame.

    Forward map for any point: ``x' = x * scale + pad_x``,
    ``y' = y * scale + pad_y``. The scaled image content occupies the
    ``content_size`` region starting at ``(pad_x, pad_y)``; everything
    outside it is zero padding.
    """

    scale: float
    pad_x: int
    pad_y: int
    content_size: tuple[int, int]  # (width, height) of the scaled content
    orig_size: tuple[int, int]     # (width, height) of the source image
    target_size: tuple[int, int]   # (width, height) of the padded output


def compute_letterbox(
    orig_size: tuple[int, int], target_size: tuple[int, int]
) -> LetterboxParams:
    """Compute the aspect-preserving scale and centred padding offsets.

    Args:
        orig_size: Source image size as ``(width, height)``.
        target_size: Output frame size as ``(width, height)``, e.g. ``(1024, 768)``.

    Returns:
        The affine parameters shared by image and box transforms.
    """
    orig_w, orig_h = orig_size
    target_w, target_h = target_size
    if min(orig_w, orig_h, target_w, target_h) <= 0:
        raise ValueError(f"non-positive size: orig={orig_size}, target={target_size}")

    scale = min(target_w / orig_w, target_h / orig_h)
    content_w = min(target_w, round(orig_w * scale))
    content_h = min(target_h, round(orig_h * scale))
    return LetterboxParams(
        scale=scale,
        pad_x=(target_w - content_w) // 2,
        pad_y=(target_h - content_h) // 2,
        content_size=(content_w, content_h),
        orig_size=(orig_w, orig_h),
        target_size=(target_w, target_h),
    )


def letterbox_image(
    image: torch.Tensor, params: LetterboxParams, fill: float = 0.0
) -> torch.Tensor:
    """Resize ``image`` uniformly and pad it into the target frame.

    Args:
        image: ``(C, H, W)`` or ``(B, C, H, W)`` float tensor whose spatial
            dims match ``params.orig_size``.
        params: Output of :func:`compute_letterbox` for this image size.
        fill: Padding value (0.0 = the required zero-padding).

    Returns:
        Tensor with the same leading dims and spatial size
        ``params.target_size`` (as ``H x W``).
    """
    squeeze = image.dim() == 3
    if squeeze:
        image = image.unsqueeze(0)
    h, w = image.shape[-2:]
    if (w, h) != params.orig_size:
        raise ValueError(f"image is {w}x{h}, params expect {params.orig_size}")

    content_w, content_h = params.content_size
    target_w, target_h = params.target_size
    out = F.interpolate(
        image, size=(content_h, content_w), mode="bilinear", antialias=True
    )
    out = F.pad(
        out,
        (
            params.pad_x,
            target_w - content_w - params.pad_x,
            params.pad_y,
            target_h - content_h - params.pad_y,
        ),
        value=fill,
    )
    return out.squeeze(0) if squeeze else out


def letterbox_boxes(boxes: torch.Tensor, params: LetterboxParams) -> torch.Tensor:
    """Map ``[x0, y0, x1, y1]`` boxes from original into letterboxed coords.

    Args:
        boxes: ``(..., 4)`` tensor in original-screenshot pixels.
        params: The same params used to letterbox the image.

    Returns:
        ``(..., 4)`` float tensor in target-frame pixels, clamped to the
        content region so rounding never leaks a box into the padding.
    """
    out = boxes.to(torch.float32) * params.scale
    pad = boxes.new_tensor(
        [params.pad_x, params.pad_y, params.pad_x, params.pad_y], dtype=torch.float32
    )
    out = out + pad
    content_w, content_h = params.content_size
    lo = pad.clone()
    hi = pad + pad.new_tensor([content_w, content_h, content_w, content_h])
    return out.clamp(min=lo, max=hi)


def unletterbox_boxes(boxes: torch.Tensor, params: LetterboxParams) -> torch.Tensor:
    """Inverse of :func:`letterbox_boxes`: target-frame -> original pixels.

    Predictions that fall inside the padding are clamped onto the nearest
    edge of the original image.
    """
    pad = boxes.new_tensor(
        [params.pad_x, params.pad_y, params.pad_x, params.pad_y], dtype=torch.float32
    )
    out = (boxes.to(torch.float32) - pad) / params.scale
    orig_w, orig_h = params.orig_size
    hi = out.new_tensor([orig_w, orig_h, orig_w, orig_h])
    return out.clamp(min=torch.zeros_like(hi), max=hi)


def records_to_frame(
    records: Iterable[Mapping], *, bbox_format: str = "normalized"
) -> pd.DataFrame:
    """Flatten raw OS-Atlas annotation records into the metadata frame.

    Each record must carry ``img_filename``, ``instruction`` and ``bbox``
    (``[left, top, right, bottom]``). With ``bbox_format="normalized"``
    (the OS-Atlas convention: 0-1 ratios of width/height) it must also
    carry ``img_w``/``img_h`` — the download step reads these once from
    the image headers — so boxes can be converted to absolute pixels here.

    Returns:
        DataFrame with :data:`FRAME_COLUMNS`; boxes in absolute pixels.
    """
    if bbox_format not in ("normalized", "absolute"):
        raise ValueError(f"unknown bbox_format: {bbox_format!r}")

    rows = []
    for rec in records:
        x0, y0, x1, y1 = (float(v) for v in rec["bbox"])
        img_w, img_h = int(rec["img_w"]), int(rec["img_h"])
        if bbox_format == "normalized":
            x0, x1 = (min(max(v, 0.0), 1.0) * img_w for v in (x0, x1))
            y0, y1 = (min(max(v, 0.0), 1.0) * img_h for v in (y0, y1))
        rows.append(
            (str(rec["img_filename"]), str(rec["instruction"]), x0, y0, x1, y1, img_w, img_h)
        )
    return pd.DataFrame(rows, columns=list(FRAME_COLUMNS))


def filter_frame(
    df: pd.DataFrame,
    *,
    min_box_px: float = 5.0,
    min_query_chars: int = 3,
    max_query_chars: int | None = None,
) -> pd.DataFrame:
    """Drop degenerate elements before Dataset construction.

    Removes rows whose box is narrower or shorter than ``min_box_px``
    (original-screenshot pixels, i.e. before any letterbox downscale) and
    rows whose stripped query is shorter than ``min_query_chars`` or —
    when ``max_query_chars`` is set — longer than it (used to keep
    document datasets at line-item granularity rather than paragraphs).

    Returns:
        A filtered copy with a reset index.
    """
    width = df["x1"] - df["x0"]
    height = df["y1"] - df["y0"]
    query_len = df["query"].str.strip().str.len()
    keep = (width >= min_box_px) & (height >= min_box_px) & (query_len >= min_query_chars)
    if max_query_chars is not None:
        keep &= query_len <= max_query_chars
    return df.loc[keep].reset_index(drop=True)

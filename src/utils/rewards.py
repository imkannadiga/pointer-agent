"""GRPO reward functions mirroring PointerBench-Text's asymmetric scoring.

PointerBench-Text scores a predicted box as correct only when
``coverage = |pred ∩ target| / |target| >= 0.90`` **and**
``precision = |pred ∩ target| / |pred| >= 0.70``. Training mirrors that
with two reward channels trl logs separately (``rewards/<name>/mean``):

* :func:`format_reward_func` — small bonus for emitting exactly the
  ``<box> [x0, y0, x1, y1] </box>`` contract and nothing else.
* :func:`asymmetric_bbox_reward_func` — the PointerBench rule: a flat
  penalty whenever coverage falls below the 0.90 floor (clipping the
  target text must never pay off), otherwise coverage-weighted quality.

Both follow trl ``GRPOTrainer``'s calling convention: invoked with
``completions`` plus every extra dataset column as keyword arguments
(ground truth arrives via the ``target_bbox`` column), returning one float
per completion. All boxes are ``[x0, y0, x1, y1]`` in absolute pixels of
the letterboxed 1024x768 frame (see ``src/data/transforms.py``).
"""

from __future__ import annotations

import re
from typing import Sequence

import torch

#: Matches the model's contract from ``conf/model/*.yaml``:
#: ``<box> [x0, y0, x1, y1] </box>`` with int or float coordinates.
_BOX_PATTERN = re.compile(
    r"<box>\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,"
    r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]\s*</box>"
)


def parse_box(text: str) -> torch.Tensor | None:
    """Extract the first ``<box> [x0, y0, x1, y1] </box>`` from ``text``.

    Returns:
        ``(4,)`` float32 tensor, or ``None`` when no box string is found.
    """
    match = _BOX_PATTERN.search(text)
    if match is None:
        return None
    return torch.tensor([float(g) for g in match.groups()], dtype=torch.float32)


def box_coverage_precision(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7
) -> tuple[torch.Tensor, torch.Tensor]:
    """Coverage and precision of ``pred`` w.r.t. ``target``, broadcasting.

    Args:
        pred: ``(..., 4)`` predicted boxes.
        target: ``(..., 4)`` ground-truth boxes, broadcastable with ``pred``.
        eps: Division guard; degenerate boxes (non-positive extent) have
            zero area and therefore zero coverage/precision.

    Returns:
        ``(coverage, precision)`` tensors of the broadcast batch shape,
        each in ``[0, 1]``.
    """
    pred = pred.to(torch.float32)
    target = target.to(torch.float32)

    inter_w = (torch.minimum(pred[..., 2], target[..., 2])
               - torch.maximum(pred[..., 0], target[..., 0])).clamp_min(0)
    inter_h = (torch.minimum(pred[..., 3], target[..., 3])
               - torch.maximum(pred[..., 1], target[..., 1])).clamp_min(0)
    inter = inter_w * inter_h

    pred_area = (pred[..., 2] - pred[..., 0]).clamp_min(0) * (
        pred[..., 3] - pred[..., 1]
    ).clamp_min(0)
    target_area = (target[..., 2] - target[..., 0]).clamp_min(0) * (
        target[..., 3] - target[..., 1]
    ).clamp_min(0)

    coverage = inter / target_area.clamp_min(eps)
    precision = inter / pred_area.clamp_min(eps)
    return coverage, precision


def format_reward_func(
    completions: Sequence, *, format_reward: float = 0.5, **kwargs: object
) -> list[float]:
    """``format_reward`` when the completion is exactly one box string.

    Strict check: the whole completion (modulo surrounding whitespace) must
    be a single ``<box> [x0, y0, x1, y1] </box>`` — trailing prose or a
    second box scores 0.0, shaping the model toward clean parseable output.
    """
    rewards = []
    for completion in completions:
        text = _completion_text(completion).strip()
        rewards.append(
            float(format_reward) if _BOX_PATTERN.fullmatch(text) else 0.0
        )
    return rewards


def asymmetric_bbox_reward_func(
    completions: Sequence,
    target_bbox: Sequence[Sequence[float]],
    *,
    coverage_weight: float = 0.6,
    precision_weight: float = 0.4,
    coverage_floor: float = 0.90,
    floor_penalty: float = -1.0,
    parse_failure_reward: float = 0.0,
    **kwargs: object,
) -> list[float]:
    """The PointerBench rule as a dense reward, one float per completion.

    Per completion: no parseable box -> ``parse_failure_reward``;
    ``coverage < coverage_floor`` -> flat ``floor_penalty`` (never clip the
    text); otherwise ``coverage_weight * coverage + precision_weight *
    precision``. The precision term is what stops "predict the whole
    screen" from gaming the coverage floor.
    """
    rewards = []
    for completion, target in zip(completions, target_bbox):
        pred = parse_box(_completion_text(completion))
        if pred is None:
            rewards.append(float(parse_failure_reward))
            continue
        coverage, precision = box_coverage_precision(
            pred, torch.tensor(list(target), dtype=torch.float32)
        )
        if coverage.item() < coverage_floor:
            rewards.append(float(floor_penalty))
        else:
            rewards.append(
                coverage_weight * coverage.item()
                + precision_weight * precision.item()
            )
    return rewards


def _completion_text(completion: object) -> str:
    """Normalize a trl completion (plain string or chat messages) to text."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, Sequence):  # [{"role": ..., "content": ...}, ...]
        return "".join(str(m.get("content", "")) for m in completion)
    return str(completion)

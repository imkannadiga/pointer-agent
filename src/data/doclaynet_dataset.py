"""DocLayNet financial-report text cells as grounding samples.

Bridges the distribution gap between OS-Atlas (web/mobile/desktop UI) and
pointerbench-text's dense documents and invoices: each sample asks the
model to ground one PDF text cell (a line item, header or table entry) on
a rendered report page.

Uses ``docling-project/DocLayNet-v1.1`` — the parquet re-release of
DocLayNet. The plain ``docling-project/DocLayNet`` repo is a legacy
script-only dataset that modern ``datasets`` refuses to load, and its
records carry no text; v1.1 adds ``pdf_cells`` (per-cell ``bbox`` +
``text``), which is exactly the line-item granularity needed here.

Cell bboxes are ``[x, y, width, height]`` in the page's COCO image space
(1025x1025). Pages whose cells overflow that frame are treated as stored
in original-PDF points and rescaled — a defensive guard, since the corpus
mixes conversion vintages.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.dataset import (
    DEFAULT_BOX_FORMAT,
    DEFAULT_PROMPT_TEMPLATE,
    encode_grpo_sample,
    encode_sft_sample,
    letterbox_sample,
    pil_to_tensor,
)
from src.data.transforms import FRAME_COLUMNS

logger = logging.getLogger(__name__)


def build_cell_frame(examples: Iterable[Mapping]) -> pd.DataFrame:
    """Flatten DocLayNet pages into one row per text cell.

    Args:
        examples: Mappings with ``pdf_cells`` (nested per-region cell lists)
            and ``metadata`` (``coco_width``/``coco_height`` and original
            page size), already filtered to the wanted ``doc_category``.

    Returns:
        ``FRAME_COLUMNS`` frame: ``image_path`` holds the example's ordinal
        index (images live in the HF dataset, not on disk), boxes are
        ``[x0, y0, x1, y1]`` absolute pixels in COCO page space, clamped to
        the page.
    """
    rows = []
    rescaled = 0
    for index, example in enumerate(examples):
        meta = example["metadata"]
        page_w, page_h = int(meta["coco_width"]), int(meta["coco_height"])
        cells = [cell for region in example["pdf_cells"] for cell in region]
        if not cells:
            continue
        # Overflowing cells mean this page stores them in original-PDF
        # points rather than COCO image space -> rescale onto the page.
        max_x = max(c["bbox"][0] + c["bbox"][2] for c in cells)
        max_y = max(c["bbox"][1] + c["bbox"][3] for c in cells)
        scale_x = scale_y = 1.0
        if (max_x > page_w * 1.05 or max_y > page_h * 1.05) and meta.get(
            "original_width"
        ):
            scale_x = page_w / float(meta["original_width"])
            scale_y = page_h / float(meta["original_height"])
            rescaled += 1
        for cell in cells:
            text = (cell.get("text") or "").strip()
            if not text:
                continue
            x, y, w, h = cell["bbox"]
            x0 = min(max(x * scale_x, 0.0), page_w)
            y0 = min(max(y * scale_y, 0.0), page_h)
            x1 = min(max((x + w) * scale_x, 0.0), page_w)
            y1 = min(max((y + h) * scale_y, 0.0), page_h)
            rows.append((str(index), text, x0, y0, x1, y1, page_w, page_h))
    if rescaled:
        logger.warning("%d pages had original-space cells and were rescaled", rescaled)
    return pd.DataFrame(rows, columns=list(FRAME_COLUMNS))


def load_doclaynet_frame(doc_cfg: Any) -> tuple[pd.DataFrame, Any]:
    """Cell metadata frame + image-bearing HF dataset, fetched on demand.

    First call downloads the parquet shards, keeps only ``doc_category``
    pages, persists that subset under ``local_dir`` and caches the
    flattened cell frame as parquet; later calls read both caches. Delete
    ``local_dir`` to rebuild.
    """
    from datasets import load_dataset, load_from_disk

    local_dir = Path(doc_cfg.local_dir)
    subset_dir = local_dir / f"{doc_cfg.doc_category}_{doc_cfg.split}"
    metadata_path = local_dir / "metadata.parquet"

    if subset_dir.exists():
        hf_dataset = load_from_disk(str(subset_dir))
    else:
        logger.info("downloading %s (first use)", doc_cfg.hf_repo)
        full = load_dataset(str(doc_cfg.hf_repo), split=str(doc_cfg.split))
        hf_dataset = full.filter(
            lambda meta: meta["doc_category"] == doc_cfg.doc_category,
            input_columns=["metadata"],
        )
        subset_dir.parent.mkdir(parents=True, exist_ok=True)
        hf_dataset.save_to_disk(str(subset_dir))
        logger.info(
            "kept %d/%d %s pages", len(hf_dataset), len(full), doc_cfg.doc_category
        )

    if metadata_path.exists():
        frame = pd.read_parquet(metadata_path)
    else:
        # Column-select so building the index never decodes an image.
        frame = build_cell_frame(hf_dataset.select_columns(["pdf_cells", "metadata"]))
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(metadata_path)
        logger.info("cell index cached to %s (%d rows)", metadata_path, len(frame))
    return frame, hf_dataset


class DocLayNetGroundingDataset(Dataset):
    """DocLayNet text cells in the repo's shared grounding-sample shapes.

    Uses the same encoders as the OS-Atlas datasets, so ``mode="sft"``
    items are structurally identical to ``OSAtlasDataset`` items (one
    collator batches a mixture) and ``mode="grpo"`` rows match
    ``OSAtlasGRPODataset`` (safe under ``GRPOTrainer``).

    Args:
        frame: Filtered output of :func:`load_doclaynet_frame`.
        hf_dataset: Indexable holding the page images; ``frame.image_path``
            values are ordinal indices into it.
        processor: VLM processor; required when ``mode="sft"``.
        mode: ``"sft"`` (tokenized items) or ``"grpo"`` (trl rows).
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        hf_dataset: Any,
        processor: Any = None,
        *,
        mode: str = "sft",
        target_size: tuple[int, int] = (1024, 768),
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        box_format: str = DEFAULT_BOX_FORMAT,
        include_answer: bool = True,
    ) -> None:
        if mode not in ("sft", "grpo"):
            raise ValueError(f"unknown mode: {mode!r}")
        if mode == "sft" and processor is None:
            raise ValueError("mode='sft' requires a processor")
        self.frame = frame.reset_index(drop=True)
        self.hf_dataset = hf_dataset
        self.processor = processor
        self.mode = mode
        self.target_size = (int(target_size[0]), int(target_size[1]))
        self.prompt_template = prompt_template
        self.box_format = box_format
        self.include_answer = include_answer

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        image = pil_to_tensor(self.hf_dataset[int(row["image_path"])]["image"])
        framed, bbox = letterbox_sample(
            image,
            (int(row["img_w"]), int(row["img_h"])),
            torch.tensor(
                [row["x0"], row["y0"], row["x1"], row["y1"]], dtype=torch.float32
            ),
            self.target_size,
        )
        query = str(row["query"])
        if self.mode == "grpo":
            return encode_grpo_sample(
                framed, query, bbox, prompt_template=self.prompt_template
            )
        return encode_sft_sample(
            self.processor,
            framed,
            query,
            bbox,
            prompt_template=self.prompt_template,
            box_format=self.box_format,
            include_answer=self.include_answer,
        )

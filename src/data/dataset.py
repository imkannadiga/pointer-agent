"""OS-Atlas PyTorch dataset: letterboxed screenshots + box-string labels.

Every sample is letterboxed into the fixed benchmark frame (default
1024x768) and its ground-truth box is mapped through the identical affine,
so the model only ever sees and speaks one coordinate space: absolute
pixels of that frame, top-left origin. The VLM's own processor may
internally re-patch the image (e.g. Qwen2.5-VL snaps to 28-px multiples);
that is fine as long as train and inference share this processor, because
SFT teaches the frame-coordinate contract end-to-end.

Batching note: ``__getitem__`` returns unpadded, variable-length token
sequences; the training scripts own the padding collator.
"""

from __future__ import annotations

import json
import logging
import shutil
import zipfile
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import Dataset

from src.data.transforms import (
    compute_letterbox,
    filter_frame,
    letterbox_boxes,
    letterbox_image,
    records_to_frame,
)

logger = logging.getLogger(__name__)

#: Instruction shown to the VLM; must stay identical between SFT, GRPO and
#: inference. Overridable via ``conf/model/*.yaml``.
DEFAULT_PROMPT_TEMPLATE = (
    'Find the bounding box for the text: "{query}". '
    "Output format: <box> [x0, y0, x1, y1] </box>"
)
#: Label/output contract; must agree with the parser in ``src/utils/rewards.py``.
DEFAULT_BOX_FORMAT = "<box> [{x0}, {y0}, {x1}, {y1}] </box>"

IGNORE_INDEX = -100


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL image -> RGB float32 ``(3, H, W)`` tensor in ``[0, 1]``."""
    arr = np.array(img.convert("RGB"), dtype=np.uint8)  # owns its memory
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0


def load_image(path: str | Path) -> torch.Tensor:
    """Read an image file as an RGB float32 ``(3, H, W)`` tensor in ``[0, 1]``."""
    with Image.open(path) as img:
        return pil_to_tensor(img)


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """Float ``(3, H, W)`` tensor in ``[0, 1]`` -> PIL RGB image."""
    arr = (image.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    return Image.fromarray(arr.permute(1, 2, 0).cpu().numpy())


def build_messages(
    query: str,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    answer: str | None = None,
) -> list[dict[str, Any]]:
    """Chat messages for one grounding turn: image + instruction, opt. answer."""
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_template.format(query=query)},
            ],
        }
    ]
    if answer is not None:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        )
    return messages


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _ensure_source_images(
    hf_repo: str, source: Any, image_dir: Path, tmp_dir: Path
) -> Path:
    """Download and extract one source's image archives (idempotent).

    Split archives (multiple files, or files without a ``.zip`` suffix —
    OS-Atlas ships e.g. ``windows_image_aa..ad``) are concatenated into a
    single zip before extraction. A ``.extracted`` marker makes re-runs
    no-ops; downloads themselves resume via the HF cache.
    """
    extract_dir = image_dir / str(source.name)
    marker = extract_dir / ".extracted"
    if marker.exists():
        return extract_dir

    paths = [
        Path(
            hf_hub_download(hf_repo, str(archive), repo_type="dataset")
        )
        for archive in source.archives
    ]
    if len(paths) == 1 and paths[0].suffix == ".zip":
        zip_path, concatenated = paths[0], False
    else:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        zip_path, concatenated = tmp_dir / f"{source.name}.zip", True
        logger.info("concatenating %d split parts for %s", len(paths), source.name)
        with open(zip_path, "wb") as out:
            for part in paths:
                with open(part, "rb") as f:
                    shutil.copyfileobj(f, out)

    logger.info("extracting %s -> %s", zip_path.name, extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    if concatenated:
        zip_path.unlink()
    marker.touch()
    return extract_dir


def _index_images(extract_dir: Path, image_dir: Path) -> dict[str, str]:
    """Map image basename -> path relative to ``image_dir``.

    Annotation ``img_filename`` values are bare basenames while archives may
    nest arbitrarily, so resolution goes through this index.
    """
    index: dict[str, str] = {}
    duplicates = 0
    for path in extract_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
            if path.name in index:
                duplicates += 1
            else:
                index[path.name] = str(path.relative_to(image_dir))
    if duplicates:
        logger.warning(
            "%s: %d duplicate basenames; first occurrence wins", extract_dir, duplicates
        )
    return index


def _iter_elements(entry: dict) -> Iterator[tuple[Any, Any]]:
    """Yield ``(instruction, bbox)`` from nested or flat annotation entries."""
    for element in entry.get("elements", [entry]):
        yield element.get("instruction"), element.get("bbox")


def load_os_atlas_frame(train_cfg: Any) -> pd.DataFrame:
    """Metadata frame for the configured OS-Atlas sources, fetched on demand.

    First call downloads each source's annotation JSON and image archives
    from the HF Hub, extracts them under ``image_dir``, reads every image's
    size (needed to de-normalize OS-Atlas' 0-1 bboxes), and caches the
    resulting frame at ``metadata_file``. Later calls just read the parquet;
    delete it after changing ``sources`` to trigger a rebuild.

    Returns:
        Unfiltered ``FRAME_COLUMNS`` frame, boxes in absolute pixels,
        ``image_path`` relative to ``image_dir``.
    """
    metadata_path = Path(train_cfg.metadata_file)
    if metadata_path.exists():
        frame = pd.read_parquet(metadata_path)
        logger.info("loaded cached metadata: %s (%d rows)", metadata_path, len(frame))
        return frame

    image_dir = Path(train_cfg.image_dir)
    records: list[dict] = []
    for source in train_cfg.sources:
        annotation_path = hf_hub_download(
            str(train_cfg.hf_repo), str(source.annotations), repo_type="dataset"
        )
        extract_dir = _ensure_source_images(
            str(train_cfg.hf_repo), source, image_dir, Path(train_cfg.local_dir) / "tmp"
        )
        index = _index_images(extract_dir, image_dir)

        with open(annotation_path) as f:
            entries = json.load(f)
        size_cache: dict[str, tuple[int, int]] = {}
        missing = skipped = 0
        start = len(records)
        for entry in entries:
            rel = index.get(Path(str(entry.get("img_filename", ""))).name)
            if rel is None:
                missing += 1
                continue
            if rel not in size_cache:
                with Image.open(image_dir / rel) as img:
                    size_cache[rel] = img.size
            width, height = size_cache[rel]
            for instruction, bbox in _iter_elements(entry):
                if not instruction or not isinstance(bbox, list) or len(bbox) != 4:
                    skipped += 1
                    continue
                records.append(
                    dict(img_filename=rel, instruction=instruction, bbox=bbox,
                         img_w=width, img_h=height)
                )
        logger.info(
            "source %s: %d elements from %d screenshots (%d entries without an "
            "image file, %d malformed elements)",
            source.name, len(records) - start, len(size_cache), missing, skipped,
        )

    frame = records_to_frame(records, bbox_format=str(train_cfg.bbox_format))
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(metadata_path)
    logger.info("metadata cached to %s (%d rows)", metadata_path, len(frame))
    return frame


def letterbox_sample(
    image: torch.Tensor,
    orig_size: tuple[int, int],
    bbox: torch.Tensor,
    target_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Letterbox one image and its box into the benchmark frame."""
    params = compute_letterbox(orig_size, target_size)
    return letterbox_image(image, params), letterbox_boxes(bbox, params)


def encode_sft_sample(
    processor: Any,
    framed: torch.Tensor,
    query: str,
    bbox: torch.Tensor,
    *,
    prompt_template: str,
    box_format: str,
    include_answer: bool,
) -> dict[str, torch.Tensor]:
    """Letterboxed frame + query [+ box answer] -> tokenized model inputs.

    Single encoding path shared by every grounding dataset, so items from
    different corpora are structurally identical by construction and any
    mixture batches cleanly through one collator. See ``OSAtlasDataset``
    for the returned dictionary contract.
    """
    pil = tensor_to_pil(framed)
    prompt_text = processor.apply_chat_template(
        build_messages(query, prompt_template),
        tokenize=False,
        add_generation_prompt=True,
    )
    if include_answer:
        answer = box_format.format(
            x0=round(bbox[0].item()),
            y0=round(bbox[1].item()),
            x1=round(bbox[2].item()),
            y1=round(bbox[3].item()),
        )
        full_text = processor.apply_chat_template(
            build_messages(query, prompt_template, answer=answer),
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        full_text = prompt_text

    encoded = dict(processor(text=[full_text], images=[pil], return_tensors="pt"))
    item: dict[str, torch.Tensor] = {}
    for key, value in encoded.items():
        # Drop the batch dim on the per-token streams; leave model-specific
        # extras (pixel_values layouts, image_grid_thw, ...) untouched.
        if key in ("input_ids", "attention_mask", "mm_token_type_ids"):
            value = value.squeeze(0)
        item[key] = value

    if include_answer:
        # The rendered prompt is a strict text prefix of the full
        # conversation (it ends exactly at the generation header), so its
        # token length is a valid mask boundary.
        prompt_len = processor(text=[prompt_text], images=[pil], return_tensors="pt")[
            "input_ids"
        ].shape[1]
        labels = item["input_ids"].clone()
        labels[:prompt_len] = IGNORE_INDEX
        item["labels"] = labels

    item["bbox"] = bbox
    return item


def encode_grpo_sample(
    framed: torch.Tensor, query: str, bbox: torch.Tensor, *, prompt_template: str
) -> dict[str, Any]:
    """Letterboxed frame + query -> a trl ``GRPOTrainer`` row.

    Plain-string conversational content — trl injects the image block and
    applies the chat template itself; ``target_bbox`` is forwarded verbatim
    to every reward function's keyword arguments.
    """
    return {
        "prompt": [{"role": "user", "content": prompt_template.format(query=query)}],
        "image": tensor_to_pil(framed),
        "target_bbox": bbox.tolist(),
    }


class OSAtlasDataset(Dataset):
    """Grounding samples from a filtered OS-Atlas metadata frame.

    Args:
        frame: Output of ``transforms.filter_frame`` — one element per row,
            boxes in absolute original-screenshot pixels.
        image_dir: Root directory the frame's ``image_path`` values are
            relative to.
        processor: The VLM's ``AutoProcessor`` (wraps tokenizer + image
            processor; both requested processors travel together in HF VLMs).
        target_size: Benchmark frame as ``(width, height)``.
        prompt_template: User-turn instruction; ``{query}`` is substituted.
        box_format: Assistant-turn label contract; coordinates are rounded
            to integer frame pixels.
        include_answer: ``True`` -> SFT mode: the sequence contains the
            assistant answer and a ``labels`` tensor with every prompt token
            masked to ``IGNORE_INDEX``. ``False`` -> prompt-only mode ending
            in the generation header (GRPO rollouts / inference).
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        image_dir: str | Path,
        processor: Any,
        *,
        target_size: tuple[int, int] = (1024, 768),
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        box_format: str = DEFAULT_BOX_FORMAT,
        include_answer: bool = True,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.processor = processor
        self.target_size = (int(target_size[0]), int(target_size[1]))
        self.prompt_template = prompt_template
        self.box_format = box_format
        self.include_answer = include_answer

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        framed, bbox = letterbox_sample(
            load_image(self.image_dir / str(row["image_path"])),
            (int(row["img_w"]), int(row["img_h"])),
            torch.tensor(
                [row["x0"], row["y0"], row["x1"], row["y1"]], dtype=torch.float32
            ),
            self.target_size,
        )
        return encode_sft_sample(
            self.processor,
            framed,
            str(row["query"]),
            bbox,
            prompt_template=self.prompt_template,
            box_format=self.box_format,
            include_answer=self.include_answer,
        )


class OSAtlasGRPODataset(Dataset):
    """Rows in trl ``GRPOTrainer``'s expected shape for VLM rollouts.

    Each row carries a conversational ``prompt`` (plain-string content —
    trl injects the image block and applies the chat template itself), the
    letterboxed frame as a PIL ``image``, and the ground-truth box as a
    ``target_bbox`` column, which trl forwards verbatim to every reward
    function's keyword arguments.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        image_dir: str | Path,
        *,
        target_size: tuple[int, int] = (1024, 768),
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.target_size = (int(target_size[0]), int(target_size[1]))
        self.prompt_template = prompt_template

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        framed, bbox = letterbox_sample(
            load_image(self.image_dir / str(row["image_path"])),
            (int(row["img_w"]), int(row["img_h"])),
            torch.tensor(
                [row["x0"], row["y0"], row["x1"], row["y1"]], dtype=torch.float32
            ),
            self.target_size,
        )
        return encode_grpo_sample(
            framed, str(row["query"]), bbox, prompt_template=self.prompt_template
        )


class MixedGroundingDataset(Dataset):
    """Ratio-controlled mixture of grounding datasets.

    The ratio is baked into a deterministic index map instead of a custom
    sampler: transformers/trl trainers build their own dataloaders and
    samplers (random + distributed), so a ``WeightedRandomSampler`` handed
    to them would simply be discarded. Mapping indices below the sampler
    keeps the mix exact under any sampler, seed-stable, and DDP-safe.

    One epoch anchors on child 0 (every sample seen exactly once, in
    seed-shuffled order); the remaining children are resampled — repeating
    with wrap-around, or subsampled — to hit their configured share.
    """

    def __init__(
        self, datasets: Sequence[Dataset], weights: Sequence[float], seed: int = 0
    ) -> None:
        if len(datasets) != len(weights) or not datasets:
            raise ValueError("need one weight per dataset")
        if min(weights) <= 0:
            raise ValueError(f"weights must be positive, got {list(weights)}")
        total = float(sum(weights))
        shares = [w / total for w in weights]

        rng = np.random.default_rng(seed)
        epoch_len = round(len(datasets[0]) / shares[0])
        counts = [len(datasets[0])] + [
            max(1, round(epoch_len * s)) for s in shares[1:]
        ]
        entries = []
        for child, count in enumerate(counts):
            order = rng.permutation(len(datasets[child]))
            picks = np.resize(order, count)  # wrap-around when oversampling
            entries.extend((child, int(j)) for j in picks)
        rng.shuffle(entries)

        self.datasets = list(datasets)
        self.index_map = entries
        logger.info(
            "mixed dataset: %s -> %d samples/epoch",
            {f"child{c}": n for c, n in enumerate(counts)}, len(entries),
        )

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, index: int) -> dict:
        child, inner = self.index_map[index]
        return self.datasets[child][inner]


def build_grounding_datasets(
    cfg: Any, processor: Any, *, mode: str = "sft"
) -> tuple[Dataset, Dataset | None]:
    """Assemble the training mixture and the SFT eval set from config.

    OS-Atlas is loaded (downloading on first use), filtered, shuffled and
    split with the same seed/holdout in every stage, so SFT and GRPO train
    on identical rows and neither ever sees the holdout. When
    ``data.mix_ratio`` gives DocLayNet a positive share, its
    financial-report cells are mixed in at that ratio via
    :class:`MixedGroundingDataset`.

    Args:
        cfg: The composed root config.
        processor: VLM processor (unused by GRPO rows but required for SFT).
        mode: ``"sft"`` -> tokenized items + OS-Atlas holdout eval set;
            ``"grpo"`` -> trl rows, eval set is ``None``.

    Returns:
        ``(train_dataset, eval_dataset_or_None)``.
    """
    if mode not in ("sft", "grpo"):
        raise ValueError(f"unknown mode: {mode!r}")
    atlas_cfg = cfg.data.os_atlas
    common = dict(
        target_size=tuple(cfg.data.target_size),
        prompt_template=cfg.model.prompt_template,
    )
    sft_kwargs = dict(common, box_format=cfg.model.box_format)

    frame = load_os_atlas_frame(atlas_cfg)
    frame = filter_frame(
        frame,
        min_box_px=atlas_cfg.min_box_px,
        min_query_chars=atlas_cfg.min_query_chars,
    )
    frame = frame.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    n_holdout = max(1, int(len(frame) * cfg.train_sft.holdout_frac))
    train_frame, holdout_frame = frame.iloc[n_holdout:], frame.iloc[:n_holdout]
    logger.info("OS-Atlas: %d train rows, %d holdout", len(train_frame), n_holdout)

    if mode == "sft":
        atlas_train = OSAtlasDataset(
            train_frame, atlas_cfg.image_dir, processor, **sft_kwargs
        )
        # Eval stays pure OS-Atlas holdout: one stable metric across stages.
        eval_ds: Dataset | None = OSAtlasDataset(
            holdout_frame, atlas_cfg.image_dir, processor, **sft_kwargs
        )
    else:
        atlas_train = OSAtlasGRPODataset(train_frame, atlas_cfg.image_dir, **common)
        eval_ds = None

    weights = [float(w) for w in cfg.data.mix_ratio]
    if len(weights) != 2:
        raise ValueError(f"mix_ratio must be [os_atlas, doclaynet], got {weights}")
    if weights[1] <= 0:
        return atlas_train, eval_ds

    # Deferred import: doclaynet_dataset imports helpers from this module.
    from src.data.doclaynet_dataset import (
        DocLayNetGroundingDataset,
        load_doclaynet_frame,
    )

    doc_cfg = cfg.data.doclaynet
    doc_frame, hf_dataset = load_doclaynet_frame(doc_cfg)
    doc_frame = filter_frame(
        doc_frame,
        min_box_px=doc_cfg.min_box_px,
        min_query_chars=doc_cfg.min_query_chars,
        max_query_chars=doc_cfg.max_query_chars,
    )
    logger.info("DocLayNet: %d cells after filtering", len(doc_frame))
    doclay = DocLayNetGroundingDataset(
        doc_frame,
        hf_dataset,
        processor if mode == "sft" else None,
        mode=mode,
        **(sft_kwargs if mode == "sft" else common),
    )
    return MixedGroundingDataset([atlas_train, doclay], weights, seed=cfg.seed), eval_ds

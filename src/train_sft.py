"""Stage 1 — supervised fine-tuning of the grounding VLM (trl + peft LoRA).

Teaches the base VLM the frame-coordinate contract: letterboxed 1024x768
screenshot + query in, ``<box> [x0, y0, x1, y1] </box>`` out. Labels are
prompt-masked by ``OSAtlasDataset``; this script owns model/LoRA setup,
the padding collator, W&B tracking, and checkpointing.

Run (single GPU)::

    venv/bin/python -m src.train_sft wandb.mode=offline

or under Accelerate for multi-GPU (device placement, DDP and grad sync are
delegated to the trainer/Accelerate; nothing here is rank-specific)::

    accelerate launch -m src.train_sft
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable

import hydra
import torch
import wandb
from accelerate import PartialState
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed
from trl import SFTConfig, SFTTrainer

from src.data.dataset import build_grounding_datasets

logger = logging.getLogger(__name__)


def build_collator(processor: Any) -> Callable[[list[dict]], dict[str, torch.Tensor]]:
    """Right-pad token streams; concatenate per-image vision tensors.

    ``input_ids`` pad with the tokenizer pad id, ``labels`` with -100 so
    padding never contributes loss. Qwen-style processors emit
    ``pixel_values`` as flattened patches ``(n_patches_i, dim)`` and
    ``image_grid_thw`` as ``(1, 3)`` per sample; the model expects both
    concatenated along dim 0 across the batch. ``bbox`` is dropped — it is
    reward/eval metadata, not a model input.
    """
    pad_id = processor.tokenizer.pad_token_id

    def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
        length = max(ex["input_ids"].shape[0] for ex in batch)
        n = len(batch)
        input_ids = torch.full((n, length), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((n, length), dtype=torch.long)
        labels = torch.full((n, length), -100, dtype=torch.long)
        mm_types = torch.zeros((n, length), dtype=torch.long)
        for i, ex in enumerate(batch):
            k = ex["input_ids"].shape[0]
            input_ids[i, :k] = ex["input_ids"]
            attention_mask[i, :k] = ex["attention_mask"]
            labels[i, :k] = ex["labels"]
            if "mm_token_type_ids" in ex:
                mm_types[i, :k] = ex["mm_token_type_ids"]
        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": torch.cat([ex["pixel_values"] for ex in batch], dim=0),
        }
        if "mm_token_type_ids" in batch[0]:
            out["mm_token_type_ids"] = mm_types
        if "image_grid_thw" in batch[0]:
            out["image_grid_thw"] = torch.cat(
                [ex["image_grid_thw"] for ex in batch], dim=0
            )
        return out

    return collate


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    set_seed(cfg.seed)
    run_name = cfg.wandb.run_name or f"sft-{time.strftime('%Y%m%d-%H%M%S')}"

    # W&B: init once on the main process; the trainer streams loss, learning
    # rate and grad_norm into the same run via report_to="wandb".
    track = cfg.wandb.mode != "disabled"
    if track and PartialState().is_main_process:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            mode=cfg.wandb.mode,
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # --- model + LoRA -----------------------------------------------------
    model = AutoModelForImageTextToText.from_pretrained(
        cfg.model.model_id,
        dtype=getattr(torch, cfg.model.dtype),
        attn_implementation=cfg.model.attn_implementation,
    )
    # LoRA over the LM blocks *and* the vision tower/merger (regex in
    # conf/model): grounding accuracy hinges on adapting visual features,
    # not just the language head.
    lora_cfg = LoraConfig(
        r=cfg.model.lora.r,
        lora_alpha=cfg.model.lora.alpha,
        lora_dropout=cfg.model.lora.dropout,
        target_modules=cfg.model.lora.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    if PartialState().is_main_process:
        model.print_trainable_parameters()
    # Required for gradient checkpointing with frozen (LoRA) base weights:
    # inputs must carry grad or checkpointed segments have nothing to backprop.
    model.enable_input_require_grads()

    # --- data -------------------------------------------------------------
    # OS-Atlas + DocLayNet mixture per data.mix_ratio (downloads on first
    # use); eval is the pure OS-Atlas holdout.
    processor = AutoProcessor.from_pretrained(cfg.model.model_id)
    train_ds, eval_ds = build_grounding_datasets(cfg, processor, mode="sft")

    # --- trainer ----------------------------------------------------------
    out_dir = Path(to_absolute_path(cfg.train_sft.output_dir))
    args = SFTConfig(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=cfg.train_sft.epochs,
        per_device_train_batch_size=cfg.train_sft.per_device_batch_size,
        per_device_eval_batch_size=cfg.train_sft.per_device_batch_size,
        gradient_accumulation_steps=cfg.train_sft.grad_accum,
        learning_rate=cfg.train_sft.lr,
        lr_scheduler_type=cfg.train_sft.lr_scheduler,
        warmup_ratio=cfg.train_sft.warmup_ratio,
        weight_decay=cfg.train_sft.weight_decay,
        max_length=cfg.train_sft.max_seq_len,
        bf16=cfg.train_sft.bf16,
        gradient_checkpointing=cfg.train_sft.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=cfg.train_sft.logging_steps,
        save_steps=cfg.train_sft.save_steps,
        eval_strategy="steps",
        eval_steps=cfg.train_sft.save_steps,
        dataloader_num_workers=cfg.train_sft.dataloader_num_workers,
        seed=cfg.seed,
        run_name=run_name,
        report_to="wandb" if track else "none",
        # The dataset is already tokenized/multimodal: keep trl's text
        # pipeline out of the way and let our collator own batching.
        packing=False,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=build_collator(processor),
        processing_class=processor,
    )

    trainer.train()

    # --- save -------------------------------------------------------------
    # save_model on a peft-wrapped model writes only the adapter weights;
    # the processor rides along so inference needs just this directory.
    trainer.save_model(str(out_dir))
    if trainer.accelerator.is_main_process:
        processor.save_pretrained(str(out_dir))
        logger.info("SFT adapter saved to %s", out_dir)


if __name__ == "__main__":
    main()

"""Stage 2 — GRPO over the SFT model, optimizing the PointerBench rule.

Loads the stage-1 SFT adapter, merges it into the base VLM, and trains a
fresh LoRA on top with trl's ``GRPOTrainer``. Merging first matters for
the KL term: with a peft model, trl computes reference logprobs by
disabling the adapter, so the reference policy is exactly the merged SFT
model rather than the raw base.

Rollouts are scored by two reward channels (see ``src/utils/rewards.py``)
which trl logs to W&B separately as ``rewards/format_reward_func/mean``
and ``rewards/asymmetric_bbox_reward_func/mean``.

The final artifact is an adapter over the *merged* model — to run it,
load the base model, merge the SFT adapter, then attach this one.

Run::

    venv/bin/python -m src.train_grpo wandb.mode=offline
    accelerate launch -m src.train_grpo
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

import hydra
import torch
import wandb
from accelerate import PartialState
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed
from trl import GRPOConfig, GRPOTrainer

from src.data.dataset import build_grounding_datasets
from src.utils import rewards

logger = logging.getLogger(__name__)


def bind_reward_funcs(reward_cfg: DictConfig) -> list[Callable[..., list[float]]]:
    """Bind config values into the reward functions, keeping their names.

    trl labels each reward channel by the function's ``__name__``, so the
    wrappers must shadow the originals exactly for W&B metric continuity.
    """

    def format_reward_func(completions, **kwargs):
        return rewards.format_reward_func(
            completions, format_reward=reward_cfg.format_reward, **kwargs
        )

    def asymmetric_bbox_reward_func(completions, target_bbox, **kwargs):
        return rewards.asymmetric_bbox_reward_func(
            completions,
            target_bbox,
            coverage_weight=reward_cfg.coverage_weight,
            precision_weight=reward_cfg.precision_weight,
            coverage_floor=reward_cfg.coverage_floor,
            floor_penalty=reward_cfg.floor_penalty,
            parse_failure_reward=reward_cfg.parse_failure_reward,
            **kwargs,
        )

    return [format_reward_func, asymmetric_bbox_reward_func]


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    set_seed(cfg.seed)
    run_name = cfg.wandb.run_name or f"grpo-{time.strftime('%Y%m%d-%H%M%S')}"

    track = cfg.wandb.mode != "disabled"
    if track and PartialState().is_main_process:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            mode=cfg.wandb.mode,
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # --- model: base + merged SFT adapter, fresh GRPO LoRA ----------------
    sft_dir = Path(to_absolute_path(cfg.train_grpo.sft_model_dir))
    if not sft_dir.exists():
        raise FileNotFoundError(
            f"{sft_dir} not found — run stage 1 first: python -m src.train_sft"
        )
    base = AutoModelForImageTextToText.from_pretrained(
        cfg.model.model_id,
        dtype=getattr(torch, cfg.model.dtype),
        attn_implementation=cfg.model.attn_implementation,
    )
    model = PeftModel.from_pretrained(base, sft_dir).merge_and_unload()
    logger.info("merged SFT adapter from %s", sft_dir)
    lora_cfg = LoraConfig(
        r=cfg.model.lora.r,
        lora_alpha=cfg.model.lora.alpha,
        lora_dropout=cfg.model.lora.dropout,
        target_modules=cfg.model.lora.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # --- data -------------------------------------------------------------
    # Same mixture, shuffle seed and holdout as SFT, so RL trains on the
    # same distribution and never sees the rows SFT validated against.
    processor = AutoProcessor.from_pretrained(cfg.model.model_id)
    train_ds, _ = build_grounding_datasets(cfg, processor, mode="grpo")

    # --- trainer ----------------------------------------------------------
    out_dir = Path(to_absolute_path(cfg.train_grpo.output_dir))
    args = GRPOConfig(
        output_dir=str(out_dir / "checkpoints"),
        max_steps=cfg.train_grpo.max_steps,
        per_device_train_batch_size=cfg.train_grpo.per_device_batch_size,
        gradient_accumulation_steps=cfg.train_grpo.grad_accum,
        learning_rate=cfg.train_grpo.lr,
        num_generations=cfg.train_grpo.num_generations,
        max_completion_length=cfg.train_grpo.max_completion_len,
        temperature=cfg.train_grpo.temperature,
        beta=cfg.train_grpo.kl_beta,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=cfg.train_grpo.logging_steps,
        save_steps=cfg.train_grpo.save_steps,
        seed=cfg.seed,
        run_name=run_name,
        report_to="wandb" if track else "none",
        # target_bbox must survive the dataloader to reach the reward funcs.
        remove_unused_columns=False,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=bind_reward_funcs(cfg.train_grpo.reward),
        args=args,
        train_dataset=train_ds,
        processing_class=processor,
        peft_config=lora_cfg,
    )

    trainer.train()

    # --- save -------------------------------------------------------------
    trainer.save_model(str(out_dir))
    if trainer.accelerator.is_main_process:
        processor.save_pretrained(str(out_dir))
        logger.info("GRPO adapter saved to %s", out_dir)


if __name__ == "__main__":
    main()

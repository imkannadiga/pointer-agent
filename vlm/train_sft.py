"""Phase 1: LoRA-SFT the VLM grounder (Florence-2) on
`<CAPTION_TO_PHRASE_GROUNDING>` phrase -> location-token bbox pairs.

Keeps Florence-2's native output vocabulary (binned `<loc_N>` tokens) rather
than switching to a JSON-coordinate output head: Florence-2 was pretrained on
referring-expression grounding with exactly this discretized location-token
representation, so LoRA-adapting it to small UI/document text is reusing a
pretrained skill, not learning a new output format from scratch. Only the
*input* phrase changes for caret/relative categories (a short referring
expression instead of a literal on-page word) - see
vlm/dataset.py and progress.md for the full rationale.

Only the language-model's attention projections get LoRA adapters (the
DaViT vision tower's `qkv`/`proj` linear names don't collide with
`q_proj`/`k_proj`/`v_proj`/`out_proj`, confirmed by inspecting
model.named_modules() - so this target_modules list can't accidentally touch
the vision tower); the vision tower stays frozen.

Hardware-profile agnostic by construction (see slm/train_sft.py's docstring
for the same accelerate/Trainer argument - it applies unchanged here).

Usage:
    python -m vlm.train_sft hardware=cpu_smoketest data.max_train_samples=32
    accelerate launch --config_file configs/accelerate/single_gpu.yaml \
        -m vlm.train_sft hardware=single_gpu_t4
"""
import os

import hydra
import torch
from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model
from torch.utils.data import Subset
from transformers import AutoModelForCausalLM, AutoProcessor, Trainer, TrainingArguments

from vlm.dataset import GroundingSFTDataset, make_collate_fn


@hydra.main(config_path="../configs", config_name="train/vlm_sft", version_base=None)
def main(cfg: DictConfig):
    hw = cfg.hardware
    if hw.quantization == "4bit" and not torch.cuda.is_available():
        raise RuntimeError(
            "hardware.quantization=4bit requires CUDA (bitsandbytes). "
            "Use hardware=cpu_smoketest for a local CPU run."
        )

    processor = AutoProcessor.from_pretrained(cfg.model_name, trust_remote_code=True)

    model_kwargs = {"trust_remote_code": True, "attn_implementation": "eager", "torch_dtype": torch.float32}
    if hw.quantization == "4bit":
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs.pop("torch_dtype")
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)

    lora_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.target_modules),
        task_type=None,  # Florence-2's forward signature is bespoke, not a stock CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    if hw.gradient_checkpointing:
        # use_reentrant=False: reentrant checkpointing re-executes forward
        # under a wrapped autograd Function during backward, which under
        # DDP (multi_gpu profile) fires per-parameter "gradient ready" hooks
        # more than once for reused parameters - DDP rejects that
        # ("Expected to mark a variable ready only once"). Non-reentrant
        # checkpointing is also the modern PyTorch-recommended default.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # With only the LoRA adapter trainable (base model frozen), the first
        # checkpointed segment can see zero tensors requiring grad at its
        # input (e.g. the frozen vision tower), which makes the whole loss
        # come back with no grad_fn - this call is the standard PEFT fix:
        # it forces the input embeddings/patch-embeddings to require grad so
        # gradient checkpointing has a valid entry point into the graph.
        model.enable_input_require_grads()

    metadata_path = hydra.utils.to_absolute_path(cfg.data.metadata_path)
    dataset = GroundingSFTDataset(metadata_path, max_samples=cfg.data.max_train_samples)
    n_val = max(1, int(len(dataset) * cfg.data.val_fraction)) if len(dataset) > 1 else 0
    train_dataset = Subset(dataset, range(n_val, len(dataset)))
    eval_dataset = Subset(dataset, range(0, n_val)) if n_val else None

    output_dir = hydra.utils.to_absolute_path(cfg.output_dir)
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=hw.per_device_train_batch_size,
        gradient_accumulation_steps=hw.gradient_accumulation_steps,
        num_train_epochs=hw.num_train_epochs,
        max_steps=hw.max_steps,
        learning_rate=cfg.learning_rate,
        logging_steps=cfg.logging_steps,
        fp16=hw.fp16,
        bf16=hw.bf16,
        dataloader_num_workers=hw.dataloader_num_workers,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        use_cpu=(hw.device == "cpu"),
        label_names=["labels"],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=make_collate_fn(processor),
    )
    result = trainer.train()
    print(f"Final train loss: {result.training_loss:.4f}")

    adapter_dir = hydra.utils.to_absolute_path(cfg.save_adapter_dir)
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    print(f"Saved LoRA adapter -> {adapter_dir}")


if __name__ == "__main__":
    main()

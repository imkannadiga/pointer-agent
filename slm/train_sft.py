"""Phase 2: LoRA-SFT the SLM planner on instruction -> structured-query pairs.

Hardware-profile agnostic by construction: batch size, precision,
quantization, and gradient checkpointing all come from `configs/hardware/*`
(cpu_smoketest / single_gpu_t4 / multi_gpu), and training runs through a
plain HF `Trainer`, which is already accelerate-integrated - multi-GPU /
multi-node is `accelerate launch --config_file configs/accelerate/multi_gpu.yaml
-m slm.train_sft` with no code changes.

Usage:
    python -m slm.train_sft hardware=cpu_smoketest data.max_train_samples=32
    accelerate launch --config_file configs/accelerate/single_gpu.yaml \
        -m slm.train_sft hardware=single_gpu_t4
"""
import functools
import os

import hydra
import torch
from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model
from torch.utils.data import Subset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from slm.dataset import PlannerSFTDataset, collate


@hydra.main(config_path="../configs", config_name="train/slm_sft", version_base=None)
def main(cfg: DictConfig):
    hw = cfg.hardware
    if hw.quantization == "4bit" and not torch.cuda.is_available():
        raise RuntimeError(
            "hardware.quantization=4bit requires CUDA (bitsandbytes). "
            "Use hardware=cpu_smoketest for a local CPU run."
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"torch_dtype": torch.float32}
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
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    if hw.gradient_checkpointing:
        # use_reentrant=False: avoids DDP's "Expected to mark a variable
        # ready only once" under multi_gpu (reentrant checkpointing re-runs
        # forward during backward, double-firing DDP's gradient-ready hooks
        # for reused parameters) - see vlm/train_sft.py's comment.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # Standard PEFT + gradient-checkpointing fix - without this, the
        # frozen base model can leave the checkpointed graph with no tensor
        # requiring grad, and the loss comes back with no grad_fn
        # ("element 0 of tensors does not require grad...").
        model.enable_input_require_grads()

    metadata_path = hydra.utils.to_absolute_path(cfg.data.metadata_path)
    dataset = PlannerSFTDataset(metadata_path, tokenizer, max_samples=cfg.data.max_train_samples)
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
        dataloader_num_workers=hw.dataloader_num_workers,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        use_cpu=(hw.device == "cpu"),
        # Every LoRA parameter participates in every forward pass, so DDP's
        # unused-parameter scan is pure overhead. Trainer would default this
        # to True because a PeftModel isn't recognized as a PreTrainedModel.
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=functools.partial(collate, tokenizer=tokenizer),
    )
    result = trainer.train()
    print(f"Final train loss: {result.training_loss:.4f}")

    adapter_dir = hydra.utils.to_absolute_path(cfg.save_adapter_dir)
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Saved LoRA adapter -> {adapter_dir}")


if __name__ == "__main__":
    main()

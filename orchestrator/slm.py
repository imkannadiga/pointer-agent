"""Shared small-language-model wrapper used by both the planner and verifier.

Loaded once and passed to both, so the same weights back the "SLM as parser"
and "SLM as verifier" roles rather than loading two copies.
"""
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


class QwenSLM:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cpu",
        adapter_path: str | None = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Qwen2.5's config.json already sets use_sliding_window=false (sliding
        # window attention is genuinely off), but this transformers version's
        # sdpa path warns off the raw `sliding_window` window-size value
        # regardless of that flag, and does so at layer-construction time
        # inside from_pretrained - so the config must be patched before the
        # model is built, not after. Clearing it here doesn't change behavior
        # since sliding window was never active.
        config = AutoConfig.from_pretrained(model_name)
        config.sliding_window = None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, config=config, torch_dtype=torch.float32
        )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.to(device)
        self.model.eval()
        self.device = device

    def chat(self, system: str, user: str, max_new_tokens: int = 128) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

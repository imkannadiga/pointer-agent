"""Shared small-language-model wrapper used by both the planner and verifier.

Loaded once and passed to both, so the same weights back the "SLM as parser"
and "SLM as verifier" roles rather than loading two copies.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenSLM:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", device: str = "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
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

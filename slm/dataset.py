"""Converts synthetic metadata rows into instruction -> structured-query SFT
pairs for the planner (Phase 2), reusing the exact system prompt and chat
template the prompted-only QwenPlanner already uses at inference time so the
SFT target distribution matches what's actually queried in the pipeline.
"""
import json

import torch
from torch.utils.data import Dataset

from orchestrator.planner_qwen import SYSTEM_PROMPT


class PlannerSFTDataset(Dataset):
    def __init__(self, metadata_path: str, tokenizer, max_samples: int | None = None, max_length: int = 512):
        with open(metadata_path) as f:
            rows = [json.loads(line) for line in f]
        if max_samples:
            rows = rows[:max_samples]
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        target = {
            "target_phrase": row["target_phrase"],
            "answer_type": row["answer_type"],
            "referring_expression": row.get("referring_expression"),
        }
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Instruction: {row['instruction']}"},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        target_text = json.dumps(target) + self.tokenizer.eos_token

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full = self.tokenizer(
            prompt + target_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = full["input_ids"]
        labels = list(input_ids)
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = -100
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": full["attention_mask"],
        }


def collate(batch: list, tokenizer) -> dict:
    max_len = max(len(b["input_ids"]) for b in batch)
    pad_id = tokenizer.pad_token_id
    input_ids, labels, attention_mask = [], [], []
    for b in batch:
        pad_len = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad_len)
        labels.append(b["labels"] + [-100] * pad_len)
        attention_mask.append(b["attention_mask"] + [0] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention_mask),
    }

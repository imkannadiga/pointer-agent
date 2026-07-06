"""CLI entrypoint: generate N synthetic pointer-grounding samples.

Usage (from repo root, with deps installed and `playwright install chromium` run once):

    python -m data_gen.generate_dataset
    python -m data_gen.generate_dataset n_samples=500 output_dir=output/synthetic_v1

Output layout matches the real pointerbench-text dataset:
    <output_dir>/images/0000.png, 0001.png, ...
    <output_dir>/metadata.jsonl
"""
import json
import os
import random

import hydra
from omegaconf import DictConfig

from data_gen.content_fillers import generate_content
from data_gen.instruction_templates import phrase
from data_gen.render import Renderer, render_html
from data_gen.task_builder import build_task
from data_gen.push_to_hub import push_to_hub, save_local_hf_dataset

IMAGE_SIZE = [1024, 768]


def generate_row(index: int, cfg: DictConfig, renderer: Renderer, images_dir: str):
    rng = random.Random(cfg.seed + index)
    surface = rng.choice(list(cfg.surfaces))
    language = rng.choice(list(cfg.languages))
    category = rng.choice(list(cfg.categories))
    theme = rng.choice(list(cfg.themes))
    font_size = rng.choice(list(cfg.font_sizes))

    content = generate_content(surface, language, rng)
    html = render_html(surface, content, theme, font_size)

    file_name = f"{index:04d}.png"
    screenshot_path = os.path.join(images_dir, file_name)
    boxes = renderer.render(html, screenshot_path)

    task = build_task(category, boxes, rng, language, phrase)
    if task is None:
        os.remove(screenshot_path)
        return None

    return {
        "file_name": file_name,
        "id": f"synth_{index:04d}",
        "instruction": task["instruction"],
        "bbox": task["bbox"],
        "point": task["point"],
        "answer_type": task["answer_type"],
        "eval": task["eval"],
        "data_type": task["data_type"],
        "category": task["category"],
        "surface": surface,
        "language": language,
        "difficulty": task["difficulty"],
        "image_size": IMAGE_SIZE,
    }


@hydra.main(config_path="../configs/data_gen", config_name="config", version_base=None)
def main(cfg: DictConfig):
    output_dir = hydra.utils.to_absolute_path(cfg.output_dir)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    rows = []
    with Renderer() as renderer:
        for i in range(cfg.n_samples):
            row = generate_row(i, cfg, renderer, images_dir)
            if row is not None:
                rows.append(row)

    metadata_path = os.path.join(output_dir, "metadata.jsonl")
    with open(metadata_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Generated {len(rows)}/{cfg.n_samples} samples -> {output_dir}")

    if cfg.push_to_hub:
        if not cfg.hub_repo_id:
            raise ValueError("push_to_hub=true requires hub_repo_id to be set")
        push_to_hub(output_dir, cfg.hub_repo_id, token=cfg.hub_token)
    else:
        hf_dir = save_local_hf_dataset(output_dir)
        print(f"Saved local HF dataset (dry run, no push) -> {hf_dir}")


if __name__ == "__main__":
    main()

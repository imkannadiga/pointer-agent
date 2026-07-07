"""CLI entrypoint: generate N synthetic pointer-grounding samples.

Usage (from repo root, with deps installed and `playwright install chromium` run once):

    python -m data_gen.generate_dataset
    python -m data_gen.generate_dataset n_samples=500 output_dir=output/synthetic_v1

Output layout matches the real pointerbench-text dataset:
    <output_dir>/images/0000.png, 0001.png, ...
    <output_dir>/metadata.jsonl

Category sampling is uniform round-robin (not independent-random per row):
`n_samples` is divided as evenly as possible across `cfg.categories`, so
every requested category gets (approximately) the same number of rows
regardless of how surface/content availability skews naive random sampling.
A category unavailable for a chosen surface (see
`task_builder.CATEGORY_SURFACES`) simply retries with a different surface,
up to `cfg.max_retries_per_row`.
"""
import json
import os
import random

import hydra
from omegaconf import DictConfig

from data_gen.content_fillers import generate_content
from data_gen.instruction_templates import phrase
from data_gen.render import Renderer, render_html
from data_gen.task_builder import build_task, CATEGORY_SURFACES
from data_gen.push_to_hub import push_to_hub, save_local_hf_dataset

IMAGE_SIZE = [1024, 768]


def _pick_surface(category: str, cfg_surfaces: list, rng: random.Random) -> str | None:
    allowed = [s for s in cfg_surfaces if s in CATEGORY_SURFACES.get(category, cfg_surfaces)]
    return rng.choice(allowed) if allowed else None


def _maybe_occlusion_box(rng: random.Random, enable_occlusion: bool):
    if not enable_occlusion or rng.random() > 0.3:
        return None
    w, h = rng.randint(120, 320), rng.randint(60, 160)
    x = rng.randint(0, max(1024 - w, 0))
    y = rng.randint(0, max(768 - h, 0))
    return {"x": x, "y": y, "w": w, "h": h}


def generate_row(index: int, category: str, cfg: DictConfig, renderer: Renderer, images_dir: str):
    rng = random.Random(cfg.seed + index)
    surface = _pick_surface(category, list(cfg.surfaces), rng)
    if surface is None:
        return None
    language = rng.choice(list(cfg.languages))
    theme = rng.choice(list(cfg.themes))
    font_size = rng.choice(list(cfg.font_sizes))
    occlusion_box = _maybe_occlusion_box(rng, cfg.enable_occlusion)

    content = generate_content(surface, language, rng, enable_distractors=cfg.enable_distractors)
    html = render_html(surface, content, theme, font_size, occlusion_box=occlusion_box)

    file_name = f"{index:04d}.png"
    screenshot_path = os.path.join(images_dir, file_name)
    boxes = renderer.render(html, screenshot_path)

    task = build_task(category, boxes, rng, language, phrase)
    if task is None:
        if os.path.exists(screenshot_path):
            os.remove(screenshot_path)
        return None

    row = {
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
        "target_phrase": task["target_phrase"],
    }
    if task.get("referring_expression"):
        row["referring_expression"] = task["referring_expression"]
    return row


def _category_plan(n_samples: int, categories: list, seed: int) -> list:
    """Round-robin category assignment so every category gets an
    (approximately) equal share of n_samples, shuffled for order variety."""
    rng = random.Random(seed)
    cats = list(categories)
    rng.shuffle(cats)
    plan = [cats[i % len(cats)] for i in range(n_samples)]
    rng.shuffle(plan)
    return plan


@hydra.main(config_path="../configs/data_gen", config_name="config", version_base=None)
def main(cfg: DictConfig):
    output_dir = hydra.utils.to_absolute_path(cfg.output_dir)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    plan = _category_plan(cfg.n_samples, list(cfg.categories), cfg.seed)

    rows = []
    with Renderer() as renderer:
        index = 0
        for category in plan:
            row = None
            for attempt in range(cfg.max_retries_per_row):
                row = generate_row(index, category, cfg, renderer, images_dir)
                if row is not None:
                    break
                index += 1  # burn the index so retry gets a fresh rng draw
            if row is not None:
                rows.append(row)
            index += 1

    # Re-sequence file_name/id to be contiguous (retries can leave gaps).
    final_rows = []
    for new_idx, row in enumerate(rows):
        old_name = row["file_name"]
        new_name = f"{new_idx:04d}.png"
        if old_name != new_name:
            os.replace(os.path.join(images_dir, old_name), os.path.join(images_dir, new_name))
        row["file_name"] = new_name
        row["id"] = f"synth_{new_idx:04d}"
        final_rows.append(row)

    metadata_path = os.path.join(output_dir, "metadata.jsonl")
    with open(metadata_path, "w") as f:
        for row in final_rows:
            f.write(json.dumps(row) + "\n")

    print(f"Generated {len(final_rows)}/{cfg.n_samples} samples -> {output_dir}")

    if cfg.push_to_hub:
        if not cfg.hub_repo_id:
            raise ValueError("push_to_hub=true requires hub_repo_id to be set")
        push_to_hub(output_dir, cfg.hub_repo_id, token=cfg.hub_token)
    else:
        hf_dir = save_local_hf_dataset(output_dir)
        print(f"Saved local HF dataset (dry run, no push) -> {hf_dir}")


if __name__ == "__main__":
    main()

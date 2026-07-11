"""CLI entrypoint: generate N synthetic pointer-grounding samples.

Usage (from repo root, with deps installed and `playwright install chromium` run once):

    python -m data_gen.generate_dataset
    python -m data_gen.generate_dataset n_samples=500 output_dir=output/synthetic_v1
    python -m data_gen.generate_dataset n_samples=500 num_workers=8

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

Parallelized across CPU cores (not GPUs - this pipeline is pure Playwright
rendering + Faker content generation + geometry, no model inference, so
there's no GPU work to schedule). One worker process per core, each with its
own Playwright browser instance created once and reused across every row
that worker handles - launching a browser per row would dominate runtime.
`num_workers` (config, default: all cores) controls the pool size; set it
lower on memory-constrained machines, since each worker's Chromium instance
has its own footprint.
"""
import json
import logging
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from data_gen.content_fillers import generate_content
from data_gen.instruction_templates import phrase
from data_gen.render import Renderer, list_variants, render_html
from data_gen.task_builder import build_task, CATEGORY_SURFACES, languages_for_category
from data_gen.push_to_hub import push_to_hub, save_local_hf_dataset

IMAGE_SIZE = [1024, 768]

logger = logging.getLogger(__name__)

# Set once per worker process by _init_worker - a fresh Playwright browser
# per process, reused across every task that process handles.
_worker_renderer: Renderer | None = None


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
    language = rng.choice(languages_for_category(category, list(cfg.languages)))
    theme = rng.choice(list(cfg.themes))
    font_size = rng.choice(list(cfg.font_sizes))
    occlusion_box = _maybe_occlusion_box(rng, cfg.enable_occlusion)

    variant = rng.choice(list_variants(surface))
    content = generate_content(surface, language, rng, enable_distractors=cfg.enable_distractors)
    html = render_html(surface, content, theme, font_size, occlusion_box=occlusion_box, variant=variant)

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
        "anchor_phrase": task["anchor_phrase"],
        "anchor_bbox": task["anchor_bbox"],
        "relation": task["relation"],
        "relation_params": task["relation_params"],
        "template_variant": variant,
    }
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


def _init_worker(images_dir: str):
    """Pool initializer: runs once per worker process. Creates one
    Playwright browser for this worker's entire lifetime rather than one per
    row - browser launch (~1-2s) would otherwise dominate runtime."""
    global _worker_renderer
    logging.basicConfig(level=logging.WARNING)
    _worker_renderer = Renderer()
    _worker_renderer.__enter__()
    os.makedirs(images_dir, exist_ok=True)


def _generate_slot(args: tuple) -> dict | None:
    """One (slot_idx, category) plan entry -> a completed row, or None if
    every retry failed. Each slot owns a reserved, collision-free block of
    `max_retries_per_row` scratch indices (slot_idx * max_retries + attempt)
    for rng-seeding and the intermediate image filename, so slots can run in
    any order across any number of workers with no shared counter needed -
    the existing post-generation re-sequencing pass compacts whatever
    scratch indices succeeded into a contiguous final range."""
    slot_idx, category, cfg_container, images_dir, max_retries = args
    cfg = OmegaConf.create(cfg_container)
    global _worker_renderer
    for attempt in range(max_retries):
        scratch_index = slot_idx * max_retries + attempt
        try:
            row = generate_row(scratch_index, category, cfg, _worker_renderer, images_dir)
        except Exception:
            logger.exception("generate_row failed for slot=%d category=%s attempt=%d", slot_idx, category, attempt)
            row = None
        if row is not None:
            return row
    return None


@hydra.main(config_path="../configs/data_gen", config_name="config", version_base=None)
def main(cfg: DictConfig):
    output_dir = hydra.utils.to_absolute_path(cfg.output_dir)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    plan = _category_plan(cfg.n_samples, list(cfg.categories), cfg.seed)
    num_workers = cfg.num_workers or os.cpu_count() or 1
    cfg_container = OmegaConf.to_container(cfg, resolve=True)
    tasks = [
        (slot_idx, category, cfg_container, images_dir, cfg.max_retries_per_row)
        for slot_idx, category in enumerate(plan)
    ]

    rows = []
    # Playwright browsers must be created after forking, never shared across
    # it - _init_worker (not this parent process) is what launches them, one
    # per worker, so the default 'fork' start method on Linux is safe here.
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_worker,
        initargs=(images_dir,),
    ) as pool:
        futures = [pool.submit(_generate_slot, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating", unit="row"):
            row = future.result()
            if row is not None:
                rows.append(row)

    # Re-sequence file_name/id to be contiguous (retries/failed slots leave
    # gaps, and completion order is unrelated to slot order under
    # parallelism) - sort by original scratch index first (as an int, not
    # the zero-padded string, which sorts incorrectly once indices exceed
    # 4 digits) so output is reproducibly ordered run-to-run regardless of
    # worker scheduling.
    rows.sort(key=lambda r: int(r["id"].rsplit("_", 1)[1]))
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

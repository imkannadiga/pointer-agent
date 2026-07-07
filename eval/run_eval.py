"""Phase 0 baseline: prompted SLM planner -> off-the-shelf VLM grounder ->
prompted SLM verifier, scored with the real pointerbench-text scorer against
either the real benchmark or the Sprint 1 synthetic data.

Usage:
    python -m eval.run_eval source=synthetic limit=30
    python -m eval.run_eval source=real limit=30
"""
import json
import os
import random

import hydra
from omegaconf import DictConfig
from PIL import Image, ImageDraw

from eval.fetch_pointerbench import fetch_images, fetch_metadata
from eval.scorer import evaluate
from orchestrator.grounder_florence2 import Florence2Grounder
from orchestrator.pipeline import Pipeline
from orchestrator.planner_qwen import QwenPlanner
from orchestrator.slm import QwenSLM
from orchestrator.verifier_qwen import QwenVerifier


def _load_gt_rows(cfg: DictConfig) -> tuple[list[dict], dict[str, str]]:
    """Returns (gt_rows, id -> local image path), subsampled to cfg.limit."""
    if cfg.source == "real":
        rows = fetch_metadata()
    elif cfg.source == "synthetic":
        metadata_path = hydra.utils.to_absolute_path(cfg.synthetic_metadata)
        with open(metadata_path) as f:
            rows = [json.loads(line) for line in f]
    else:
        raise ValueError(f"Unknown source: {cfg.source}")

    if cfg.limit and cfg.limit < len(rows):
        rng = random.Random(cfg.seed)
        rows = rng.sample(rows, cfg.limit)

    if cfg.source == "real":
        image_paths = fetch_images(rows)
    else:
        images_dir = hydra.utils.to_absolute_path(cfg.synthetic_images_dir)
        image_paths = {row["id"]: os.path.join(images_dir, row["file_name"]) for row in rows}

    return rows, image_paths


def _save_viz(image: Image.Image, row: dict, prediction: dict, viz_dir: str):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    draw.rectangle(row["bbox"], outline="red", width=2)
    draw.rectangle(prediction["bbox"], outline="blue", width=2)
    px, py = prediction["point"]
    r = 4
    draw.ellipse([px - r, py - r, px + r, py + r], fill="cyan", outline="black")
    img.save(os.path.join(viz_dir, f"{row['id']}.png"))


@hydra.main(config_path="../configs", config_name="eval/phase0", version_base=None)
def main(cfg: DictConfig):
    output_dir = hydra.utils.to_absolute_path(cfg.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    viz_dir = os.path.join(output_dir, "viz")
    if cfg.save_viz:
        os.makedirs(viz_dir, exist_ok=True)

    print(f"Loading planner/verifier SLM: {cfg.model.planner.name}")
    slm = QwenSLM(
        cfg.model.planner.name,
        device=cfg.model.planner.device,
        adapter_path=cfg.model.planner.get("adapter_path"),
    )
    planner = QwenPlanner(slm)
    if cfg.model.verifier.name == cfg.model.planner.name and not cfg.model.planner.get("adapter_path"):
        verifier_slm = slm
    else:
        verifier_slm = QwenSLM(cfg.model.verifier.name, device=cfg.model.verifier.device)
    verifier = QwenVerifier(verifier_slm)

    print(f"Loading grounder VLM: {cfg.model.grounder.name}")
    grounder = Florence2Grounder(
        cfg.model.grounder.name,
        device=cfg.model.grounder.device,
        adapter_path=cfg.model.grounder.get("adapter_path"),
    )

    pipeline = Pipeline(planner, grounder, verifier)

    print(f"Loading ground truth (source={cfg.source}, limit={cfg.limit})")
    gt_rows, image_paths = _load_gt_rows(cfg)

    predictions = {}
    for i, row in enumerate(gt_rows):
        image = Image.open(image_paths[row["id"]]).convert("RGB")
        prediction = pipeline.run(image, row["instruction"])
        predictions[row["id"]] = {"id": row["id"], **prediction}
        if cfg.save_viz:
            _save_viz(image, row, prediction, viz_dir)
        print(f"[{i + 1}/{len(gt_rows)}] {row['id']} ({row['category']}): {row['instruction']!r}")

    predictions_path = os.path.join(output_dir, "predictions.jsonl")
    with open(predictions_path, "w") as f:
        for pred in predictions.values():
            f.write(json.dumps(pred) + "\n")
    print(f"Predictions -> {predictions_path}")

    report = evaluate(gt_rows, predictions)
    report_path = os.path.join(output_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nPhase 0 ({cfg.source}, n={report['n']}): accuracy={report['accuracy'] * 100:.2f}%")
    for axis in ("answer_type", "data_type", "category", "surface", "language", "difficulty"):
        print(f"\nBy {axis}:")
        for key, v in report[f"by_{axis}"].items():
            print(f"  {key:20s} {v['acc'] * 100:6.2f}%  (n={v['n']})")
    print(f"\nFull report -> {report_path}")


if __name__ == "__main__":
    main()

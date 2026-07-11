"""Package the generated dataset as a HF `datasets.Dataset` and either save it
locally (default, no credentials required) or push it to the Hub.

Decoupled from generation on purpose: generation is a one-time/periodic job,
training can pull from the Hub from any machine without re-running it.
"""
import json
import os

from datasets import Dataset, Features, Image, Sequence, Value


def _load_rows(output_dir: str) -> list[dict]:
    metadata_path = os.path.join(output_dir, "metadata.jsonl")
    with open(metadata_path) as f:
        rows = [json.loads(line) for line in f]
    for row in rows:
        row["image"] = os.path.join(output_dir, "images", row["file_name"])
    return rows


def _features() -> Features:
    return Features(
        {
            "file_name": Value("string"),
            "id": Value("string"),
            "instruction": Value("string"),
            "bbox": Sequence(Value("int64")),
            "point": Sequence(Value("int64")),
            "answer_type": Value("string"),
            "eval": {
                "type": Value("string"),
                "bbox": Sequence(Value("int64")),
                "min_coverage": Value("float64"),
                "min_precision": Value("float64"),
            },
            "data_type": Value("string"),
            "category": Value("string"),
            "surface": Value("string"),
            "language": Value("string"),
            "difficulty": Value("string"),
            "image_size": Sequence(Value("int64")),
            "image": Image(),
            "anchor_phrase": Value("string"),
            "anchor_bbox": Sequence(Value("int64")),
            "relation": Value("string"),
            "relation_params": {
                "target_char": Value("string"),
                "occurrence": Value("int64"),
                "char1": Value("string"),
                "char2": Value("string"),
                "index": Value("int64"),
                "second_anchor_phrase": Value("string"),
            },
            "template_variant": Value("string"),
        }
    )


_RELATION_PARAM_KEYS = (
    "target_char", "occurrence", "char1", "char2", "index", "second_anchor_phrase",
)


def _build_dataset(output_dir: str) -> Dataset:
    rows = _load_rows(output_dir)
    for row in rows:
        # bbox_overlap eval rows don't have min_coverage/min_precision on point
        # tasks' point_in_bbox eval - normalize so the shared Features schema fits.
        row["eval"].setdefault("min_coverage", None)
        row["eval"].setdefault("min_precision", None)
        # relation_params only ever has a subset of keys per category; normalize
        # for the shared Features schema.
        params = row.get("relation_params") or {}
        row["relation_params"] = {key: params.get(key) for key in _RELATION_PARAM_KEYS}
    return Dataset.from_list(rows, features=_features())


def save_local_hf_dataset(output_dir: str) -> str:
    ds = _build_dataset(output_dir)
    hf_dir = os.path.join(output_dir, "hf_dataset")
    ds.save_to_disk(hf_dir)
    return hf_dir


def push_to_hub(output_dir: str, repo_id: str, token: str | None = None) -> None:
    ds = _build_dataset(output_dir)
    ds.push_to_hub(repo_id, token=token)

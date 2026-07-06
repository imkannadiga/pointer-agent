"""Pull the real pointerbench-text benchmark from HF Hub on demand.

Downloads only metadata.jsonl up front, then only the specific images actually
needed (e.g. a sampled subset), rather than the full ~500-image set every time.
"""
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "WarmwindOS/pointerbench"
SUBSET = "pointerbench-text"


def fetch_metadata() -> list[dict]:
    import json

    path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=f"{SUBSET}/data/test/metadata.jsonl",
    )
    with open(path) as f:
        return [json.loads(line) for line in f]


def fetch_eval_script() -> Path:
    """Download the real eval.py for eval/scorer.py to import directly."""
    path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=f"{SUBSET}/eval.py",
    )
    return Path(path)


def fetch_images(rows: list[dict]) -> dict[str, str]:
    """Download only the PNGs needed for the given rows. Returns id -> local path."""
    paths = {}
    for row in rows:
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=f"{SUBSET}/data/test/{row['file_name']}",
        )
        paths[row["id"]] = local_path
    return paths

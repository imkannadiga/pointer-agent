"""Internal consistency checks on generated metadata.jsonl.

Optionally diffs field names against a real pointerbench-text metadata.jsonl
if a path to one is provided, to confirm schema parity.

Usage:
    python scripts/validate_schema.py <output_dir> [--real path/to/real/metadata.jsonl]
"""
import argparse
import json
import os
import sys

REQUIRED_KEYS = {
    "file_name", "id", "instruction", "bbox", "point", "answer_type",
    "eval", "data_type", "category", "surface", "language", "difficulty",
    "image_size",
}


def check_row(row: dict, idx: int, errors: list):
    missing = REQUIRED_KEYS - row.keys()
    if missing:
        errors.append(f"row {idx}: missing keys {missing}")
        return

    if row["answer_type"] not in ("point", "bbox"):
        errors.append(f"row {idx}: bad answer_type {row['answer_type']!r}")

    bbox = row["bbox"]
    if not (isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, int) for v in bbox)):
        errors.append(f"row {idx}: bad bbox {bbox!r}")
    elif not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
        errors.append(f"row {idx}: degenerate bbox {bbox!r}")

    point = row["point"]
    if not (isinstance(point, list) and len(point) == 2 and all(isinstance(v, int) for v in point)):
        errors.append(f"row {idx}: bad point {point!r}")
    elif isinstance(bbox, list) and len(bbox) == 4:
        x, y = point
        if not (bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]):
            errors.append(f"row {idx}: point {point} not inside bbox {bbox}")

    if row["image_size"] != [1024, 768]:
        errors.append(f"row {idx}: unexpected image_size {row['image_size']!r}")

    img_path = os.path.join(args.output_dir, "images", row["file_name"])
    if not os.path.exists(img_path):
        errors.append(f"row {idx}: missing image file {img_path}")

    ev = row["eval"]
    if ev.get("type") not in ("point_in_bbox", "bbox_overlap"):
        errors.append(f"row {idx}: bad eval.type {ev.get('type')!r}")


def main():
    global args
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir")
    ap.add_argument("--real", default=None, help="path to a real pointerbench-text metadata.jsonl")
    args = ap.parse_args()

    metadata_path = os.path.join(args.output_dir, "metadata.jsonl")
    rows = [json.loads(l) for l in open(metadata_path)]

    errors = []
    for i, row in enumerate(rows):
        check_row(row, i, errors)

    print(f"Checked {len(rows)} rows.")
    if errors:
        print(f"{len(errors)} ERRORS:")
        for e in errors[:50]:
            print(" -", e)
    else:
        print("All internal consistency checks passed.")

    print("\nField coverage:")
    for field in ("category", "data_type", "answer_type", "surface", "language", "difficulty"):
        values = sorted({row[field] for row in rows})
        print(f"  {field}: {values}")

    if args.real:
        real_rows = [json.loads(l) for l in open(args.real)]
        real_keys = set(real_rows[0].keys())
        gen_keys = set(rows[0].keys())
        print("\nSchema diff vs real pointerbench-text:")
        print("  missing here:", real_keys - gen_keys)
        print("  extra here:  ", gen_keys - real_keys)

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Draw each row's bbox/point back onto its screenshot for a visual sanity check.

Usage: python scripts/visualize_samples.py <output_dir> [--limit N]
"""
import argparse
import json
import os

from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir")
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()

    metadata_path = os.path.join(args.output_dir, "metadata.jsonl")
    rows = [json.loads(l) for l in open(metadata_path)][: args.limit]

    viz_dir = os.path.join(args.output_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    for row in rows:
        img_path = os.path.join(args.output_dir, "images", row["file_name"])
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        bbox = row["bbox"]
        draw.rectangle(bbox, outline="red", width=2)
        px, py = row["point"]
        r = 4
        draw.ellipse([px - r, py - r, px + r, py + r], fill="lime", outline="black")
        out_path = os.path.join(viz_dir, row["file_name"])
        img.save(out_path)
        print(f"{row['file_name']}: [{row['category']}] {row['instruction']!r} -> {out_path}")


if __name__ == "__main__":
    main()

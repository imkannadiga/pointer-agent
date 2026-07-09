"""Per-category diagnosis of 0%-accuracy rows from a Phase 0 eval run.

Computes (dx, dy) = pred_point - gt_point per row (center-to-center for
bbox-answer_type categories, since the raw "point" field isn't guaranteed to
be the bbox center for every resolver path), then buckets each category into
exactly one of:

  fallback         - char_locator's OCR sanity check declined to trust the
                      OCR read and returned the anchor's own grounded box
                      unchanged (only possible for non-"self" relations,
                      where the resolved bbox is expected to differ from
                      anchor_bbox). Not a resolver miss - the resolver never
                      ran its real logic.
  constant-offset  - |mean dx| or |mean dy| is large relative to its std:
                      every prediction is wrong by roughly the same amount in
                      the same direction. On caret_before_word/
                      caret_after_word this points at the caret-margin
                      constant disagreeing between task_builder.py and
                      char_locator.py; on char_center/char_bbox/
                      caret_between_chars a systematic dy points at a
                      residual image_to_boxes origin/transform bug.
  scatter          - no fallback, no systematic offset: errors are spread
                      out, consistent with OCR noise (a data/params problem,
                      not a code bug).

Usage:
    python scripts/diagnose_offsets.py predictions.jsonl metadata.jsonl
    python scripts/diagnose_offsets.py predictions.jsonl metadata.jsonl --fallback-threshold 0.3
"""
import argparse
import json
import statistics as stats
from collections import defaultdict


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def bbox_center(bbox: list) -> tuple:
    x0, y0, x1, y1 = bbox
    return (x0 + x1) / 2, (y0 + y1) / 2


def point_for(d: dict, answer_type: str) -> tuple:
    """Center-to-center for bbox categories; the raw point field otherwise."""
    if answer_type == "bbox" and d.get("bbox"):
        return bbox_center(d["bbox"])
    return tuple(d["point"])


def is_hit(row: dict, pred: dict) -> bool:
    """Mirrors the real pointerbench-text scoring rules (point-in-bbox
    inclusive of edges; asymmetric coverage/precision bbox overlap) closely
    enough for this diagnostic - the authoritative scorer is eval/scorer.py."""
    ev = row["eval"]
    gx0, gy0, gx1, gy1 = ev["bbox"]
    if ev["type"] == "point_in_bbox":
        px, py = pred["point"]
        return gx0 <= px <= gx1 and gy0 <= py <= gy1
    px0, py0, px1, py1 = pred["bbox"]
    ix0, iy0 = max(gx0, px0), max(gy0, py0)
    ix1, iy1 = min(gx1, px1), min(gy1, py1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    gt_area = max(1.0, (gx1 - gx0) * (gy1 - gy0))
    pred_area = max(1.0, (px1 - px0) * (py1 - py0))
    coverage = inter / gt_area
    precision = inter / pred_area
    return coverage >= ev["min_coverage"] and precision >= ev["min_precision"]


def is_fallback(row: dict, pred: dict) -> bool:
    if row.get("relation", "self") == "self":
        return False
    anchor = row.get("anchor_bbox")
    return anchor is not None and list(pred.get("bbox", [])) == list(anchor)


def classify(dxs, dys, fallback_frac, fallback_threshold, offset_abs_px, offset_ratio) -> str:
    if fallback_frac >= fallback_threshold:
        return "fallback"
    if not dxs:
        return "scatter"
    mean_dx, mean_dy = stats.mean(dxs), stats.mean(dys)
    std_dx = stats.pstdev(dxs) if len(dxs) > 1 else 0.0
    std_dy = stats.pstdev(dys) if len(dys) > 1 else 0.0
    dx_offset = abs(mean_dx) >= offset_abs_px and abs(mean_dx) >= offset_ratio * (std_dx + 1e-6)
    dy_offset = abs(mean_dy) >= offset_abs_px and abs(mean_dy) >= offset_ratio * (std_dy + 1e-6)
    if dx_offset and dy_offset:
        return "constant-offset (dx+dy)"
    if dx_offset:
        return "constant-offset (dx)"
    if dy_offset:
        return "constant-offset (dy)"
    return "scatter"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", help="path to predictions.jsonl written by eval/run_eval.py")
    ap.add_argument("metadata", help="path to the matching metadata.jsonl (ground truth)")
    ap.add_argument("--fallback-threshold", type=float, default=0.5,
                     help="fraction of a category's rows that must hit the char_locator fallback to label it 'fallback' (default 0.5)")
    ap.add_argument("--offset-abs-px", type=float, default=3.0,
                     help="minimum |mean dx or dy| (px) to even consider 'constant-offset' (default 3.0)")
    ap.add_argument("--offset-ratio", type=float, default=2.5,
                     help="minimum |mean| / std ratio to call an axis a constant offset rather than scatter (default 2.5)")
    args = ap.parse_args()

    preds = {r["id"]: r for r in load_jsonl(args.predictions)}
    rows = load_jsonl(args.metadata)

    by_category = defaultdict(lambda: {"dxs": [], "dys": [], "n": 0, "hits": 0, "fallbacks": 0, "missing": 0})

    for row in rows:
        pred = preds.get(row["id"])
        bucket = by_category[row["category"]]
        if pred is None:
            bucket["missing"] += 1
            continue
        bucket["n"] += 1
        bucket["hits"] += int(is_hit(row, pred))
        bucket["fallbacks"] += int(is_fallback(row, pred))
        gx, gy = point_for(row, row["answer_type"])
        px, py = point_for(pred, row["answer_type"])
        bucket["dxs"].append(px - gx)
        bucket["dys"].append(py - gy)

    header = f"{'category':24s} {'n':>4s} {'acc%':>6s} {'fallback%':>10s} {'mean_dx':>8s} {'std_dx':>7s} {'mean_dy':>8s} {'std_dy':>7s}  label"
    print(header)
    print("-" * len(header))
    for cat in sorted(by_category):
        b = by_category[cat]
        n = b["n"]
        if n == 0:
            print(f"{cat:24s} {0:4d}      -          -        -       -        -       -  no predictions found")
            continue
        acc = b["hits"] / n * 100
        fallback_frac = b["fallbacks"] / n
        mean_dx, mean_dy = stats.mean(b["dxs"]), stats.mean(b["dys"])
        std_dx = stats.pstdev(b["dxs"]) if n > 1 else 0.0
        std_dy = stats.pstdev(b["dys"]) if n > 1 else 0.0
        label = ""
        if acc == 0:
            label = classify(
                b["dxs"], b["dys"], fallback_frac,
                args.fallback_threshold, args.offset_abs_px, args.offset_ratio,
            )
        print(
            f"{cat:24s} {n:4d} {acc:6.1f} {fallback_frac * 100:9.1f}% "
            f"{mean_dx:8.2f} {std_dx:7.2f} {mean_dy:8.2f} {std_dy:7.2f}  {label}"
        )

    unlabeled = [
        cat for cat, b in by_category.items()
        if b["n"] and b["hits"] == 0
        and classify(b["dxs"], b["dys"], b["fallbacks"] / b["n"],
                      args.fallback_threshold, args.offset_abs_px, args.offset_ratio) == "scatter"
        and stats.pstdev(b["dxs"]) < args.offset_abs_px and stats.pstdev(b["dys"]) < args.offset_abs_px
        and abs(stats.mean(b["dxs"])) < args.offset_abs_px and abs(stats.mean(b["dys"])) < args.offset_abs_px
    ]
    if unlabeled:
        print(f"\nNote: {unlabeled} labeled 'scatter' with near-zero mean AND near-zero std "
              f"- check these by hand, that combination usually means too few 0%-accuracy "
              f"rows in the category to say anything statistically.")


if __name__ == "__main__":
    main()

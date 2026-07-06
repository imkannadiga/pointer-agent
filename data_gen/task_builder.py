"""Boxes (from render.py) -> (instruction, target) task rows.

Implements the two task families in Sprint 1 scope:
  - direct-reference: word_center (point), word_bbox (bbox)
  - positional/line-based: line_start (point), line_end (point)

data_type assignment mirrors the real pointerbench-text category->data_type
mapping confirmed by inspection: word_center -> "word", word_bbox -> "bbox",
line_start/line_end -> "caret" (caret/cursor-position semantics, not a
bbox-shaped answer).
"""

LINE_CLUSTER_TOL_PX = 6


def _round_box(b):
    return [round(b["x0"]), round(b["y0"]), round(b["x1"]), round(b["y1"])]


def _bbox_center(bbox):
    return [round((bbox[0] + bbox[2]) / 2), round((bbox[1] + bbox[3]) / 2)]


def cluster_lines(boxes, tol=LINE_CLUSTER_TOL_PX):
    """Group boxes into visual lines by clustering on vertical center."""
    boxes_sorted = sorted(boxes, key=lambda b: (b["y0"] + b["y1"]) / 2)
    lines = []
    for b in boxes_sorted:
        cy = (b["y0"] + b["y1"]) / 2
        placed = False
        for line in lines:
            line_cy = sum((x["y0"] + x["y1"]) / 2 for x in line) / len(line)
            if abs(cy - line_cy) <= tol:
                line.append(b)
                placed = True
                break
        if not placed:
            lines.append([b])
    for line in lines:
        line.sort(key=lambda b: b["x0"])
    return lines


def compute_difficulty(target_text, boxes):
    """Heuristic difficulty: more on-page repeats of the target text = harder
    (acts as a naturally-occurring distractor count without extra content-gen
    machinery)."""
    matches = sum(1 for b in boxes if b["text"].strip().lower() == target_text.strip().lower())
    if matches > 2:
        return "hard"
    if matches > 1:
        return "medium"
    return "easy"


def _point_task(bbox, point, category, data_type, instruction):
    return {
        "instruction": instruction,
        "bbox": bbox,
        "point": point,
        "answer_type": "point",
        "eval": {"type": "point_in_bbox", "bbox": bbox},
        "data_type": data_type,
        "category": category,
    }


def _bbox_task(bbox, point, category, data_type, instruction):
    return {
        "instruction": instruction,
        "bbox": bbox,
        "point": point,
        "answer_type": "bbox",
        "eval": {
            "type": "bbox_overlap",
            "bbox": bbox,
            "min_coverage": 0.9,
            "min_precision": 0.7,
        },
        "data_type": data_type,
        "category": category,
    }


def build_word_center_task(boxes, rng, language, phrase_fn):
    box = rng.choice(boxes)
    bbox = _round_box(box)
    point = _bbox_center(bbox)
    instruction = phrase_fn("word_center", language, rng, text=box["text"])
    task = _point_task(bbox, point, "word_center", "word", instruction)
    task["_target_text"] = box["text"]
    return task


def build_word_bbox_task(boxes, rng, language, phrase_fn):
    box = rng.choice(boxes)
    bbox = _round_box(box)
    point = _bbox_center(bbox)
    instruction = phrase_fn("word_bbox", language, rng, text=box["text"])
    task = _bbox_task(bbox, point, "word_bbox", "bbox", instruction)
    task["_target_text"] = box["text"]
    return task


def build_line_start_task(boxes, rng, language, phrase_fn):
    lines = cluster_lines(boxes)
    if not lines:
        return None
    line = rng.choice(lines)
    first = line[0]
    bbox = _round_box(first)
    point = [bbox[0], round((bbox[1] + bbox[3]) / 2)]
    instruction = phrase_fn("line_start", language, rng, text=first["text"])
    task = _point_task(bbox, point, "line_start", "caret", instruction)
    task["_target_text"] = first["text"]
    return task


def build_line_end_task(boxes, rng, language, phrase_fn):
    lines = cluster_lines(boxes)
    if not lines:
        return None
    line = rng.choice(lines)
    last = line[-1]
    bbox = _round_box(last)
    point = [bbox[2], round((bbox[1] + bbox[3]) / 2)]
    instruction = phrase_fn("line_end", language, rng, text=last["text"])
    task = _point_task(bbox, point, "line_end", "caret", instruction)
    task["_target_text"] = last["text"]
    return task


BUILDERS = {
    "word_center": build_word_center_task,
    "word_bbox": build_word_bbox_task,
    "line_start": build_line_start_task,
    "line_end": build_line_end_task,
}


def build_task(category, boxes, rng, language, phrase_fn):
    if not boxes:
        return None
    builder = BUILDERS.get(category)
    if builder is None:
        raise ValueError(f"Unknown category: {category}")
    task = builder(boxes, rng, language, phrase_fn)
    if task is None:
        return None
    task["difficulty"] = compute_difficulty(task.pop("_target_text"), boxes)
    return task

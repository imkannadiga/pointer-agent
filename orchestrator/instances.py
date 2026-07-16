"""Occurrence ("nth instance") machinery shared across the whole stack.

Single source of truth for three things that MUST agree with each other or
occurrence tasks silently train/eval against different indices:

- data_gen/task_builder.py computes the ground-truth 1-based occurrence index
  of the chosen word instance with `sort_reading_order` at generation time;
- orchestrator/pipeline.py selects the nth candidate box returned by the VLM
  with the same sort at inference time;
- vlm/dataset.py and orchestrator/grounder_florence2.py build the identical
  enriched referring expression (`format_grounding_phrase`) at train time and
  inference time respectively, so the grounder never sees a phrase shape at
  inference that SFT never showed it.

Pure Python on purpose (no torch/transformers imports) so data_gen can import
it without dragging model dependencies into the generation pipeline.

Per the section-4b architecture principle, ordinal counting is plain Python
over geometry - the deterministic `select_nth` is the primary mechanism; the
enriched phrase only helps the VLM rank instances when it returns fewer
candidates than actually exist on the page.
"""


def sort_reading_order(items: list, bbox_of=None) -> list:
    """Sort boxes in reading order: line by line top-to-bottom, left-to-right
    within a line. Lines are clustered greedily on y-center with a tolerance
    of half the median box height (so slight baseline jitter doesn't split a
    visual line in two).

    `items` are [x0, y0, x1, y1] boxes unless `bbox_of` maps an item to one.
    """
    bbox_of = bbox_of or (lambda item: item)
    if not items:
        return []
    pairs = [(item, bbox_of(item)) for item in items]
    heights = sorted(b[3] - b[1] for _, b in pairs)
    tol = max(heights[len(heights) // 2] / 2.0, 1.0)

    pairs.sort(key=lambda p: ((p[1][1] + p[1][3]) / 2.0, p[1][0]))
    lines = []  # each: {"cy_sum": float, "members": [(item, bbox), ...]}
    for item, b in pairs:
        cy = (b[1] + b[3]) / 2.0
        # pairs are y-sorted, so only the most recent line can still match
        if lines and abs(cy - lines[-1]["cy_sum"] / len(lines[-1]["members"])) <= tol:
            lines[-1]["members"].append((item, b))
            lines[-1]["cy_sum"] += cy
        else:
            lines.append({"cy_sum": cy, "members": [(item, b)]})

    out = []
    for line in lines:
        line["members"].sort(key=lambda p: p[1][0])
        out.extend(item for item, _ in line["members"])
    return out


def occurrence_index(items: list, chosen, bbox_of=None) -> int:
    """1-based reading-order index of `chosen` among `items` (identity
    comparison, so duplicate geometry can't alias two instances)."""
    ordered = sort_reading_order(items, bbox_of=bbox_of)
    for i, item in enumerate(ordered):
        if item is chosen:
            return i + 1
    raise ValueError("chosen item not among items")


def select_nth(candidates: list, n) -> list | None:
    """Pick the nth (1-based, reading order) candidate box.

    Returns None when there is nothing to disambiguate - no/one candidate, or
    no usable n - meaning: keep the grounder's own top box. n beyond the
    candidate count clamps to the last one (the grounder found fewer
    instances than the instruction implies; the last is the best guess for
    "a later occurrence" while staying deterministic).
    """
    try:
        n = int(n)
    except (TypeError, ValueError):
        return None
    if n < 1 or len(candidates) < 2:
        return None
    ordered = sort_reading_order(candidates)
    return ordered[min(n, len(ordered)) - 1]


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def format_grounding_phrase(anchor_phrase: str, n=None) -> str:
    """The literal phrase handed to the VLM grounder. Without n it is the
    bare anchor. With n it becomes a referring expression carrying the
    ordinal context ('the 2nd "Submit"') - Florence-2 handles longer
    referring expressions natively, and SFT (vlm/dataset.py) trains on
    exactly this same shape. Always English regardless of the instruction's
    language: the phrase is constructed by code from (anchor_phrase, n),
    never echoed from the instruction."""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return anchor_phrase
    if n < 1:
        return anchor_phrase
    return f'the {_ordinal(n)} "{anchor_phrase}"'

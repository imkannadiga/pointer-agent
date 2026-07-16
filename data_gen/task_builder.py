"""Boxes (from render.py) -> (instruction, target) task rows.

Sprint 1 shipped two task families (direct-reference, positional/line-based).
Sprint 3 added the remaining families from the real pointerbench-text
taxonomy: relative (between_words), character-level (char_center, char_bbox,
punctuation, blank_line), caret-level (caret_before_word, caret_after_word,
caret_between_chars), line-level (line_bbox, sentence_boundary,
structural_text), paragraph-level (paragraph_start/end/bbox), and the 21
invoice field-extraction categories.

Sprint 4 (see progress.md section 4b) replaced the old `target_phrase`/
`referring_expression` pair with a fixed `{anchor_phrase, relation,
relation_params}` shape on every task, decided statically per category here
(never inferred at runtime): `relation == "self"` means the VLM's own
grounded box for `anchor_phrase` is the final answer; every other relation
is resolved downstream by `orchestrator/char_locator.py` from a crop around
the grounded anchor. Two data-gen bugs found during that design review are
also fixed here: invoice categories now use the semantic `FIELD_LABELS`
descriptor as `anchor_phrase` (not the literal on-page value, which the SLM
planner could never produce without vision), and `structural_text`'s
instruction now quotes the literal title text so the planner has something
to echo back as `anchor_phrase`. `line_start`/`line_end`/`paragraph_start`/
`paragraph_end` also switched from whole-word ground truth to genuine
caret-width ground truth, matching `caret_before_word`/`caret_after_word`'s
existing precision and the real dataset's `data_type: "caret"` categorization.

data_type assignment mirrors the real pointerbench-text 7-value enum
confirmed by inspection (caret, word, invoice, bbox, char, punctuation,
chrome) rather than inventing new buckets: any "_bbox"-shaped category maps
to data_type "bbox" (matching the Sprint 1 word_bbox -> bbox precedent),
positional/cursor categories map to "caret", and structural_text maps to
"chrome" (its real-dataset count of 17 matches exactly).
"""
import re
import string

from data_gen.instruction_templates import FIELD_LABELS as _FIELD_LABELS
from orchestrator.instances import occurrence_index

LINE_CLUSTER_TOL_PX = 6
CARET_WIDTH_PX = 2

ALL_SURFACES = (
    "document", "code_editor", "server_inventory", "magazine_spread",
    "chat_ui", "form", "invoice",
)
PROSE_SURFACES = ("document", "magazine_spread")
STRUCTURAL_SURFACES = ("document", "magazine_spread", "server_inventory", "form")

_CHAR_ID_RE = re.compile(r"^(?P<word_id>.+)c\d+$")


def _round_box(b):
    return [round(b["x0"]), round(b["y0"]), round(b["x1"]), round(b["y1"])]


def _bbox_center(bbox):
    return [round((bbox[0] + bbox[2]) / 2), round((bbox[1] + bbox[3]) / 2)]


def _words(boxes):
    return [b for b in boxes if b["kind"] == "word"]


def _chars(boxes):
    return [b for b in boxes if b["kind"] == "char"]


def _paragraphs(boxes):
    return [b for b in boxes if b["kind"] == "paragraph"]


def _structural(boxes):
    return [b for b in boxes if b["kind"] == "structural"]


def _fields(boxes, field=None):
    fs = [b for b in boxes if b["kind"] == "field"]
    if field:
        fs = [b for b in fs if b["field"] == field]
    return fs


def _char_parent_word_id(char_id):
    m = _CHAR_ID_RE.match(char_id)
    return m.group("word_id") if m else None


def _word_chars(chars, word_id):
    matched = [c for c in chars if c["id"].startswith(word_id + "c")]
    matched.sort(key=lambda c: c["id"])
    return matched


def _words_in_paragraph(words, para, eps=1):
    return [
        w for w in words
        if w["x0"] >= para["x0"] - eps and w["x1"] <= para["x1"] + eps
        and w["y0"] >= para["y0"] - eps and w["y1"] <= para["y1"] + eps
    ]


def _same_text_instances(words, chosen):
    text = chosen["text"].strip().lower()
    return [w for w in words if w["text"].strip().lower() == text]


def _word_occurrence(words, chosen):
    """1-based reading-order index of `chosen` among the page's word boxes
    with the same text, or None when the text is unique (nothing to
    disambiguate). Uses the same sort orchestrator/pipeline.py applies to
    the VLM's candidate boxes at inference, so the index means the same
    thing on both sides."""
    same = _same_text_instances(words, chosen)
    if len(same) < 2:
        return None
    return occurrence_index(
        same, chosen, bbox_of=lambda w: (w["x0"], w["y0"], w["x1"], w["y1"])
    )


def _prefer_unique(words, candidates, rng):
    """Pick a candidate whose text is unique on the page when possible -
    for word-anchored categories that have no occurrence phrasing, a
    repeated anchor makes the ground truth ambiguous (the instruction can't
    say WHICH instance), so repeats are avoided rather than disambiguated."""
    unique = [w for w in candidates if len(_same_text_instances(words, w)) == 1]
    return rng.choice(unique or candidates)


def _occurrence(wchars, target_id, ch_text):
    """1-indexed occurrence count of target_id among wchars matching ch_text
    (case-insensitive), in on-page order - plain string-logic, no model call,
    matching what the planner is expected to reproduce from the instruction."""
    count = 0
    for c in wchars:
        if c["text"].lower() == ch_text.lower():
            count += 1
            if c["id"] == target_id:
                return count
    return count


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


def compute_difficulty(repeat_text, boxes):
    """Heuristic difficulty: more on-page repeats of the anchor's literal
    text = harder (acts as a naturally-occurring distractor count without
    extra content-gen machinery)."""
    if not repeat_text:
        return "easy"
    matches = sum(
        1 for b in boxes
        if b.get("text") and b["text"].strip().lower() == repeat_text.strip().lower()
    )
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


def _finish(task, anchor_phrase, relation, relation_params=None, repeat_text=None, anchor_bbox=None):
    """anchor_bbox is the anchor's OWN bounding box (e.g. the whole word
    "cores"), not the task's final answer bbox - the VLM is only ever
    trained/grounded to find the anchor itself; the final caret/gap/edge
    target (task["bbox"]) is resolved deterministically downstream. Defaults
    to task["bbox"] for relation="self" categories, where the two coincide."""
    task["anchor_phrase"] = anchor_phrase
    task["relation"] = relation
    task["relation_params"] = relation_params or {}
    task["anchor_bbox"] = anchor_bbox if anchor_bbox is not None else task["bbox"]
    task["_repeat_text"] = repeat_text if repeat_text is not None else anchor_phrase
    return task


# ---------------------------------------------------------------------------
# relation: "self" - word/invoice/structural (VLM's own grounded box is final)
# ---------------------------------------------------------------------------

def build_word_center_task(boxes, rng, language, phrase_fn):
    words = _words(boxes)
    if not words:
        return None
    box = rng.choice(words)
    bbox = _round_box(box)
    point = _bbox_center(bbox)
    # Repeated anchor text -> occurrence-disambiguated instruction ("the
    # second X") + relation_params.n, instead of an ambiguous "click X"
    # whose ground truth is one arbitrary instance among several.
    n = _word_occurrence(words, box)
    instruction = phrase_fn("word_center", language, rng, text=box["text"], occurrence=n)
    task = _point_task(bbox, point, "word_center", "word", instruction)
    return _finish(task, box["text"], "self", relation_params={"n": n} if n else {})


def build_word_bbox_task(boxes, rng, language, phrase_fn):
    words = _words(boxes)
    if not words:
        return None
    box = _prefer_unique(words, words, rng)
    bbox = _round_box(box)
    point = _bbox_center(bbox)
    instruction = phrase_fn("word_bbox", language, rng, text=box["text"])
    task = _bbox_task(bbox, point, "word_bbox", "bbox", instruction)
    return _finish(task, box["text"], "self")


def build_structural_text_task(boxes, rng, language, phrase_fn):
    structs = _structural(boxes)
    if not structs:
        return None
    block = rng.choice(structs)
    bbox = _round_box(block)
    point = _bbox_center(bbox)
    title_text = block.get("text") or ""
    instruction = phrase_fn("structural_text", language, rng, text=title_text)
    task = _bbox_task(bbox, point, "structural_text", "chrome", instruction)
    return _finish(task, title_text, "self")


def _make_invoice_field_builder(field_key):
    is_bbox = field_key == "invoice_line_items"

    def _builder(boxes, rng, language, phrase_fn):
        candidates = _fields(boxes, field_key)
        if not candidates:
            return None
        item = rng.choice(candidates) if field_key == "invoice_line_item" else candidates[0]
        bbox = _round_box(item)
        point = _bbox_center(bbox)
        instruction = phrase_fn(field_key, language, rng, field=field_key)
        if is_bbox:
            task = _bbox_task(bbox, point, field_key, "invoice", instruction)
        else:
            task = _point_task(bbox, point, field_key, "invoice", instruction)
        # anchor_phrase is the semantic field label (e.g. "the invoice
        # number"), not item["text"] (the literal value) - the planner has
        # no vision and can never produce the literal value from the
        # instruction alone. repeat_text stays the literal value for the
        # difficulty heuristic only.
        anchor_phrase = _FIELD_LABELS[field_key][language]
        return _finish(task, anchor_phrase, "self", repeat_text=item["text"])

    return _builder


# ---------------------------------------------------------------------------
# relation: line_edge_start / line_edge_end / paragraph_edge_start /
# paragraph_edge_end - caret-width sliver AT the anchor word's own edge
# ---------------------------------------------------------------------------

def build_line_start_task(boxes, rng, language, phrase_fn):
    lines = cluster_lines(_words(boxes))
    if not lines:
        return None
    line = rng.choice(lines)
    first = line[0]
    x0 = first["x0"]
    bbox = [round(x0), round(first["y0"]), round(x0 + CARET_WIDTH_PX), round(first["y1"])]
    point = [round(x0), round((first["y0"] + first["y1"]) / 2)]
    instruction = phrase_fn("line_start", language, rng, text=first["text"])
    task = _point_task(bbox, point, "line_start", "caret", instruction)
    return _finish(task, first["text"], "line_edge_start", anchor_bbox=_round_box(first))


def build_line_end_task(boxes, rng, language, phrase_fn):
    lines = cluster_lines(_words(boxes))
    if not lines:
        return None
    line = rng.choice(lines)
    last = line[-1]
    x1 = last["x1"]
    bbox = [round(x1 - CARET_WIDTH_PX), round(last["y0"]), round(x1), round(last["y1"])]
    point = [round(x1), round((last["y0"] + last["y1"]) / 2)]
    instruction = phrase_fn("line_end", language, rng, text=last["text"])
    task = _point_task(bbox, point, "line_end", "caret", instruction)
    return _finish(task, last["text"], "line_edge_end", anchor_bbox=_round_box(last))


def _paragraph_candidates(boxes):
    words, paras = _words(boxes), _paragraphs(boxes)
    out = [(p, _words_in_paragraph(words, p)) for p in paras]
    return [(p, ws) for p, ws in out if ws]


def build_paragraph_start_task(boxes, rng, language, phrase_fn):
    candidates = _paragraph_candidates(boxes)
    if not candidates:
        return None
    _, ws = rng.choice(candidates)
    first = sorted(ws, key=lambda w: (w["y0"], w["x0"]))[0]
    x0 = first["x0"]
    bbox = [round(x0), round(first["y0"]), round(x0 + CARET_WIDTH_PX), round(first["y1"])]
    point = [round(x0), round((first["y0"] + first["y1"]) / 2)]
    instruction = phrase_fn("paragraph_start", language, rng, text=first["text"])
    task = _point_task(bbox, point, "paragraph_start", "caret", instruction)
    return _finish(task, first["text"], "paragraph_edge_start", anchor_bbox=_round_box(first))


def build_paragraph_end_task(boxes, rng, language, phrase_fn):
    candidates = _paragraph_candidates(boxes)
    if not candidates:
        return None
    _, ws = rng.choice(candidates)
    last = sorted(ws, key=lambda w: (w["y0"], w["x0"]))[-1]
    x1 = last["x1"]
    bbox = [round(x1 - CARET_WIDTH_PX), round(last["y0"]), round(x1), round(last["y1"])]
    point = [round(x1), round((last["y0"] + last["y1"]) / 2)]
    instruction = phrase_fn("paragraph_end", language, rng, text=last["text"])
    task = _point_task(bbox, point, "paragraph_end", "caret", instruction)
    return _finish(task, last["text"], "paragraph_edge_end", anchor_bbox=_round_box(last))


# ---------------------------------------------------------------------------
# relation: before_word / after_word - caret-width sliver OUTSIDE the anchor
# word's edge (with an outward margin, unlike line/paragraph edge relations)
# ---------------------------------------------------------------------------

def build_caret_before_word_task(boxes, rng, language, phrase_fn):
    words = _words(boxes)
    if not words:
        return None
    word = rng.choice(words)
    x = word["x0"]
    bbox = [round(x - CARET_WIDTH_PX), round(word["y0"]), round(x), round(word["y1"])]
    point = [round(x), round((word["y0"] + word["y1"]) / 2)]
    n = _word_occurrence(words, word)
    instruction = phrase_fn("caret_before_word", language, rng, text=word["text"], occurrence=n)
    task = _point_task(bbox, point, "caret_before_word", "caret", instruction)
    return _finish(
        task, word["text"], "before_word",
        relation_params={"n": n} if n else {},
        anchor_bbox=_round_box(word),
    )


def build_caret_after_word_task(boxes, rng, language, phrase_fn):
    words = _words(boxes)
    if not words:
        return None
    word = rng.choice(words)
    x = word["x1"]
    bbox = [round(x), round(word["y0"]), round(x + CARET_WIDTH_PX), round(word["y1"])]
    point = [round(x), round((word["y0"] + word["y1"]) / 2)]
    n = _word_occurrence(words, word)
    instruction = phrase_fn("caret_after_word", language, rng, text=word["text"], occurrence=n)
    task = _point_task(bbox, point, "caret_after_word", "caret", instruction)
    return _finish(
        task, word["text"], "after_word",
        relation_params={"n": n} if n else {},
        anchor_bbox=_round_box(word),
    )


def build_caret_between_chars_task(boxes, rng, language, phrase_fn):
    words, chars = _words(boxes), _chars(boxes)
    candidates = [w for w in words if len(_word_chars(chars, w["id"])) >= 2]
    if not candidates:
        return None
    word = rng.choice(candidates)
    wchars = _word_chars(chars, word["id"])
    i = rng.randrange(len(wchars) - 1)
    c1, c2 = wchars[i], wchars[i + 1]
    x = (c1["x1"] + c2["x0"]) / 2
    y0, y1 = min(c1["y0"], c2["y0"]), max(c1["y1"], c2["y1"])
    bbox = [round(x - CARET_WIDTH_PX / 2), round(y0), round(x + CARET_WIDTH_PX / 2), round(y1)]
    point = [round(x), round((y0 + y1) / 2)]
    n = _word_occurrence(words, word)
    instruction = phrase_fn(
        "caret_between_chars", language, rng,
        ch1=c1["text"], ch2=c2["text"], text=word["text"], occurrence=n,
    )
    task = _point_task(bbox, point, "caret_between_chars", "caret", instruction)
    params = {"char1": c1["text"], "char2": c2["text"], "index": i}
    if n:
        params["n"] = n
    return _finish(
        task, word["text"], "between_chars",
        relation_params=params,
        anchor_bbox=_round_box(word),
    )


# ---------------------------------------------------------------------------
# relation: char_at_occurrence - char_center / char_bbox / punctuation
# ---------------------------------------------------------------------------

def build_char_center_task(boxes, rng, language, phrase_fn):
    words, chars = _words(boxes), _chars(boxes)
    candidates = [w for w in words if _word_chars(chars, w["id"])]
    if not candidates:
        return None
    word = _prefer_unique(words, candidates, rng)
    wchars = _word_chars(chars, word["id"])
    ch = rng.choice(wchars)
    bbox = _round_box(ch)
    point = _bbox_center(bbox)
    instruction = phrase_fn("char_center", language, rng, ch=ch["text"], text=word["text"])
    task = _point_task(bbox, point, "char_center", "char", instruction)
    occurrence = _occurrence(wchars, ch["id"], ch["text"])
    return _finish(
        task, word["text"], "char_at_occurrence",
        relation_params={"target_char": ch["text"], "occurrence": occurrence},
        anchor_bbox=_round_box(word),
    )


def build_char_bbox_task(boxes, rng, language, phrase_fn):
    words, chars = _words(boxes), _chars(boxes)
    candidates = [w for w in words if _word_chars(chars, w["id"])]
    if not candidates:
        return None
    word = _prefer_unique(words, candidates, rng)
    wchars = _word_chars(chars, word["id"])
    ch = rng.choice(wchars)
    bbox = _round_box(ch)
    point = _bbox_center(bbox)
    instruction = phrase_fn("char_bbox", language, rng, ch=ch["text"], text=word["text"])
    task = _bbox_task(bbox, point, "char_bbox", "bbox", instruction)
    occurrence = _occurrence(wchars, ch["id"], ch["text"])
    return _finish(
        task, word["text"], "char_at_occurrence",
        relation_params={"target_char": ch["text"], "occurrence": occurrence},
        anchor_bbox=_round_box(word),
    )


def build_punctuation_task(boxes, rng, language, phrase_fn):
    words, chars = _words(boxes), _chars(boxes)
    punct_chars = [c for c in chars if c["text"] in string.punctuation]
    if not punct_chars:
        return None
    ch = rng.choice(punct_chars)
    word_id = _char_parent_word_id(ch["id"])
    word = next((w for w in words if w["id"] == word_id), None)
    word_text = word["text"] if word else ""
    bbox = _round_box(ch)
    point = _bbox_center(bbox)
    instruction = phrase_fn("punctuation", language, rng, ch=ch["text"], text=word_text)
    task = _point_task(bbox, point, "punctuation", "punctuation", instruction)
    wchars = _word_chars(chars, word_id) if word_id else [ch]
    occurrence = _occurrence(wchars, ch["id"], ch["text"])
    anchor_bbox = _round_box(word) if word else _round_box(ch)
    return _finish(
        task, word_text, "char_at_occurrence",
        relation_params={"target_char": ch["text"], "occurrence": occurrence},
        anchor_bbox=anchor_bbox,
    )


# ---------------------------------------------------------------------------
# relation: sentence_end / line_bbox / paragraph_bbox
# ---------------------------------------------------------------------------

_SENTENCE_ENDERS = {".", "!", "?"}


def build_sentence_boundary_task(boxes, rng, language, phrase_fn):
    words, chars = _words(boxes), _chars(boxes)
    enders = [c for c in chars if c["text"] in _SENTENCE_ENDERS]
    if not enders:
        return None
    ch = rng.choice(enders)
    word_id = _char_parent_word_id(ch["id"])
    word = next((w for w in words if w["id"] == word_id), None)
    word_text = word["text"] if word else ch["text"]
    bbox = _round_box(ch)
    point = _bbox_center(bbox)
    instruction = phrase_fn("sentence_boundary", language, rng, text=word_text)
    task = _point_task(bbox, point, "sentence_boundary", "caret", instruction)
    anchor_bbox = _round_box(word) if word else _round_box(ch)
    return _finish(task, word_text, "sentence_end", anchor_bbox=anchor_bbox)


def build_line_bbox_task(boxes, rng, language, phrase_fn):
    lines = cluster_lines(_words(boxes))
    if not lines:
        return None
    line = rng.choice(lines)
    x0 = min(w["x0"] for w in line)
    y0 = min(w["y0"] for w in line)
    x1 = max(w["x1"] for w in line)
    y1 = max(w["y1"] for w in line)
    bbox = [round(x0), round(y0), round(x1), round(y1)]
    point = _bbox_center(bbox)
    instruction = phrase_fn("line_bbox", language, rng, text=line[0]["text"])
    task = _bbox_task(bbox, point, "line_bbox", "bbox", instruction)
    return _finish(task, line[0]["text"], "line_bbox", anchor_bbox=_round_box(line[0]))


def build_paragraph_bbox_task(boxes, rng, language, phrase_fn):
    candidates = _paragraph_candidates(boxes)
    if not candidates:
        return None
    para, ws = rng.choice(candidates)
    first = sorted(ws, key=lambda w: (w["y0"], w["x0"]))[0]
    bbox = _round_box(para)
    point = _bbox_center(bbox)
    instruction = phrase_fn("paragraph_bbox", language, rng, text=first["text"])
    task = _bbox_task(bbox, point, "paragraph_bbox", "bbox", instruction)
    return _finish(task, first["text"], "paragraph_bbox", anchor_bbox=_round_box(first))


# ---------------------------------------------------------------------------
# relation: between_anchors - two separate VLM grounding calls, pure geometry
# gap between them (no char locator)
# ---------------------------------------------------------------------------

def build_between_words_task(boxes, rng, language, phrase_fn):
    lines = cluster_lines(_words(boxes))
    candidates = [l for l in lines if len(l) >= 2]
    if not candidates:
        return None
    line = rng.choice(candidates)
    i = rng.randrange(len(line) - 1)
    w1, w2 = line[i], line[i + 1]
    x = (w1["x1"] + w2["x0"]) / 2
    y0, y1 = min(w1["y0"], w2["y0"]), max(w1["y1"], w2["y1"])
    bbox = [round(x - 4), round(y0), round(x + 4), round(y1)]
    point = [round(x), round((y0 + y1) / 2)]
    # The ordinal disambiguates the FIRST anchor only - the second anchor
    # rides along as a raw phrase (relation_params keeps its single
    # second_anchor_phrase shape, per the section-4b two-anchor decision).
    n = _word_occurrence(_words(boxes), w1)
    instruction = phrase_fn(
        "between_words", language, rng, text1=w1["text"], text2=w2["text"], occurrence=n
    )
    task = _point_task(bbox, point, "between_words", "word", instruction)
    params = {"second_anchor_phrase": w2["text"]}
    if n:
        params["n"] = n
    return _finish(
        task, w1["text"], "between_anchors",
        relation_params=params,
        anchor_bbox=_round_box(w1),
    )


def build_blank_line_task(boxes, rng, language, phrase_fn):
    words = _words(boxes)
    paras = sorted(_paragraphs(boxes), key=lambda p: p["y0"])
    if len(paras) < 2:
        return None
    idx = rng.randrange(len(paras) - 1)
    p1, p2 = paras[idx], paras[idx + 1]
    gap_y0, gap_y1 = p1["y1"], p2["y0"]
    if gap_y1 - gap_y0 < 2:
        return None
    ws1 = _words_in_paragraph(words, p1)
    ws2 = _words_in_paragraph(words, p2)
    if not ws1 or not ws2:
        return None
    first1 = sorted(ws1, key=lambda w: (w["y0"], w["x0"]))[0]
    first2 = sorted(ws2, key=lambda w: (w["y0"], w["x0"]))[0]
    x0 = max(min(p1["x0"], p2["x0"]), 0)
    x1 = max(p1["x1"], p2["x1"])
    bbox = [round(x0), round(gap_y0), round(x1), round(gap_y1)]
    point = _bbox_center(bbox)
    instruction = phrase_fn("blank_line", language, rng)
    task = _point_task(bbox, point, "blank_line", "caret", instruction)
    return _finish(
        task, first1["text"], "between_anchors",
        relation_params={"second_anchor_phrase": first2["text"]},
        repeat_text="",
        anchor_bbox=_round_box(first1),
    )


BUILDERS = {
    "word_center": build_word_center_task,
    "word_bbox": build_word_bbox_task,
    "line_start": build_line_start_task,
    "line_end": build_line_end_task,
    "between_words": build_between_words_task,
    "char_center": build_char_center_task,
    "char_bbox": build_char_bbox_task,
    "punctuation": build_punctuation_task,
    "blank_line": build_blank_line_task,
    "caret_before_word": build_caret_before_word_task,
    "caret_after_word": build_caret_after_word_task,
    "caret_between_chars": build_caret_between_chars_task,
    "line_bbox": build_line_bbox_task,
    "sentence_boundary": build_sentence_boundary_task,
    "structural_text": build_structural_text_task,
    "paragraph_start": build_paragraph_start_task,
    "paragraph_end": build_paragraph_end_task,
    "paragraph_bbox": build_paragraph_bbox_task,
}
for _field_key in _FIELD_LABELS:
    BUILDERS[_field_key] = _make_invoice_field_builder(_field_key)

CATEGORY_SURFACES = {
    "word_center": ALL_SURFACES,
    "word_bbox": ALL_SURFACES,
    "line_start": ALL_SURFACES,
    "line_end": ALL_SURFACES,
    "between_words": ALL_SURFACES,
    "caret_before_word": ALL_SURFACES,
    "caret_after_word": ALL_SURFACES,
    "line_bbox": ALL_SURFACES,
    "char_center": PROSE_SURFACES,
    "char_bbox": PROSE_SURFACES,
    "punctuation": PROSE_SURFACES,
    "caret_between_chars": PROSE_SURFACES,
    "sentence_boundary": PROSE_SURFACES,
    "blank_line": PROSE_SURFACES,
    "paragraph_start": PROSE_SURFACES,
    "paragraph_end": PROSE_SURFACES,
    "paragraph_bbox": PROSE_SURFACES,
    "structural_text": STRUCTURAL_SURFACES,
}
for _field_key in _FIELD_LABELS:
    CATEGORY_SURFACES[_field_key] = ("invoice",)

# Categories restricted to en/de only in the real pointerbench-text benchmark
# (confirmed by re-reading its subset README - see progress.md section 4b).
# Every other category still generates across all configured languages.
NARROW_LANGUAGE_CATEGORIES = {
    "char_center", "char_bbox", "punctuation", "blank_line",
    "caret_before_word", "caret_after_word", "caret_between_chars",
    "sentence_boundary",
}
NARROW_LANGUAGES = ("en", "de")

ALL_CATEGORIES = tuple(BUILDERS.keys())


def languages_for_category(category, configured_languages):
    if category in NARROW_LANGUAGE_CATEGORIES:
        return [l for l in configured_languages if l in NARROW_LANGUAGES] or list(NARROW_LANGUAGES)
    return list(configured_languages)


def build_task(category, boxes, rng, language, phrase_fn):
    if not boxes:
        return None
    builder = BUILDERS.get(category)
    if builder is None:
        raise ValueError(f"Unknown category: {category}")
    task = builder(boxes, rng, language, phrase_fn)
    if task is None:
        return None
    repeat_text = task.pop("_repeat_text", task.get("anchor_phrase", ""))
    task["difficulty"] = compute_difficulty(repeat_text, boxes)
    return task

"""Boxes (from render.py) -> (instruction, target) task rows.

Sprint 1 shipped two task families (direct-reference, positional/line-based).
Sprint 3 adds the remaining families from the real pointerbench-text
taxonomy: relative (between_words), character-level (char_center, char_bbox,
punctuation, blank_line), caret-level (caret_before_word, caret_after_word,
caret_between_chars), line-level (line_bbox, sentence_boundary,
structural_text), paragraph-level (paragraph_start/end/bbox), and the ~20
invoice field-extraction categories.

data_type assignment mirrors the real pointerbench-text 7-value enum
confirmed by inspection (caret, word, invoice, bbox, char, punctuation,
chrome) rather than inventing new buckets: any "_bbox"-shaped category maps
to data_type "bbox" (matching the Sprint 1 word_bbox -> bbox precedent),
positional/cursor categories map to "caret", and structural_text maps to
"chrome" (its real-dataset count of 17 matches exactly).
"""
import re
import string

from data_gen.instruction_templates import REF_BUILDERS as _REF_BUILDERS
from data_gen.instruction_templates import FIELD_LABELS as _FIELD_LABELS

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
    if not target_text:
        return "easy"
    matches = sum(
        1 for b in boxes
        if b.get("text") and b["text"].strip().lower() == target_text.strip().lower()
    )
    if matches > 2:
        return "hard"
    if matches > 1:
        return "medium"
    return "easy"


def _referring_expression(category, language, **kwargs):
    builder = _REF_BUILDERS.get(category)
    return builder(language, **kwargs) if builder else None


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


# ---------------------------------------------------------------------------
# Sprint 1 categories (unchanged behavior)
# ---------------------------------------------------------------------------

def build_word_center_task(boxes, rng, language, phrase_fn):
    words = _words(boxes)
    if not words:
        return None
    box = rng.choice(words)
    bbox = _round_box(box)
    point = _bbox_center(bbox)
    instruction = phrase_fn("word_center", language, rng, text=box["text"])
    task = _point_task(bbox, point, "word_center", "word", instruction)
    task["target_phrase"] = box["text"]
    return task


def build_word_bbox_task(boxes, rng, language, phrase_fn):
    words = _words(boxes)
    if not words:
        return None
    box = rng.choice(words)
    bbox = _round_box(box)
    point = _bbox_center(bbox)
    instruction = phrase_fn("word_bbox", language, rng, text=box["text"])
    task = _bbox_task(bbox, point, "word_bbox", "bbox", instruction)
    task["target_phrase"] = box["text"]
    return task


def build_line_start_task(boxes, rng, language, phrase_fn):
    lines = cluster_lines(_words(boxes))
    if not lines:
        return None
    line = rng.choice(lines)
    first = line[0]
    bbox = _round_box(first)
    point = [bbox[0], round((bbox[1] + bbox[3]) / 2)]
    instruction = phrase_fn("line_start", language, rng, text=first["text"])
    task = _point_task(bbox, point, "line_start", "caret", instruction)
    task["target_phrase"] = first["text"]
    return task


def build_line_end_task(boxes, rng, language, phrase_fn):
    lines = cluster_lines(_words(boxes))
    if not lines:
        return None
    line = rng.choice(lines)
    last = line[-1]
    bbox = _round_box(last)
    point = [bbox[2], round((bbox[1] + bbox[3]) / 2)]
    instruction = phrase_fn("line_end", language, rng, text=last["text"])
    task = _point_task(bbox, point, "line_end", "caret", instruction)
    task["target_phrase"] = last["text"]
    return task


# ---------------------------------------------------------------------------
# Sprint 3: word-level relative
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
    instruction = phrase_fn("between_words", language, rng, text1=w1["text"], text2=w2["text"])
    task = _point_task(bbox, point, "between_words", "word", instruction)
    task["target_phrase"] = w1["text"]
    task["referring_expression"] = _referring_expression(
        "between_words", language, text1=w1["text"], text2=w2["text"]
    )
    return task


# ---------------------------------------------------------------------------
# Sprint 3: character-level (prose surfaces only)
# ---------------------------------------------------------------------------

def build_char_center_task(boxes, rng, language, phrase_fn):
    words, chars = _words(boxes), _chars(boxes)
    candidates = [w for w in words if _word_chars(chars, w["id"])]
    if not candidates:
        return None
    word = rng.choice(candidates)
    ch = rng.choice(_word_chars(chars, word["id"]))
    bbox = _round_box(ch)
    point = _bbox_center(bbox)
    instruction = phrase_fn("char_center", language, rng, ch=ch["text"], text=word["text"])
    task = _point_task(bbox, point, "char_center", "char", instruction)
    task["target_phrase"] = word["text"]
    return task


def build_char_bbox_task(boxes, rng, language, phrase_fn):
    words, chars = _words(boxes), _chars(boxes)
    candidates = [w for w in words if _word_chars(chars, w["id"])]
    if not candidates:
        return None
    word = rng.choice(candidates)
    ch = rng.choice(_word_chars(chars, word["id"]))
    bbox = _round_box(ch)
    point = _bbox_center(bbox)
    instruction = phrase_fn("char_bbox", language, rng, ch=ch["text"], text=word["text"])
    task = _bbox_task(bbox, point, "char_bbox", "bbox", instruction)
    task["target_phrase"] = word["text"]
    return task


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
    task["target_phrase"] = word_text
    return task


def build_blank_line_task(boxes, rng, language, phrase_fn):
    paras = sorted(_paragraphs(boxes), key=lambda p: p["y0"])
    if len(paras) < 2:
        return None
    idx = rng.randrange(len(paras) - 1)
    p1, p2 = paras[idx], paras[idx + 1]
    gap_y0, gap_y1 = p1["y1"], p2["y0"]
    if gap_y1 - gap_y0 < 2:
        return None
    x0 = max(min(p1["x0"], p2["x0"]), 0)
    x1 = max(p1["x1"], p2["x1"])
    bbox = [round(x0), round(gap_y0), round(x1), round(gap_y1)]
    point = _bbox_center(bbox)
    instruction = phrase_fn("blank_line", language, rng)
    task = _point_task(bbox, point, "blank_line", "caret", instruction)
    task["target_phrase"] = ""
    return task


# ---------------------------------------------------------------------------
# Sprint 3: caret-level (prose surfaces for between_chars; any surface for
# before/after word since those only need a single word's box edges)
# ---------------------------------------------------------------------------

def build_caret_before_word_task(boxes, rng, language, phrase_fn):
    words = _words(boxes)
    if not words:
        return None
    word = rng.choice(words)
    x = word["x0"]
    bbox = [round(x - CARET_WIDTH_PX), round(word["y0"]), round(x), round(word["y1"])]
    point = [round(x), round((word["y0"] + word["y1"]) / 2)]
    instruction = phrase_fn("caret_before_word", language, rng, text=word["text"])
    task = _point_task(bbox, point, "caret_before_word", "caret", instruction)
    task["target_phrase"] = word["text"]
    task["referring_expression"] = _referring_expression(
        "caret_before_word", language, text=word["text"]
    )
    return task


def build_caret_after_word_task(boxes, rng, language, phrase_fn):
    words = _words(boxes)
    if not words:
        return None
    word = rng.choice(words)
    x = word["x1"]
    bbox = [round(x), round(word["y0"]), round(x + CARET_WIDTH_PX), round(word["y1"])]
    point = [round(x), round((word["y0"] + word["y1"]) / 2)]
    instruction = phrase_fn("caret_after_word", language, rng, text=word["text"])
    task = _point_task(bbox, point, "caret_after_word", "caret", instruction)
    task["target_phrase"] = word["text"]
    task["referring_expression"] = _referring_expression(
        "caret_after_word", language, text=word["text"]
    )
    return task


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
    instruction = phrase_fn(
        "caret_between_chars", language, rng, ch1=c1["text"], ch2=c2["text"], text=word["text"]
    )
    task = _point_task(bbox, point, "caret_between_chars", "caret", instruction)
    task["target_phrase"] = word["text"]
    task["referring_expression"] = _referring_expression(
        "caret_between_chars", language, ch1=c1["text"], ch2=c2["text"], text=word["text"]
    )
    return task


# ---------------------------------------------------------------------------
# Sprint 3: line-level
# ---------------------------------------------------------------------------

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
    task["target_phrase"] = line[0]["text"]
    return task


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
    task["target_phrase"] = word_text
    return task


def build_structural_text_task(boxes, rng, language, phrase_fn):
    structs = _structural(boxes)
    if not structs:
        return None
    block = rng.choice(structs)
    bbox = _round_box(block)
    point = _bbox_center(bbox)
    instruction = phrase_fn("structural_text", language, rng)
    task = _bbox_task(bbox, point, "structural_text", "chrome", instruction)
    task["target_phrase"] = ""
    return task


# ---------------------------------------------------------------------------
# Sprint 3: paragraph-level (prose surfaces only)
# ---------------------------------------------------------------------------

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
    bbox = _round_box(first)
    point = [bbox[0], round((bbox[1] + bbox[3]) / 2)]
    instruction = phrase_fn("paragraph_start", language, rng, text=first["text"])
    task = _point_task(bbox, point, "paragraph_start", "caret", instruction)
    task["target_phrase"] = first["text"]
    return task


def build_paragraph_end_task(boxes, rng, language, phrase_fn):
    candidates = _paragraph_candidates(boxes)
    if not candidates:
        return None
    _, ws = rng.choice(candidates)
    last = sorted(ws, key=lambda w: (w["y0"], w["x0"]))[-1]
    bbox = _round_box(last)
    point = [bbox[2], round((bbox[1] + bbox[3]) / 2)]
    instruction = phrase_fn("paragraph_end", language, rng, text=last["text"])
    task = _point_task(bbox, point, "paragraph_end", "caret", instruction)
    task["target_phrase"] = last["text"]
    return task


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
    task["target_phrase"] = first["text"]
    return task


# ---------------------------------------------------------------------------
# Sprint 3: invoice field extraction (invoice surface only)
# ---------------------------------------------------------------------------

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
        task["target_phrase"] = item["text"]
        return task

    return _builder


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

ALL_CATEGORIES = tuple(BUILDERS.keys())


def build_task(category, boxes, rng, language, phrase_fn):
    if not boxes:
        return None
    builder = BUILDERS.get(category)
    if builder is None:
        raise ValueError(f"Unknown category: {category}")
    task = builder(boxes, rng, language, phrase_fn)
    if task is None:
        return None
    task["difficulty"] = compute_difficulty(task.get("target_phrase", ""), boxes)
    task.setdefault("referring_expression", None)
    return task

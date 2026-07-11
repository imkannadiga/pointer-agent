"""Deterministic (non-model) resolution of the precise final point/bbox for
relational categories, given a VLM-grounded anchor bbox. See
`progress.md` section 4b for the design and `orchestrator/base.py` for the
`BaseCharLocator` contract this implements.

Coordinate handling is the single riskiest part of this module, and gets
handled in exactly one place (`_char_boxes`) rather than per-relation:
Tesseract's `image_to_boxes()` API returns **bottom-left-origin, y-growing-
upward** coordinates (a well-known Tesseract gotcha, distinct from
`image_to_data()`'s ordinary top-left-origin `left/top/width/height`), and
both are crop-local and upscaled on top of that. Every coordinate this
module returns has been flipped, descaled, and offset back to full-image,
top-left-origin pixels before it leaves this file - see
`tests/test_char_locator.py` for the correctness proof against a real
Tesseract run.
"""
import logging

import pytesseract
from PIL import Image, ImageOps

from orchestrator.base import BaseCharLocator

logger = logging.getLogger(__name__)

UPSCALE = 4
CROP_MARGIN_PX = 10
LINE_CROP_MARGIN_PX = 6
PARAGRAPH_CROP_MARGIN_PX = 250
CARET_WIDTH_PX = 2
TESS_CONFIG = "--psm 8"
_SENTENCE_ENDERS = {".", "!", "?"}


def _bbox_of(box: dict) -> list[int]:
    return [round(box["x0"]), round(box["y0"]), round(box["x1"]), round(box["y1"])]


def _center(bbox: list[int]) -> list[int]:
    return [round((bbox[0] + bbox[2]) / 2), round((bbox[1] + bbox[3]) / 2)]


def _crop_with_margin(image: Image.Image, bbox: list[int], margin: int = CROP_MARGIN_PX):
    """Returns (crop, (offset_x, offset_y)) - offset is the crop's top-left
    corner in the ORIGINAL image's coordinates, needed to translate anything
    found in the crop back to full-image coordinates."""
    x0, y0, x1, y1 = bbox
    cx0 = max(0, x0 - margin)
    cy0 = max(0, y0 - margin)
    cx1 = min(image.width, x1 + margin)
    cy1 = min(image.height, y1 + margin)
    return image.crop((cx0, cy0, cx1, cy1)), (cx0, cy0)


def _prep_for_ocr(crop: Image.Image, upscale: int = UPSCALE) -> Image.Image:
    """Upscale + grayscale + auto-invert-if-dark - all deterministic, no
    model involvement. Dark-theme screenshots have light text on a dark
    background; Tesseract expects dark text on a light background."""
    w, h = crop.size
    upscaled = crop.resize((max(1, w * upscale), max(1, h * upscale)), Image.LANCZOS)
    gray = upscaled.convert("L")
    lo, hi = gray.getextrema()
    if (lo + hi) / 2 < 128:
        gray = ImageOps.invert(gray)
    return gray


def _char_boxes(crop_prepped: Image.Image, crop_offset: tuple, upscale: int = UPSCALE) -> list[dict]:
    """Runs Tesseract's character-box API on an already-upscaled crop and
    returns each character in FULL-IMAGE, top-left-origin pixel coordinates.

    `image_to_boxes()`'s own coordinate system is bottom-left-origin with y
    growing upward (`bottom`/`top` are both measured from the image's
    bottom edge, with `top > bottom` for every character) - that gets
    flipped to top-left-origin here, then descaled by `upscale` and offset
    by `crop_offset`, all in one place so no caller has to re-derive this.
    """
    ox, oy = crop_offset
    h = crop_prepped.height
    raw = pytesseract.image_to_boxes(crop_prepped, config=TESS_CONFIG)
    out = []
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) != 6:
            continue
        ch, left, bottom, right, top, _page = parts
        left, bottom, right, top = int(left), int(bottom), int(right), int(top)
        y0_crop, y1_crop = h - top, h - bottom  # flip bottom-left -> top-left origin
        out.append({
            "ch": ch,
            "x0": left / upscale + ox,
            "y0": y0_crop / upscale + oy,
            "x1": right / upscale + ox,
            "y1": y1_crop / upscale + oy,
        })
    return out


def _words_from_data(data: dict, crop_offset: tuple, upscale: int = UPSCALE) -> list[dict]:
    """`image_to_data()` (unlike `image_to_boxes()`) already uses ordinary
    top-left-origin `left/top/width/height` - no y-flip needed here, only
    descaling + the crop offset."""
    ox, oy = crop_offset
    out = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if not text:
            continue
        left, top = data["left"][i], data["top"][i]
        width, height = data["width"][i], data["height"][i]
        out.append({
            "text": text,
            "x0": left / upscale + ox,
            "y0": top / upscale + oy,
            "x1": (left + width) / upscale + ox,
            "y1": (top + height) / upscale + oy,
            "line_num": data["line_num"][i],
            "par_num": data["par_num"][i],
            "block_num": data["block_num"][i],
        })
    return out


def _best_overlap(words: list[dict], anchor_bbox: list[int]) -> dict | None:
    best, best_score = None, 0.0
    for w in words:
        ox0, oy0 = max(w["x0"], anchor_bbox[0]), max(w["y0"], anchor_bbox[1])
        ox1, oy1 = min(w["x1"], anchor_bbox[2]), min(w["y1"], anchor_bbox[3])
        if ox1 > ox0 and oy1 > oy0:
            score = (ox1 - ox0) * (oy1 - oy0)
            if score > best_score:
                best, best_score = w, score
    return best


class TesseractCharLocator(BaseCharLocator):
    def resolve(self, image, anchor_phrase, anchor_bbox, relation, relation_params, answer_type):
        handler = getattr(self, f"_resolve_{relation}", None)
        if handler is None:
            raise ValueError(f"TesseractCharLocator has no resolver for relation: {relation}")
        return handler(image, anchor_phrase, anchor_bbox, relation_params or {})

    def _word_box(self, image: Image.Image, anchor_phrase: str, anchor_bbox: list[int]) -> list[int]:
        """OCR the anchor crop and return the tightest full-word bbox found.
        Sanity-checks the OCR'd character count against len(anchor_phrase)
        (ignoring spaces) before trusting it - falls back to anchor_bbox
        itself (the VLM's own box) if OCR looks unreliable, rather than
        risking a confidently-wrong precise answer."""
        crop, offset = _crop_with_margin(image, anchor_bbox)
        chars = _char_boxes(_prep_for_ocr(crop), offset)
        expected = len(anchor_phrase.replace(" ", ""))
        if not chars or expected == 0 or abs(len(chars) - expected) > max(2, expected // 2):
            logger.warning(
                "OCR sanity check failed for anchor_phrase=%r (expected %d chars, "
                "OCR found %d) - falling back to anchor_bbox %s unchanged",
                anchor_phrase, expected, len(chars), anchor_bbox,
            )
            return list(anchor_bbox)
        x0 = min(c["x0"] for c in chars)
        y0 = min(c["y0"] for c in chars)
        x1 = max(c["x1"] for c in chars)
        y1 = max(c["y1"] for c in chars)
        return [round(x0), round(y0), round(x1), round(y1)]

    # -- word-edge relations (precise word bbox, no outward margin) --------

    def _resolve_line_edge_start(self, image, anchor_phrase, anchor_bbox, params):
        wbox = self._word_box(image, anchor_phrase, anchor_bbox)
        x = wbox[0]
        bbox = [x, wbox[1], x + CARET_WIDTH_PX, wbox[3]]
        return {"bbox": bbox, "point": [x, round((wbox[1] + wbox[3]) / 2)]}

    def _resolve_line_edge_end(self, image, anchor_phrase, anchor_bbox, params):
        wbox = self._word_box(image, anchor_phrase, anchor_bbox)
        x = wbox[2]
        bbox = [x - CARET_WIDTH_PX, wbox[1], x, wbox[3]]
        return {"bbox": bbox, "point": [x, round((wbox[1] + wbox[3]) / 2)]}

    _resolve_paragraph_edge_start = _resolve_line_edge_start
    _resolve_paragraph_edge_end = _resolve_line_edge_end

    # -- caret relations (word edge + outward margin) ----------------------

    def _resolve_before_word(self, image, anchor_phrase, anchor_bbox, params):
        wbox = self._word_box(image, anchor_phrase, anchor_bbox)
        x = wbox[0]
        bbox = [x - CARET_WIDTH_PX, wbox[1], x, wbox[3]]
        return {"bbox": bbox, "point": [x, round((wbox[1] + wbox[3]) / 2)]}

    def _resolve_after_word(self, image, anchor_phrase, anchor_bbox, params):
        wbox = self._word_box(image, anchor_phrase, anchor_bbox)
        x = wbox[2]
        bbox = [x, wbox[1], x + CARET_WIDTH_PX, wbox[3]]
        return {"bbox": bbox, "point": [x, round((wbox[1] + wbox[3]) / 2)]}

    def _resolve_between_chars(self, image, anchor_phrase, anchor_bbox, params):
        crop, offset = _crop_with_margin(image, anchor_bbox)
        chars = sorted(_char_boxes(_prep_for_ocr(crop), offset), key=lambda c: c["x0"])
        index = params.get("index")
        c1_text, c2_text = params.get("char1"), params.get("char2")
        pair = None
        if isinstance(index, int) and 0 <= index < len(chars) - 1:
            pair = (chars[index], chars[index + 1])
        elif c1_text and c2_text:
            for i in range(len(chars) - 1):
                if chars[i]["ch"] == c1_text and chars[i + 1]["ch"] == c2_text:
                    pair = (chars[i], chars[i + 1])
                    break
        if pair is None:
            wbox = self._word_box(image, anchor_phrase, anchor_bbox)
            return {"bbox": wbox, "point": _center(wbox)}
        c1, c2 = pair
        x = (c1["x1"] + c2["x0"]) / 2
        y0, y1 = min(c1["y0"], c2["y0"]), max(c1["y1"], c2["y1"])
        bbox = [round(x - CARET_WIDTH_PX / 2), round(y0), round(x + CARET_WIDTH_PX / 2), round(y1)]
        return {"bbox": bbox, "point": [round(x), round((y0 + y1) / 2)]}

    # -- character/sentence lookup ------------------------------------------

    def _resolve_char_at_occurrence(self, image, anchor_phrase, anchor_bbox, params):
        crop, offset = _crop_with_margin(image, anchor_bbox)
        chars = sorted(_char_boxes(_prep_for_ocr(crop), offset), key=lambda c: c["x0"])
        target = params.get("target_char", "")
        occurrence = params.get("occurrence", 1)
        count = 0
        match = None
        for c in chars:
            if c["ch"].lower() == str(target).lower():
                count += 1
                if count == occurrence:
                    match = c
                    break
        if match is None:
            wbox = self._word_box(image, anchor_phrase, anchor_bbox)
            return {"bbox": wbox, "point": _center(wbox)}
        bbox = _bbox_of(match)
        return {"bbox": bbox, "point": _center(bbox)}

    def _resolve_sentence_end(self, image, anchor_phrase, anchor_bbox, params):
        crop, offset = _crop_with_margin(image, anchor_bbox, margin=CROP_MARGIN_PX * 3)
        chars = _char_boxes(_prep_for_ocr(crop), offset)
        enders = [c for c in chars if c["ch"] in _SENTENCE_ENDERS]
        if not enders:
            wbox = self._word_box(image, anchor_phrase, anchor_bbox)
            return {"bbox": wbox, "point": _center(wbox)}
        target = min(enders, key=lambda c: abs(c["x0"] - anchor_bbox[2]))
        bbox = _bbox_of(target)
        return {"bbox": bbox, "point": _center(bbox)}

    # -- line/paragraph aggregation (image_to_data, word/line/par grouping) -

    def _resolve_line_bbox(self, image, anchor_phrase, anchor_bbox, params):
        x0, x1 = 0, image.width
        y0 = max(0, anchor_bbox[1] - LINE_CROP_MARGIN_PX)
        y1 = min(image.height, anchor_bbox[3] + LINE_CROP_MARGIN_PX)
        crop = image.crop((x0, y0, x1, y1))
        data = pytesseract.image_to_data(_prep_for_ocr(crop), output_type=pytesseract.Output.DICT)
        words = _words_from_data(data, (x0, y0))
        anchor_word = _best_overlap(words, anchor_bbox)
        if anchor_word is None:
            return {"bbox": list(anchor_bbox), "point": _center(anchor_bbox)}
        line_words = [
            w for w in words
            if w["line_num"] == anchor_word["line_num"] and w["block_num"] == anchor_word["block_num"]
        ]
        bbox = [
            round(min(w["x0"] for w in line_words)), round(min(w["y0"] for w in line_words)),
            round(max(w["x1"] for w in line_words)), round(max(w["y1"] for w in line_words)),
        ]
        return {"bbox": bbox, "point": _center(bbox)}

    def _resolve_paragraph_bbox(self, image, anchor_phrase, anchor_bbox, params):
        x0, x1 = 0, image.width
        y0 = max(0, anchor_bbox[1] - PARAGRAPH_CROP_MARGIN_PX)
        y1 = min(image.height, anchor_bbox[3] + PARAGRAPH_CROP_MARGIN_PX)
        crop = image.crop((x0, y0, x1, y1))
        data = pytesseract.image_to_data(_prep_for_ocr(crop), output_type=pytesseract.Output.DICT)
        words = _words_from_data(data, (x0, y0))
        anchor_word = _best_overlap(words, anchor_bbox)
        if anchor_word is None:
            return {"bbox": list(anchor_bbox), "point": _center(anchor_bbox)}
        para_words = [
            w for w in words
            if w["par_num"] == anchor_word["par_num"] and w["block_num"] == anchor_word["block_num"]
        ]
        bbox = [
            round(min(w["x0"] for w in para_words)), round(min(w["y0"] for w in para_words)),
            round(max(w["x1"] for w in para_words)), round(max(w["y1"] for w in para_words)),
        ]
        return {"bbox": bbox, "point": _center(bbox)}

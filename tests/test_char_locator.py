"""Sprint 4's required correctness proof (see progress.md section 4b): the
char-locator path crops to a VLM-grounded anchor bbox, upscales, and runs
Tesseract - whose own coordinate systems are crop-local and (for
image_to_boxes) y-flipped. If either transform-back step is wrong, this
fails silently (everything runs, boxes just land in a plausible-looking
wrong place). This test renders real text at a KNOWN, non-zero offset in a
larger canvas and asserts TesseractCharLocator returns coordinates in that
same full-image frame - not crop-local, not y-flipped, not still scaled up.

Requires the system tesseract-ocr binary (see README install instructions).
"""
import pytest
from PIL import Image, ImageDraw, ImageFont

from orchestrator.char_locator import TesseractCharLocator

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
CANVAS_SIZE = (400, 300)
TEXT = "HELLO"
TEXT_ORIGIN = (150, 100)  # deliberately non-zero, non-trivial offset
FONT_SIZE = 40


def _make_test_image():
    """A white canvas with black text at a known location, plus a known
    exact text bbox (from PIL's own text measurement, not OCR) to compare
    the locator's OCR-derived answer against."""
    image = Image.new("RGB", CANVAS_SIZE, "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    draw.text(TEXT_ORIGIN, TEXT, fill="black", font=font)
    true_bbox = draw.textbbox(TEXT_ORIGIN, TEXT, font=font)
    return image, list(true_bbox)


@pytest.fixture(scope="module")
def locator():
    return TesseractCharLocator()


@pytest.fixture(scope="module")
def image_and_true_bbox():
    return _make_test_image()


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def test_word_box_lands_in_full_image_frame(locator, image_and_true_bbox):
    """A slightly-loose 'anchor bbox' (as a VLM's own box regression would
    give) around the known text, fed through the crop -> Tesseract ->
    transform-back path, must recover a bbox that overlaps the TRUE
    full-image text location - not the origin, not the crop's own local
    frame, not a y-flipped mirror image of the true location."""
    image, true_bbox = image_and_true_bbox
    # A loose anchor - not pixel-exact, like a real grounder's output.
    anchor_bbox = [true_bbox[0] - 6, true_bbox[1] - 6, true_bbox[2] + 6, true_bbox[3] + 6]

    result_bbox = locator._word_box(image, TEXT, anchor_bbox)

    # Must be within the image bounds (a crop-local or unflipped-y result
    # would very plausibly land outside these bounds, or inside them but at
    # the wrong place - the overlap check below is the real assertion).
    assert 0 <= result_bbox[0] < image.width
    assert 0 <= result_bbox[1] < image.height

    assert _iou(result_bbox, true_bbox) > 0.5, (
        f"resolved bbox {result_bbox} does not overlap the true full-image "
        f"text location {result_bbox} - a crop-local or y-flipped result "
        f"would fail exactly this way while still 'running successfully'."
    )

    # A wrong-but-plausible failure mode: returning the crop-local box
    # unmodified (i.e. as if crop_offset were (0, 0)). Explicitly rule it out.
    crop_local_x0 = result_bbox[0] - anchor_bbox[0]
    assert not (
        abs(crop_local_x0) < 3 and abs(result_bbox[0]) > 20
    ), "bbox looks like it was never offset back from crop-local coordinates"


def test_before_word_point_is_left_of_text_in_full_image_frame(locator, image_and_true_bbox):
    """resolve() for the "before_word" relation must place its point just
    left of the anchor's TRUE full-image x-position, not at a crop-local
    x-position (which, given the crop starts near the text's own left edge,
    would be close to 0 - a very different, silently-wrong answer)."""
    image, true_bbox = image_and_true_bbox
    anchor_bbox = [true_bbox[0] - 6, true_bbox[1] - 6, true_bbox[2] + 6, true_bbox[3] + 6]

    result = locator.resolve(image, TEXT, anchor_bbox, "before_word", {}, "point")
    px, py = result["point"]

    assert TEXT_ORIGIN[0] - 15 <= px <= true_bbox[0] + 5
    assert true_bbox[1] - 5 <= py <= true_bbox[3] + 5

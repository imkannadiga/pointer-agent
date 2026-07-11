"""Jinja2 template rendering + Playwright screenshot / DOM bbox extraction.

Ground truth boxes are read directly from the rendered DOM via
getBoundingClientRect() on every tagged span - no OCR, no heuristics.

Each surface has multiple template variants (templates/<surface>/*.html.j2),
each a distinct real-world-looking design (layout, chrome, fonts, colors) so
the synthetic set doesn't collapse to one visual style per surface. The
variant is sampled per row in generate_dataset.py; all variants of a surface
consume the same content structure from content_fillers.py.
"""
import os
from functools import lru_cache

from jinja2 import Environment, FileSystemLoader
from playwright.sync_api import sync_playwright

VIEWPORT = {"width": 1024, "height": 768}
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")

_env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))


@lru_cache(maxsize=None)
def list_variants(surface: str) -> tuple[str, ...]:
    surface_dir = os.path.join(TEMPLATES_DIR, surface)
    if not os.path.isdir(surface_dir):
        raise ValueError(f"No template directory for surface: {surface}")
    variants = tuple(sorted(
        name[: -len(".html.j2")]
        for name in os.listdir(surface_dir)
        if name.endswith(".html.j2")
    ))
    if not variants:
        raise ValueError(f"No template variants found for surface: {surface}")
    return variants


def render_html(
    surface: str,
    content: dict,
    theme: str,
    font_size: int = 16,
    occlusion_box: dict | None = None,
    variant: str | None = None,
) -> str:
    if variant is None:
        variant = list_variants(surface)[0]
    template = _env.get_template(f"{surface}/{variant}.html.j2")
    return template.render(
        content=content, theme=theme, font_size=font_size, occlusion_box=occlusion_box
    )


class Renderer:
    """Keeps a single browser instance alive across many render() calls."""

    def __enter__(self):
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch()
        self._page = self._browser.new_page(viewport=VIEWPORT)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._page.close()
        self._browser.close()
        self._playwright.stop()

    def render(self, html: str, screenshot_path: str) -> list[dict]:
        """Render html, save a screenshot, return extracted word/token boxes."""
        self._page.set_content(html, wait_until="load")
        self._page.screenshot(path=screenshot_path)
        boxes = self._page.eval_on_selector_all(
            "[data-target-id]",
            """els => els.map(e => {
                const r = e.getBoundingClientRect();
                // Hit-test the box center: template variants have inner
                // overflow:hidden containers (IDE panes, phone frames, chat
                // widgets) whose clipped content keeps an in-viewport rect
                // while being invisible - the viewport bounds check alone
                // can't catch that. elementFromPoint skips pointer-events:
                // none elements, so the deliberate occlusion overlay does
                // not swallow the boxes underneath it.
                const hit = document.elementFromPoint(
                    r.left + r.width / 2, r.top + r.height / 2
                );
                const visible = hit !== null
                    && (e === hit || e.contains(hit) || hit.contains(e));
                return {
                    id: e.getAttribute('data-target-id'),
                    text: e.getAttribute('data-text'),
                    kind: e.getAttribute('data-kind'),
                    field: e.getAttribute('data-field'),
                    x0: r.left, y0: r.top, x1: r.right, y1: r.bottom,
                    visible: visible,
                };
            })""",
        )
        # Drop off-viewport / zero-area / center-occluded boxes. Containers
        # (paragraph/structural/field/message/field-row) can be
        # whitespace-only or lack literal text, so the text-emptiness check
        # is word/char only.
        clean = []
        for b in boxes:
            if b["x1"] <= b["x0"] or b["y1"] <= b["y0"]:
                continue
            if b["kind"] in ("word", "char") and (not b["text"] or not b["text"].strip()):
                continue
            if b["x0"] < 0 or b["y0"] < 0 or b["x1"] > VIEWPORT["width"] or b["y1"] > VIEWPORT["height"]:
                continue
            if not b.pop("visible"):
                continue
            clean.append(b)
        return clean

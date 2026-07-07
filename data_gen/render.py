"""Jinja2 template rendering + Playwright screenshot / DOM bbox extraction.

Ground truth boxes are read directly from the rendered DOM via
getBoundingClientRect() on every tagged span - no OCR, no heuristics.
"""
import os

from jinja2 import Environment, FileSystemLoader
from playwright.sync_api import sync_playwright

VIEWPORT = {"width": 1024, "height": 768}
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")

_env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))


def render_html(
    surface: str,
    content: dict,
    theme: str,
    font_size: int = 16,
    occlusion_box: dict | None = None,
) -> str:
    template = _env.get_template(f"{surface}.html.j2")
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
                return {
                    id: e.getAttribute('data-target-id'),
                    text: e.getAttribute('data-text'),
                    kind: e.getAttribute('data-kind'),
                    field: e.getAttribute('data-field'),
                    x0: r.left, y0: r.top, x1: r.right, y1: r.bottom,
                };
            })""",
        )
        # Drop off-viewport / zero-area boxes. Containers (paragraph/
        # structural/field/message/field-row) can be whitespace-only or
        # lack literal text, so the text-emptiness check is word/char only.
        clean = []
        for b in boxes:
            if b["x1"] <= b["x0"] or b["y1"] <= b["y0"]:
                continue
            if b["kind"] in ("word", "char") and (not b["text"] or not b["text"].strip()):
                continue
            if b["x0"] < 0 or b["y0"] < 0 or b["x1"] > VIEWPORT["width"] or b["y1"] > VIEWPORT["height"]:
                continue
            clean.append(b)
        return clean

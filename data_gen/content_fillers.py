"""Faker-driven content generation for each surface template.

Content is first built as plain nested strings/lists using Faker, then walked
to tag every word/token with a unique ``data-target-id`` so templates can wrap
each one in a span. Ground-truth boxes are recovered post-render by querying
those ids (see render.py) rather than by any text-matching heuristic.
"""
import itertools

from faker import Faker

LOCALE_MAP = {"en": "en_US", "de": "de_DE"}

CODE_KEYWORDS = [
    "def", "return", "if", "else", "elif", "for", "while", "import",
    "class", "print", "True", "False", "None", "try", "except", "with",
]

STATUSES = ["online", "offline", "maintenance", "degraded"]

SERVER_HEADERS = ["Hostname", "IP Address", "Status", "CPU", "Memory", "Location"]


def _get_faker(language: str) -> Faker:
    return Faker(LOCALE_MAP[language])


def _tag_words(obj, counter):
    """Recursively replace every string leaf with a {id, text} dict."""
    if isinstance(obj, str):
        return {"id": f"w{next(counter):05d}", "text": obj}
    if isinstance(obj, list):
        return [_tag_words(x, counter) for x in obj]
    if isinstance(obj, dict):
        return {k: _tag_words(v, counter) for k, v in obj.items()}
    return obj


def _raw_document(fake: Faker, rng):
    title = fake.sentence(nb_words=4).rstrip(".").split()
    paragraphs = []
    for _ in range(rng.randint(3, 5)):
        text = fake.paragraph(nb_sentences=rng.randint(3, 5))
        paragraphs.append(text.split())
    return {"title": title, "paragraphs": paragraphs}


def _raw_code_editor(fake: Faker, rng):
    lines = []
    for _ in range(rng.randint(15, 25)):
        n_tokens = rng.randint(3, 8)
        tokens = []
        for _ in range(n_tokens):
            tokens.append(
                rng.choice([fake.word(), rng.choice(CODE_KEYWORDS), str(rng.randint(0, 100))])
            )
        indent = rng.choice([0, 0, 1, 1, 2])
        lines.append({"indent": indent, "tokens": tokens})
    return {"lines": lines}


def _raw_server_inventory(fake: Faker, rng):
    title = fake.sentence(nb_words=3).rstrip(".").split()
    headers = [h.split() for h in SERVER_HEADERS]
    rows = []
    for _ in range(rng.randint(8, 14)):
        cells = [
            fake.hostname(),
            fake.ipv4_private(),
            rng.choice(STATUSES),
            f"{rng.randint(1, 64)} cores",
            f"{rng.randint(1, 256)} GB",
            fake.city(),
        ]
        rows.append([cell.split() for cell in cells])
    return {"title": title, "headers": headers, "rows": rows}


_RAW_BUILDERS = {
    "document": _raw_document,
    "code_editor": _raw_code_editor,
    "server_inventory": _raw_server_inventory,
}


def generate_content(surface: str, language: str, rng) -> dict:
    """Build tagged content for a given surface/language, ready for templating."""
    if surface not in _RAW_BUILDERS:
        raise ValueError(f"Unknown surface: {surface}")
    fake = _get_faker(language)
    raw = _RAW_BUILDERS[surface](fake, rng)
    counter = itertools.count()
    return _tag_words(raw, counter)

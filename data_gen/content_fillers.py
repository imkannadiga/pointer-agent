"""Faker-driven content generation for each surface template.

Content is first built as plain nested strings/lists using Faker, then walked
to tag every word/token with a unique ``data-target-id`` so templates can wrap
each one in a span. Ground-truth boxes are recovered post-render by querying
those ids (see render.py) rather than by any text-matching heuristic.

Tagged nodes carry a ``kind`` (word / char / paragraph / structural / field)
that render.py reads back from a ``data-kind`` attribute, so task_builder.py
can select targets by kind instead of guessing from shape.
"""
import itertools

from faker import Faker

LOCALE_MAP = {
    "en": "en_US",
    "de": "de_DE",
    "fr": "fr_FR",
    "es": "es_ES",
    "it": "it_IT",
    "nl": "nl_NL",
}

CODE_KEYWORDS = [
    "def", "return", "if", "else", "elif", "for", "while", "import",
    "class", "print", "True", "False", "None", "try", "except", "with",
]

STATUSES = ["online", "offline", "maintenance", "degraded"]
STATUSES_BIASED = ["online", "offline"]  # smaller pool -> more natural duplicates

SERVER_HEADERS = ["Hostname", "IP Address", "Status", "CPU", "Memory", "Location"]

# Real pointerbench-text invoice field categories (progress.md section 1.1).
INVOICE_FIELD_KEYS = [
    "invoice_number", "invoice_date", "invoice_due_date", "invoice_subtotal",
    "invoice_tax", "invoice_tax_id", "invoice_total", "invoice_totals",
    "invoice_sender", "invoice_sender_city", "invoice_sender_zip",
    "invoice_recipient", "invoice_recipient_city", "invoice_recipient_zip",
    "invoice_bic", "invoice_iban", "invoice_bank_details", "invoice_email",
    "invoice_phone",
]


def _get_faker(language: str) -> Faker:
    return Faker(LOCALE_MAP[language])


def _tag_word(word_id: str, text: str, with_chars: bool) -> dict:
    tagged = {"id": word_id, "text": text, "kind": "word"}
    if with_chars:
        char_counter = itertools.count()
        tagged["chars"] = [
            {"id": f"{word_id}c{next(char_counter):03d}", "ch": ch, "kind": "char"}
            for ch in text
        ]
    return tagged


def _tag_words(words: list[str], counter, with_chars: bool = False) -> list[dict]:
    """Tag a flat list of plain word strings."""
    return [_tag_word(f"w{next(counter):05d}", w, with_chars) for w in words]


def _tag_paragraph(text: str, counter, para_counter, with_chars: bool = True) -> dict:
    return {
        "id": f"p{next(para_counter):05d}",
        "kind": "paragraph",
        "words": _tag_words(text.split(), counter, with_chars=with_chars),
    }


def _tag_structural(words: list[str], counter, struct_counter) -> dict:
    """A container (e.g. a title/headline) selectable as a whole, plus its
    individual tagged words for word-level tasks."""
    return {
        "id": f"s{next(struct_counter):05d}",
        "kind": "structural",
        "words": _tag_words(words, counter, with_chars=False),
    }


def _tag_field(key: str, text: str, counter) -> dict:
    return {"id": f"f{next(counter):05d}", "kind": "field", "field": key, "text": text}


def _raw_document(fake: Faker, rng, counter):
    struct_counter = itertools.count()
    para_counter = itertools.count()
    title_words = fake.sentence(nb_words=4).rstrip(".").split()
    title = _tag_structural(title_words, counter, struct_counter)
    paragraphs = []
    for _ in range(rng.randint(3, 5)):
        text = fake.paragraph(nb_sentences=rng.randint(3, 5))
        paragraphs.append(_tag_paragraph(text, counter, para_counter, with_chars=True))
    return {"title": title, "paragraphs": paragraphs}


def _raw_magazine_spread(fake: Faker, rng, counter):
    struct_counter = itertools.count()
    para_counter = itertools.count()
    headline_words = fake.sentence(nb_words=6).rstrip(".").split()
    headline = _tag_structural(headline_words, counter, struct_counter)
    n_columns = rng.randint(2, 3)
    columns = []
    for _ in range(n_columns):
        paragraphs = []
        for _ in range(rng.randint(2, 4)):
            text = fake.paragraph(nb_sentences=rng.randint(2, 4))
            paragraphs.append(_tag_paragraph(text, counter, para_counter, with_chars=True))
        columns.append({"paragraphs": paragraphs})
    return {"headline": headline, "columns": columns}


def _raw_code_editor(fake: Faker, rng, counter):
    lines = []
    for _ in range(rng.randint(15, 25)):
        n_tokens = rng.randint(3, 8)
        tokens = []
        for _ in range(n_tokens):
            tokens.append(
                rng.choice([fake.word(), rng.choice(CODE_KEYWORDS), str(rng.randint(0, 100))])
            )
        indent = rng.choice([0, 0, 1, 1, 2])
        lines.append({"indent": indent, "tokens": _tag_words(tokens, counter)})
    return {"lines": lines}


def _raw_server_inventory(fake: Faker, rng, counter, enable_distractors: bool):
    struct_counter = itertools.count()
    title = _tag_structural(fake.sentence(nb_words=3).rstrip(".").split(), counter, struct_counter)
    headers = [_tag_words(h.split(), counter) for h in SERVER_HEADERS]
    status_pool = STATUSES_BIASED if enable_distractors else STATUSES
    rows = []
    for _ in range(rng.randint(8, 14)):
        cells = [
            fake.hostname(),
            fake.ipv4_private(),
            rng.choice(status_pool),
            f"{rng.randint(1, 64)} cores",
            f"{rng.randint(1, 256)} GB",
            fake.city(),
        ]
        rows.append([_tag_words(cell.split(), counter) for cell in cells])
    return {"title": title, "headers": headers, "rows": rows}


def _raw_chat_ui(fake: Faker, rng, counter):
    msg_counter = itertools.count()
    n_senders = rng.randint(2, 3)
    senders = [fake.first_name() for _ in range(n_senders)]
    messages = []
    for _ in range(rng.randint(8, 16)):
        sender = rng.choice(senders)
        text = fake.sentence(nb_words=rng.randint(4, 12)).rstrip(".")
        messages.append({
            "id": f"m{next(msg_counter):05d}",
            "sender": _tag_words(sender.split(), counter),
            "text": _tag_words(text.split(), counter),
        })
    return {"messages": messages}


def _raw_form(fake: Faker, rng, counter):
    struct_counter = itertools.count()
    title = _tag_structural(fake.sentence(nb_words=3).rstrip(".").split(), counter, struct_counter)
    field_specs = [
        ("Full Name", fake.name()),
        ("Email Address", fake.email()),
        ("Phone Number", fake.phone_number()),
        ("City", fake.city()),
        ("Postal Code", fake.postcode()),
        ("Company", fake.company()),
        ("Job Title", fake.job()),
        ("Country", fake.country()),
    ]
    n_fields = rng.randint(4, len(field_specs))
    chosen = rng.sample(field_specs, n_fields)
    fields = []
    field_counter = itertools.count()
    for label, value in chosen:
        fields.append({
            "id": f"fld{next(field_counter):05d}",
            "label": _tag_words(label.split(), counter),
            "value": _tag_words(str(value).split(), counter),
        })
    return {"title": title, "fields": fields}


def _raw_invoice(fake: Faker, rng, counter, enable_distractors: bool):
    values = {}
    subtotal = round(rng.uniform(100, 5000), 2)
    tax = round(subtotal * 0.19, 2)
    total = round(subtotal + tax, 2)
    values["invoice_number"] = f"INV-{rng.randint(10000, 99999)}"
    values["invoice_date"] = fake.date_this_year().strftime("%Y-%m-%d")
    values["invoice_due_date"] = fake.date_this_year().strftime("%Y-%m-%d")
    values["invoice_subtotal"] = f"{subtotal:.2f}"
    values["invoice_tax"] = f"{tax:.2f}"
    values["invoice_tax_id"] = fake.bothify(text="??########").upper()
    values["invoice_total"] = f"{total:.2f}"
    values["invoice_totals"] = f"Total due: {total:.2f}"
    values["invoice_sender"] = fake.company()
    values["invoice_sender_city"] = fake.city()
    values["invoice_sender_zip"] = fake.postcode()
    values["invoice_recipient"] = fake.name()
    values["invoice_recipient_city"] = fake.city()
    values["invoice_recipient_zip"] = fake.postcode()
    values["invoice_bic"] = fake.swift()
    values["invoice_iban"] = fake.iban()
    values["invoice_bank_details"] = f"{fake.bank_country()} - {fake.iban()}"
    values["invoice_email"] = fake.company_email()
    values["invoice_phone"] = fake.phone_number()

    fields = {key: _tag_field(key, values[key], counter) for key in INVOICE_FIELD_KEYS}

    n_items = rng.randint(2, 5)
    item_texts = []
    for _ in range(n_items):
        desc = fake.bs()
        qty = rng.randint(1, 10)
        price = round(rng.uniform(10, 500), 2)
        item_texts.append(f"{desc} x{qty} @ {price:.2f}")

    line_items = [_tag_field("invoice_line_item", text, counter) for text in item_texts]
    line_items_container = _tag_field("invoice_line_items", " | ".join(item_texts), counter)

    return {"fields": fields, "line_items": line_items, "line_items_container": line_items_container}


_RAW_BUILDERS = {
    "document": lambda fake, rng, counter, enable_distractors: _raw_document(fake, rng, counter),
    "code_editor": lambda fake, rng, counter, enable_distractors: _raw_code_editor(fake, rng, counter),
    "server_inventory": lambda fake, rng, counter, enable_distractors: _raw_server_inventory(
        fake, rng, counter, enable_distractors
    ),
    "magazine_spread": lambda fake, rng, counter, enable_distractors: _raw_magazine_spread(fake, rng, counter),
    "chat_ui": lambda fake, rng, counter, enable_distractors: _raw_chat_ui(fake, rng, counter),
    "form": lambda fake, rng, counter, enable_distractors: _raw_form(fake, rng, counter),
    "invoice": lambda fake, rng, counter, enable_distractors: _raw_invoice(fake, rng, counter, enable_distractors),
}

# Surfaces that support char-level / caret / paragraph / sentence /
# structural tasks (i.e. carry real per-character DOM spans). Kept narrow
# to bound DOM size/render time, per Sprint 3 plan.
PROSE_SURFACES = ("document", "magazine_spread")


def generate_content(surface: str, language: str, rng, enable_distractors: bool = False) -> dict:
    """Build tagged content for a given surface/language, ready for templating."""
    if surface not in _RAW_BUILDERS:
        raise ValueError(f"Unknown surface: {surface}")
    fake = _get_faker(language)
    counter = itertools.count()
    return _RAW_BUILDERS[surface](fake, rng, counter, enable_distractors)

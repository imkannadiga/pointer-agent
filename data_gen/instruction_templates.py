"""Natural-sounding phrasing variants per task category, en + de.

Several variants per category so the same category doesn't read as one rigid
template with a variable swapped in (matching the conversational tone of the
real pointerbench-text instructions).
"""

TEMPLATES = {
    "word_center": {
        "en": [
            'Click on the word "{text}".',
            'Where is the word "{text}" located?',
            'Point to "{text}".',
            'Can you tap the word "{text}"?',
            'Select "{text}" on the page.',
            'I need you to find "{text}".',
        ],
        "de": [
            'Klicke auf das Wort "{text}".',
            'Wo befindet sich das Wort "{text}"?',
            'Zeige auf "{text}".',
            'Kannst du "{text}" antippen?',
            'Wähle "{text}" auf der Seite aus.',
            'Finde bitte "{text}".',
        ],
    },
    "word_bbox": {
        "en": [
            'Give me the bounding box of the word "{text}".',
            'Highlight the region containing "{text}".',
            'What are the boundaries of "{text}"?',
            'Draw a box around "{text}".',
            'Select the full extent of "{text}".',
        ],
        "de": [
            'Gib mir die Begrenzung des Wortes "{text}".',
            'Markiere den Bereich, der "{text}" enthält.',
            'Wie sind die Grenzen von "{text}"?',
            'Zeichne einen Rahmen um "{text}".',
            'Wähle den gesamten Bereich von "{text}" aus.',
        ],
    },
    "line_start": {
        "en": [
            'Where does the line starting with "{text}" begin?',
            'Point to the start of the line beginning with "{text}".',
            'Can you find the left edge of the line that starts with "{text}"?',
            'Click at the beginning of the line starting "{text}".',
            'Show me where the line with "{text}" starts.',
        ],
        "de": [
            'Wo beginnt die Zeile, die mit "{text}" anfängt?',
            'Zeige auf den Anfang der Zeile, die mit "{text}" beginnt.',
            'Kannst du den linken Rand der Zeile finden, die mit "{text}" beginnt?',
            'Klicke auf den Anfang der Zeile, die mit "{text}" beginnt.',
            'Zeig mir, wo die Zeile mit "{text}" beginnt.',
        ],
    },
    "line_end": {
        "en": [
            'Where does the line ending with "{text}" finish?',
            'Point to the end of the line that ends with "{text}".',
            'Can you find the right edge of the line ending in "{text}"?',
            'Click at the end of the line that finishes with "{text}".',
            'Show me where the line with "{text}" ends.',
        ],
        "de": [
            'Wo endet die Zeile, die mit "{text}" aufhört?',
            'Zeige auf das Ende der Zeile, die mit "{text}" endet.',
            'Kannst du den rechten Rand der Zeile finden, die mit "{text}" endet?',
            'Klicke auf das Ende der Zeile, die mit "{text}" endet.',
            'Zeig mir, wo die Zeile mit "{text}" endet.',
        ],
    },
}


def phrase(category: str, language: str, rng, **kwargs) -> str:
    options = TEMPLATES[category][language]
    tmpl = rng.choice(options)
    return tmpl.format(**kwargs)

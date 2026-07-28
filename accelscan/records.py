"""Parse S2ORC v2 full-text records into paragraphs with section context.

Record layout (s2orc_v2 JSONL): `body.text` holds the full body; the
bibliography is a separate top-level `bibliography` object and is never read,
so reference lists are excluded by construction. `body.annotations.paragraph`
and `.section_header` are JSON-encoded strings (a second json.loads) of
`[{start, end, attributes}, ...]` char offsets into `body.text`.
"""

from dataclasses import dataclass

import orjson

MIN_PARA_CHARS = 30


@dataclass
class Paragraph:
    idx: int
    start: int
    end: int
    text: str
    section: str | None


def _load_spans(annotations: dict, key: str) -> list[tuple[int, int]]:
    raw = annotations.get(key)
    if not raw or not isinstance(raw, str):
        return []
    try:
        spans = orjson.loads(raw)
    except Exception:
        return []
    out = []
    for s in spans:
        start, end = s.get('start'), s.get('end')
        if isinstance(start, int) and isinstance(end, int) and end > start:
            out.append((start, end))
    return sorted(out)


def parse_body(record: dict) -> list[Paragraph]:
    """Return body paragraphs in document order, each with the nearest
    preceding section-header text (None before the first header)."""
    body = record.get('body')
    if not isinstance(body, dict):
        return []
    text = body.get('text')
    if not text or not isinstance(text, str):
        return []

    annotations = body.get('annotations') or {}
    para_spans = _load_spans(annotations, 'paragraph')
    header_spans = _load_spans(annotations, 'section_header')

    if not para_spans:
        # Some records ship null annotations; fall back to blank-line splitting.
        para_spans = []
        pos = 0
        for chunk in text.split('\n\n'):
            if chunk.strip():
                para_spans.append((pos, pos + len(chunk)))
            pos += len(chunk) + 2

    headers = [(start, text[start:end].strip()) for start, end in header_spans]

    out = []
    h_i = 0
    section = None
    for idx, (start, end) in enumerate(para_spans):
        while h_i < len(headers) and headers[h_i][0] <= start:
            section = headers[h_i][1]
            h_i += 1
        para_text = text[start:end].strip()
        if len(para_text) < MIN_PARA_CHARS:
            continue
        out.append(Paragraph(idx=idx, start=start, end=end, text=para_text, section=section))
    return out

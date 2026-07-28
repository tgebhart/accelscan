import orjson

from accelscan.records import parse_body


def make_record(text, paragraphs=None, headers=None):
    ann = {}
    if paragraphs is not None:
        ann['paragraph'] = orjson.dumps(
            [{'start': s, 'end': e, 'attributes': None} for s, e in paragraphs]).decode()
    if headers is not None:
        ann['section_header'] = orjson.dumps(
            [{'start': s, 'end': e, 'attributes': None} for s, e in headers]).decode()
    return {'corpusid': 1, 'body': {'text': text, 'annotations': ann},
            'bibliography': {'text': 'REFS SHOULD NEVER APPEAR', 'annotations': {}}}


def test_paragraphs_and_sections():
    text = 'Methods\n' + 'We trained a model on GPUs for many epochs today.\n' \
           + 'Results\n' + 'The model performed well on the benchmark suite.'
    p1 = (8, 58)
    h1 = (0, 7)
    h2 = (58, 65)
    p2 = (66, len(text))
    rec = make_record(text, paragraphs=[p1, p2], headers=[h1, h2])
    paras = parse_body(rec)
    assert len(paras) == 2
    assert paras[0].section == 'Methods'
    assert paras[1].section == 'Results'
    assert 'REFS' not in ' '.join(p.text for p in paras)


def test_fallback_split_when_annotations_missing():
    text = ('First paragraph with enough characters to survive filtering.\n\n'
            'Second paragraph, also long enough to be kept by the parser.')
    rec = {'corpusid': 2, 'body': {'text': text, 'annotations': {'paragraph': None}}}
    paras = parse_body(rec)
    assert len(paras) == 2
    assert paras[0].section is None


def test_short_paragraphs_dropped():
    rec = make_record('tiny\n' + 'x' * 60, paragraphs=[(0, 4), (5, 65)])
    paras = parse_body(rec)
    assert len(paras) == 1


def test_missing_body():
    assert parse_body({'corpusid': 3}) == []
    assert parse_body({'corpusid': 4, 'body': {'text': ''}}) == []

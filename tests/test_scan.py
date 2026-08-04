from pathlib import Path

import pytest

from accelscan.config import MAX_PASSAGES_GENERIC_ONLY
from accelscan.records import Paragraph
from accelscan.registry import load_registry
from accelscan.scan import scan_paragraphs, scan_record

ROOT = Path(__file__).resolve().parents[1]
FILLER = 'This filler sentence pads the paragraph to a plausible length.'


@pytest.fixture(scope='session')
def reg():
    return load_registry(ROOT / 'registry')


def record_from_paragraphs(paragraphs):
    return {'corpusid': 42, 'body': {'text': '\n\n'.join(paragraphs),
                                     'annotations': {}}}


def test_non_candidate(reg):
    rec = record_from_paragraphs([f'Nothing about hardware here. {FILLER}'] * 3)
    s = scan_record(rec, reg, 'shard0')
    assert s.inventory['is_candidate'] is False
    assert s.candidates == []


def test_candidate_with_neighbor_context(reg):
    rec = record_from_paragraphs([
        f'Introduction to the broader research problem. {FILLER}',
        f'We trained the network on an NVIDIA V100 GPU for ten hours. {FILLER}',
        f'Evaluation was carried out on held-out data. {FILLER}',
    ])
    s = scan_record(rec, reg, 'shard0')
    assert s.inventory['is_candidate'] is True
    assert len(s.candidates) == 1
    c = s.candidates[0]
    assert 'nvidia-v100' in c['matched_models']
    assert c['model_specific'] is True
    # passage includes neighbors and match offsets point at the surface
    for surf, a, b in zip(c['matched_surfaces'], c['match_starts'], c['match_ends']):
        assert c['passage_text'][a:b] == surf


def test_paper_level_gate_rescue(reg):
    # bare 'A100' with no local GPU context is rescued by a tier-A hit elsewhere
    bare = f'Throughput measured on the A100 was twice as high. {FILLER}'
    rec_alone = record_from_paragraphs([bare])
    assert scan_record(rec_alone, reg, 's').inventory['is_candidate'] is False

    rec_anchored = record_from_paragraphs([
        f'We used an NVIDIA A100 GPU for training runs. {FILLER}',
        f'Unrelated middle paragraph about datasets and metrics. {FILLER}',
        bare,
    ])
    s = scan_record(rec_anchored, reg, 's')
    assert len(s.candidates) == 2
    assert 'nvidia-a100' in s.candidates[1]['matched_models']
    # the rescue's contribution has to be measurable after the fact: the anchor
    # passage stands on its own, the bare one exists only because of the rescue
    assert s.candidates[0]['gate_rescued'] is False
    assert s.candidates[1]['gate_rescued'] is True


def test_generic_only_cap(reg):
    para = f'The computation ran on a GPU cluster for several days. {FILLER}'
    rec = record_from_paragraphs([para] * (MAX_PASSAGES_GENERIC_ONLY + 3))
    s = scan_record(rec, reg, 's')
    assert s.inventory['is_candidate'] is True
    assert len(s.candidates) == MAX_PASSAGES_GENERIC_ONLY
    assert s.inventory['passages_truncated'] is True
    assert all(c['model_specific'] is False for c in s.candidates)


def test_shared_core_is_corpus_agnostic(reg):
    """The matcher must be reached by exactly one code path.

    Same prose, once via the S2ORC record adapter and once via `scan_paragraphs`
    with hand-built Paragraphs (the seam the arXiv LaTeX driver uses). Every
    measurement-bearing field must agree; only the ids may differ. If someone
    forks the match/gate/cap logic for a second corpus, this fails.
    """
    paras = [f'We trained on 8 NVIDIA A100 GPUs for two weeks. {FILLER}',
             f'An unrelated paragraph about the dataset and metrics. {FILLER}',
             f'Inference used a single RTX 3090 card. {FILLER}']
    s2 = scan_record(record_from_paragraphs(paras), reg, 'shard-a')
    ax = scan_paragraphs(
        [Paragraph(idx=i, start=0, end=len(t), text=t, section=None)
         for i, t in enumerate(paras)],
        reg, paper_id='arxiv:2301.01234', corpusid=None, shard_id='shard-a',
        body_chars=len('\n\n'.join(paras)))

    assert len(s2.candidates) == len(ax.candidates) > 0
    measured = ('para_idx', 'passage_text', 'section_header', 'matched_models',
                'matched_surfaces', 'match_starts', 'match_ends', 'gated_only',
                'gate_rescued', 'model_specific')
    for a, b in zip(s2.candidates, ax.candidates):
        assert {k: a[k] for k in measured} == {k: b[k] for k in measured}
    for k in ('is_candidate', 'n_paragraphs', 'n_candidate_passages',
              'passages_truncated'):
        assert s2.inventory[k] == ax.inventory[k]

    # ids: S2ORC derives paper_id from corpusid; arXiv carries a string id only
    assert s2.inventory['paper_id'] == '42' and s2.inventory['corpusid'] == 42
    assert ax.inventory['paper_id'] == 'arxiv:2301.01234'
    assert ax.inventory['corpusid'] is None
    assert ax.candidates[0]['passage_id'].startswith('arxiv:2301.01234:')


def test_match_offsets_index_the_stored_passage(reg):
    """match_starts/ends must index passage_text after neighbour padding."""
    s = scan_record(record_from_paragraphs(
        [f'Baseline paragraph with no hardware. {FILLER}',
         f'We ran the model on an NVIDIA V100 GPU. {FILLER}',
         f'Trailing paragraph for context. {FILLER}']), reg, 's')
    c = s.candidates[0]
    for start, end, surface in zip(c['match_starts'], c['match_ends'],
                                   c['matched_surfaces']):
        assert c['passage_text'][start:end] == surface


# --- the span invariant: a reported offset must index the passage it belongs to ---

def test_match_spans_index_passage_text_when_the_match_is_past_the_cap():
    """Head-truncating the paragraph cut the trigger out of its own passage.

    Found by auditing the finished arXiv run: 0.56% of passages carried offsets
    past the end of `passage_text`, and those passages reached the LLM containing no
    hardware mention at all, so it correctly reported none. The window must follow
    the match.
    """
    from accelscan.config import PASSAGE_CHAR_CAP
    from accelscan.records import Paragraph
    from accelscan.registry import load_registry
    from accelscan.scan import scan_paragraphs

    reg = load_registry()
    filler = 'The derivation proceeds by induction on the number of terms here. '
    tail = 'All timings were collected on a single NVIDIA V100 GPU in our cluster.'
    para = filler * ((PASSAGE_CHAR_CAP // len(filler)) + 10) + tail   # match in the tail
    assert len(para) > PASSAGE_CHAR_CAP
    scan = scan_paragraphs([Paragraph(idx=0, start=0, end=len(para), text=para,
                                      section='Setup')],
                           reg, paper_id='x', corpusid=None, shard_id='s',
                           body_chars=len(para))
    assert scan.candidates, 'the match in the paragraph tail was lost entirely'
    row = scan.candidates[0]
    assert 'NVIDIA V100' in row['passage_text'], 'trigger absent from its own passage'
    for surf, s, e in zip(row['matched_surfaces'], row['match_starts'],
                          row['match_ends']):
        assert row['passage_text'][s:e] == surf, (surf, s, e)
    assert 'nvidia-v100' in row['matched_models']


def test_split_cap_does_not_exceed_passage_cap():
    """Splitting paragraphs longer than a passage guarantees truncated matches."""
    from accelscan.config import PASSAGE_CHAR_CAP, SPLIT_LONG_PARA_CHARS
    assert SPLIT_LONG_PARA_CHARS <= PASSAGE_CHAR_CAP

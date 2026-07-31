r"""arXiv source -> `records.Paragraph` list, by searching the LaTeX **raw**.

No TeX parsing, no dependency, no de-TeXing. Hardware names are ASCII literals that
appear verbatim in source (`NVIDIA V100`, `GPU`, `CUDA`), so the matcher can read the
source directly. Two previous designs did parse -- a hand-rolled de-TeXer that hung
two full-history runs on an unbalanced `\cite{`, then pylatexenc -- and the measured
benefit over raw search was confined to *exclusions*: comments, captions, listings and
tables no longer contributing false hardware mentions. **That trade is accepted
deliberately** (Gebhart, 2026-07-31): the bias is upward and mostly lands in the
generic-mention series, the LLM's `usage_context` already labels a reference list or a
caption as not-used-in-this-work, and no parser can hang a worker.

Consequences to state in the paper, not to hide:

- **Upward bias in any-mention prevalence.** Commented-out text, figure captions,
  spec tables and code listings are searched. A caption reading "Throughput on the
  A100" makes its paper an A100 paper at the regex stage.
- **Comments are read.** `% we used to use a K80` counts as a mention.
- The primary defence downstream is `usage_context`: the LLM sees the raw passage and
  is asked whether the work *used* the hardware. `reported use` and model-specific
  series are therefore far better protected than the raw any-mention series.
- **arXiv is biased upward relative to S2ORC**, which gets these exclusions free
  (GROBID drops floats; `bibliography` is a separate object). Cross-corpus *levels*
  were already non-comparable; this widens the gap in a known direction.
- A cheap robustness column exists later if a reviewer asks: rerun the matcher over
  the stored passages with the exclusions applied. No LLM, no re-download.

The gotchas below are kept because each is one regex and each is not a small effect:

1. **All text members are concatenated**, rather than resolving `\input`. Simpler
   *and* higher recall: arXiv submissions are multi-file and the methods section is
   often its own `.tex`, so an unresolved include loses the hardware paragraph
   outright (measured: 0 candidates vs 1 on the `input-resolved` fixture).
2. **`$8$` becomes `8`.** Numeric inline math is unwrapped, symbolic inline math
   becomes a space. Device counts are a headline estimand and "$8$ V100" must not
   read as "V100" with no count; the space stops `word$x$word` gluing.
3. **Truncate at `\begin{thebibliography}` / `\bibliography{`** past the halfway
   guard (so a preamble `\bibliographystyle` cannot decapitate the paper), plus a
   reference-shaped paragraph filter. `.bbl` members are already excluded by
   `arxiv_source`. Reference *titles* routinely say "GPU-accelerated ...", which
   would inflate the generic series by a large factor rather than a slight one.
4. **Paragraphs are blank-line separated** -- which is TeX's own rule -- with long
   ones re-split, because `PASSAGE_CHAR_CAP` is a character budget.
5. **Section headings are captured** from `\section{...}` and friends, since the
   header goes into the LLM prompt as context.

Returns real `records.Paragraph` objects so `scan.scan_paragraphs` -- the shared
match/gate/cap core -- runs unchanged over both corpora.
"""

import re

from accelscan.config import SPLIT_LONG_PARA_CHARS
from accelscan.records import MIN_PARA_CHARS, Paragraph

MAX_SOURCE_BYTES = 20_000_000
BIB_POSITION_GUARD = 0.5      # bibliography macros before this fraction are ignored

# Sentinel for a section heading: NUL cannot occur in decoded TeX prose.
SEC_OPEN, SEC_CLOSE = '\x00SEC:', '\x00'

TEXT_EXTS = ('.tex', '.ltx', '.txt')

# preamble is configuration not prose; anything after \end{document} is stray
_BODY = re.compile(r'\\begin\s*\{document\}(.*?)(?:\\end\s*\{document\}|\Z)', re.S)
_BIB_STOP = re.compile(r'\\begin\s*\{thebibliography\}|\\bibliography\s*\{'
                       r'|\\printbibliography')
# Inline math only. Display math is left alone: it is rarely prose and never names
# hardware, and skipping it is one less rule.
_INLINE_MATH = re.compile(r'(?<!\\)\$([^$\n]{0,200})(?<!\\)\$|\\\((.{0,200}?)\\\)', re.S)
_NUMERIC_MATH = re.compile(r'^[\d\s.,]+$')
_SECTION = re.compile(r'\\(?:chapter|(?:sub){0,2}section|paragraph)\s*\*?\s*'
                      r'\{([^}\n]{0,200})\}')
_WS = re.compile(r'\s+')
# A reference entry starting its own line becomes its own paragraph, so the
# reference filter cannot take neighbouring prose down with it.
_REF_LINE = re.compile(r'(?m)^[ \t]*(\[\d{1,3}\]|\\bibitem)')
_PARA_SPLIT = re.compile(r'\n\s*\n|\\par\b')

# A paragraph is reference-shaped if it opens like a numbered entry, or is short and
# carries several bibliographic tells.
_REF_OPEN = re.compile(r'^\s*(\[\d{1,3}\]|\\bibitem|\(\d{4}\))')
_REF_SIGNALS = (re.compile(r'\(\d{4}\)|\b(19|20)\d{2}\b'), re.compile(r'\bet al\b'),
                re.compile(r'\bpp\.|\bvol\.|\bno\.\s*\d'), re.compile(r'arXiv:'),
                re.compile(r'\bJ\.\s|\bProc\.|\bIEEE\b|\bACM\b'))


def decode_tex(data: bytes) -> tuple[str, str]:
    """Decode TeX bytes, reporting which encoding won."""
    for enc in ('utf-8', 'cp1252', 'latin-1'):
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return data.decode('utf-8', errors='replace'), 'utf-8-replace'


def assemble_source(files: dict[str, bytes]) -> tuple[str, dict]:
    """Concatenate every text member, longest first.

    No root selection and no `\\input` resolution: for a literal search, document
    order is irrelevant and *every* member is worth searching. Longest first only so
    that the main file's prose leads, which makes `plain_chars` and the paragraph
    indices stable when a submission has stray fragments.
    """
    texts, encodings, found_body = [], [], False
    for name, data in sorted(files.items()):
        if name.lower().endswith(TEXT_EXTS) or '.' not in name.rsplit('/', 1)[-1]:
            text, enc = decode_tex(data[:MAX_SOURCE_BYTES])
            # Body extraction is PER FILE: an included fragment has no
            # \begin{document} and is prose throughout, while doing this on the
            # concatenation would drop every file ordered before the root.
            body = _BODY.search(text)
            if body:
                text = body.group(1)
                found_body = True
            texts.append((len(text), name, text))
            encodings.append(enc)
    texts.sort(key=lambda t: -t[0])
    stats = {'n_files': len(files), 'n_tex_files': len(texts),
             'encoding': encodings[0] if encodings else None,
             'root': texts[0][1] if texts else None, 'body_found': found_body,
             'includes_resolved': max(len(texts) - 1, 0), 'includes_missing': 0}
    return '\n\n'.join(t[2] for t in texts), stats


def _unwrap_inline_math(s: str) -> str:
    """`$8$` -> `8` (device counts survive); `$\\alpha$` -> ' ' (no word gluing)."""
    def repl(m):
        body = (m.group(1) if m.group(1) is not None else m.group(2)) or ''
        return body if _NUMERIC_MATH.match(body) else ' '
    return _INLINE_MATH.sub(repl, s)


def _cut_bibliography(s: str) -> tuple[str, bool]:
    """Truncate at a bibliography macro past the position guard."""
    for m in _BIB_STOP.finditer(s):
        if m.start() >= BIB_POSITION_GUARD * len(s):
            return s[:m.start()], True
    return s, False


def looks_like_reference(p: str) -> bool:
    if _REF_OPEN.match(p):
        return True
    if len(p) < 400 and sum(bool(r.search(p)) for r in _REF_SIGNALS) >= 2:
        return True
    return False


def latex_to_text(src: str) -> tuple[str, dict]:
    """Raw source -> searchable text with section sentinels, plus stats.

    "Searchable" is the whole bar: markup is left in place except where a gotcha
    above says otherwise. Nothing here loops over parser state, so nothing here can
    fail to terminate.
    """
    stats = {'had_bibliography': False, 'macros_expanded': 0,
             'convert_warnings': 0, 'convert_error': None}
    src, cut = _cut_bibliography(src)
    stats['had_bibliography'] = cut
    src = _unwrap_inline_math(src)
    src = _SECTION.sub(lambda m: f'\n\n{SEC_OPEN}{m.group(1).strip()}{SEC_CLOSE}\n\n', src)
    return _REF_LINE.sub(r'\n\n\1', src), stats


def _split_long(text: str, cap: int) -> list[str]:
    """Re-split an over-long paragraph at sentence boundaries.

    TeX authors write very long paragraphs; without this the matched sentence can
    fall outside `PASSAGE_CHAR_CAP` when the passage is assembled.
    """
    if len(text) <= cap:
        return [text]
    out, cur = [], ''
    for sent in re.split(r'(?<=[.!?])\s+', text):
        if cur and len(cur) + len(sent) + 1 > cap:
            out.append(cur)
            cur = sent
        else:
            cur = f'{cur} {sent}'.strip()
    if cur:
        out.append(cur)
    return out


def latex_to_paragraphs(files: dict[str, bytes]) -> tuple[list[Paragraph], dict]:
    """arXiv source members -> paragraphs with section context, plus stats.

    `.bbl` must already be excluded by the caller (see `arxiv_source`).
    """
    src, stats = assemble_source(files)
    if not src:
        return [], {**stats, 'n_paragraphs': 0, 'n_ref_paras_filtered': 0,
                    'plain_chars': 0}
    plain, tstats = latex_to_text(src)
    stats.update(tstats)

    pieces, section, sections = [], None, []
    for chunk in _PARA_SPLIT.split(plain):
        if chunk is None:
            continue
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk.startswith(SEC_OPEN):
            section = chunk[len(SEC_OPEN):].split(SEC_CLOSE)[0].strip() or None
            rest = chunk.split(SEC_CLOSE, 1)[1] if SEC_CLOSE in chunk else ''
            chunk = rest.strip()
            if not chunk:
                continue
        chunk = _WS.sub(' ', chunk.replace(SEC_OPEN, ' ').replace(SEC_CLOSE, ' ')).strip()
        for piece in _split_long(chunk, SPLIT_LONG_PARA_CHARS):
            pieces.append(piece)
            sections.append(section)

    # idx is assigned before the length filter, mirroring records.parse_body so
    # para_idx (and therefore passage_id) means the same thing in both corpora
    out, n_refs = [], 0
    pos = 0
    for idx, (text, sec) in enumerate(zip(pieces, sections)):
        if len(text) < MIN_PARA_CHARS:
            pos += len(text) + 2
            continue
        if looks_like_reference(text):
            n_refs += 1
            pos += len(text) + 2
            continue
        out.append(Paragraph(idx=idx, start=pos, end=pos + len(text), text=text,
                             section=sec))
        pos += len(text) + 2
    stats.update(n_paragraphs=len(out), n_ref_paras_filtered=n_refs,
                 plain_chars=len(plain))
    return out, stats

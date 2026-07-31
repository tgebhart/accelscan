r"""arXiv LaTeX source -> `records.Paragraph` list, parsed by pylatexenc.

**The TeX parsing is not ours.** A hand-rolled de-TeXer lived here and was replaced
after it killed two full-history runs: `_drop_with_args` looped on `_braced`, which
reports an unbalanced group by returning its own start index, so an unclosed `\cite{`
-- ordinary in truncated arXiv source -- spun forever and consumed one worker per bad
paper. 32 fixtures had not caught it. A tokenizer maintained by other people is worth
more than a fixture suite we write ourselves, so parsing now goes through
`pylatexenc.latex2text` and this module supplies only *policy* and plumbing.

Why pylatexenc and not pandoc, measured rather than assumed: pandoc parses the
structure better (a real AST, so floats and captions drop structurally) and is
subprocess-isolated, but it **rejects** broken input with exit 64 -- unbalanced
`\cite{`, unclosed `\begin{align}`, and 1990s `\documentstyle` files with no
`\begin{document}` all fail, and truncate-at-the-reported-error-line does not
converge because pandoc reports where it *notices* (EOF), not where the unclosed
brace is. That failure mode is fatal here for a reason specific to this paper:
rejection correlates with era, so text quality would degrade going back in time and
manufacture exactly the upward trend the study measures. pylatexenc never rejects a
document -- all four broken cases above convert -- which makes it the right engine
even though pandoc is the better parser.

What this module still owns, all of it policy or plumbing with no TeX grammar in it:

- **`\input`/`\include` resolution across tar members.** pylatexenc has no file
  model; arXiv submissions are multi-file and the methods section is often its own
  file, so without this the hardware paragraph is simply absent.
- **Zero-argument macro expansion.** pylatexenc 2 does not expand user macros, and
  drops them: `\newcommand{\ourgpu}{A100}` ... `\ourgpu` becomes nothing, hiding the
  hardware name. Bounded (`limit=2` rounds, self-referential bodies skipped).
- **Bibliography exclusion in three layers.** S2ORC gets this free (`bibliography` is
  a separate object); arXiv must not lose it, or a reference titled "GPU-accelerated
  Monte Carlo..." becomes a false candidate. Layers: `.bbl` files are never read; the
  document is truncated at a bibliography macro past the halfway guard (so a preamble
  `\bibliographystyle` cannot decapitate the paper); any surviving reference-shaped
  paragraph is dropped. `thebibliography` is also in the discard list below.
- **Floats dropped with their captions** (figure/table/algorithm/listing), for
  comparability with S2ORC's GROBID body text, which largely excludes them.
- **Display math dropped, inline math kept as text.** `math_mode='remove'` would
  delete `$8$` from "we used $8$ V100 GPUs" and device counts are a headline
  estimand, so inline math is rendered; display environments are discarded instead.
- **Whitespace collapsed to single spaces.** `GATE_WINDOW_CHARS` (250) and
  `PASSAGE_CHAR_CAP` (2500) are character budgets, and LaTeX line-wrapping would
  otherwise spend 5-10% of them on newlines and indentation.

Returns real `records.Paragraph` objects so `scan.scan_paragraphs` -- the shared
match/gate/cap core -- runs unchanged over both corpora. Every function returns a
stats dict alongside its output; those land in the `ingest_stats` product, the only
way to spot the converter degrading across eras.
"""

import logging
import re
from functools import lru_cache

from pylatexenc.latex2text import (EnvironmentTextSpec, LatexNodes2Text, MacroTextSpec,
                                   get_default_latex_context_db)
from pylatexenc.latexwalker import LatexWalker
from pylatexenc.latexwalker import get_default_latex_context_db as walker_context_db
from pylatexenc.macrospec import std_macro

from accelscan.config import SPLIT_LONG_PARA_CHARS
from accelscan.records import MIN_PARA_CHARS, Paragraph

MAX_SOURCE_BYTES = 20_000_000
MAX_INCLUDE_DEPTH = 8
BIB_POSITION_GUARD = 0.5      # bibliography macros before this fraction are ignored

# Sentinel for a section heading: NUL cannot occur in decoded TeX prose.
SEC_OPEN, SEC_CLOSE = '\x00SEC:', '\x00'

TEXT_EXTS = ('.tex', '.ltx', '.txt')

# Discarded whole. Math, floats, code, verbatim, and the bibliography.
DROP_ENVIRONMENTS = tuple("""
    equation equation* eqnarray eqnarray* align align* alignat alignat* gather
    gather* multline multline* flalign flalign* displaymath math
    array matrix pmatrix bmatrix vmatrix Vmatrix smallmatrix cases split
    figure figure* table table* tabular tabular* tabularx longtable
    wraptable wrapfigure subfigure subtable sidewaysfigure sidewaystable
    algorithm algorithm* algorithmic algorithm2e algorithmicx pseudocode
    lstlisting listing verbatim Verbatim minted alltt code
    tikzpicture pgfpicture picture circuitikz forest
    thebibliography IEEEkeywords keywords
""".split())

# Dropped with their argument: citations, cross-references, notes, graphics.
DROP_WITH_ARG = tuple("""
    cite citep citet citealp citealt citeauthor citeyear citenum nocite Cite
    ref eqref autoref cref Cref pageref label footnote footnotetext thanks
    index glossary marginpar bibliographystyle bibliography printbibliography
    includegraphics graphicspath
""".split())

SECTION_MACROS = ('chapter', 'section', 'subsection', 'subsubsection', 'paragraph',
                  'subparagraph')

_COMMENT = re.compile(r'(?<!\\)((?:\\\\)*)%[^\n]*\n[ \t]*')
_DISPLAY_MATH = re.compile(r'(?<!\\)\$\$.*?(?<!\\)\$\$|\\\[.*?\\\]', re.S)
_BIB_STOP = re.compile(r'\\begin\s*\{thebibliography\}|\\bibliography\s*\{'
                       r'|\\printbibliography|\\bibliographystyle\s*\{')
_WS = re.compile(r'\s+')
# A reference entry that starts its own line becomes its own paragraph, so the
# reference filter cannot take neighbouring prose down with it.
_REF_LINE = re.compile(r'(?m)^[ \t]*(\[\d{1,3}\]|\\bibitem)')
_PARA_SPLIT = re.compile(r'\n\s*\n|\\par\b')

# A paragraph is reference-shaped if it opens like a numbered entry, or is short
# and carries several bibliographic tells. Final net behind the two bib layers.
_REF_OPEN = re.compile(r'^\s*(\[\d{1,3}\]|\\bibitem|\(\d{4}\))')
_REF_SIGNALS = (re.compile(r'\(\d{4}\)|\b(19|20)\d{2}\b'), re.compile(r'\bet al\b'),
                re.compile(r'\bpp\.|\bvol\.|\bno\.\s*\d'), re.compile(r'arXiv:'),
                re.compile(r'\bJ\.\s|\bProc\.|\bIEEE\b|\bACM\b'))


def _section_marker(node, l2tobj=None, **_kw) -> str:
    """Render a sectioning macro as a NUL-delimited heading sentinel.

    `simplify_repl='%s'` cannot be used: `\\section`'s spec carries three arguments
    (star, optional, title) and the substitution fails with a configuration warning.
    """
    args = [a for a in (getattr(node.nodeargd, 'argnlist', None) or []) if a is not None]
    title = ' '.join(l2tobj.nodelist_to_text(args[-1:]).split()) if args and l2tobj else ''
    return f'\n\n{SEC_OPEN}{title}{SEC_CLOSE}\n\n'


class _WarningCounter(logging.Handler):
    """Count pylatexenc's warnings instead of letting them reach the run log.

    pylatexenc logs "macro '\\frac' failed its substitution" whenever one of *its own*
    default specs meets a malformed usage -- `\\frac`, `\\textfrac` and friends were
    all seen within the first tar. Across 3M papers of author-written TeX that is
    unbounded stderr, and a full disk is simply a third way to stall the run. The
    warnings are also not actionable one at a time: the fix is never to chase the
    next macro, it is to know the rate. So the count goes to `ingest_stats`, where
    converter drift across eras is already audited, and the log stays readable.
    """

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.n = 0

    def emit(self, record):
        self.n += 1


@lru_cache(maxsize=1)
def _warning_counter() -> _WarningCounter:
    handler = _WarningCounter()
    log = logging.getLogger('pylatexenc')
    log.handlers = [handler]
    log.propagate = False
    log.setLevel(logging.WARNING)
    return handler


@lru_cache(maxsize=1)
def _parse_context():
    """Walker context with argument counts pylatexenc's default db gets wrong here.

    `\\href` is the important one: latex2text's spec reads two arguments while the
    default *parsing* db gives it fewer, so converting any document containing
    `\\href` raises IndexError inside pylatexenc -- which would have dropped every
    such paper. `\\paragraph`/`\\subparagraph` need `*[{` or their title is not an
    argument and cannot become a section heading.
    """
    db = walker_context_db()
    db.add_context_category('accelscan-parse', prepend=True, macros=[
        std_macro('paragraph', '*[{'), std_macro('subparagraph', '*[{'),
        std_macro('href', '{{'), std_macro('url', '{'), std_macro('nolinkurl', '{'),
    ])
    return db


@lru_cache(maxsize=1)
def _converter() -> LatexNodes2Text:
    """pylatexenc configured to this project's policy. Built once per process."""
    db = get_default_latex_context_db()
    db.add_context_category(
        'accelscan',
        environments=[EnvironmentTextSpec(name, discard=True)
                      for name in DROP_ENVIRONMENTS],
        macros=([MacroTextSpec(name, simplify_repl='') for name in DROP_WITH_ARG]
                + [MacroTextSpec(name, simplify_repl=_section_marker)
                   for name in SECTION_MACROS]),
        prepend=True)
    # math_mode='text' so "$8$ V100" keeps its device count; display math is
    # discarded via DROP_ENVIRONMENTS and _DISPLAY_MATH instead.
    return LatexNodes2Text(latex_context=db, math_mode='text', keep_comments=False,
                           strict_latex_spaces=False)


def _braced(s: str, i: int) -> tuple[str, int]:
    """Content of the brace group starting at s[i]=='{', and the index after it.

    Returns ('', i) when unbalanced -- truncated source is normal on arXiv. Callers
    that loop over this MUST check that the index advanced; not doing so is what
    hung the previous converter. `collect_macros` is now the only caller, and it
    does not loop.
    """
    if i >= len(s) or s[i] != '{':
        return '', i
    depth, j = 0, i
    while j < len(s):
        ch = s[j]
        if ch == '\\':
            j += 2
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    return '', i


def decode_tex(data: bytes) -> tuple[str, str]:
    """Decode TeX bytes, reporting which encoding won."""
    for enc in ('utf-8', 'cp1252', 'latin-1'):
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return data.decode('utf-8', errors='replace'), 'utf-8-replace'


def _pick_root(texts: dict[str, str]) -> str | None:
    """The file that looks like the main document."""
    scored = []
    for name, body in texts.items():
        score = 0
        if re.search(r'\\document(class|style)', body):
            score += 4
        if '\\begin{document}' in body:
            score += 4
        if re.search(r'(?i)\b(main|ms|paper|arxiv|manuscript)\b', name):
            score += 1
        if score:
            scored.append((score, len(body), name))
    if scored:
        return max(scored)[2]
    return max(((len(b), n) for n, b in texts.items()), default=(0, None))[1]


def assemble_source(files: dict[str, bytes]) -> tuple[str, dict]:
    """Root document with \\input/\\include resolved against the other members."""
    texts, encodings = {}, {}
    for name, data in files.items():
        if name.lower().endswith(TEXT_EXTS) or '.' not in name.rsplit('/', 1)[-1]:
            text, enc = decode_tex(data[:MAX_SOURCE_BYTES])
            texts[name] = text
            encodings[name] = enc
    stats = {'n_files': len(files), 'n_tex_files': len(texts),
             'encoding': None, 'includes_resolved': 0, 'includes_missing': 0,
             'root': None}
    if not texts:
        return '', stats

    root = _pick_root(texts)
    stats['root'] = root
    stats['encoding'] = encodings.get(root)

    by_base = {}
    for name in texts:
        for k in (name, name.rsplit('/', 1)[-1]):
            by_base.setdefault(k, name)
            by_base.setdefault(k.removesuffix('.tex'), name)

    include_re = re.compile(r'\\(?:input|include|subfile)\s*\{([^}]{1,200})\}'
                            r'|\\import\s*\{[^}]*\}\s*\{([^}]{1,200})\}')

    def expand(name: str, depth: int, seen: set[str]) -> str:
        body = texts.get(name, '')
        if depth >= MAX_INCLUDE_DEPTH:
            return body

        def sub(m):
            target = (m.group(1) or m.group(2) or '').strip()
            for cand in (target, f'{target}.tex', target.rsplit('/', 1)[-1],
                         f'{target.rsplit("/", 1)[-1]}.tex'):
                hit = by_base.get(cand)
                if hit and hit not in seen:
                    stats['includes_resolved'] += 1
                    return '\n\n' + expand(hit, depth + 1, seen | {hit}) + '\n\n'
            stats['includes_missing'] += 1
            return '\n\n'

        return include_re.sub(sub, body)

    return expand(root, 0, {root}), stats


def _strip_comments(s: str) -> str:
    """TeX comment rule: '%' to EOL *plus* the newline and the next line's indent.

    pylatexenc strips comments itself; this runs first so the pre-passes below
    (macro collection, bibliography truncation) do not read commented-out source --
    a `% \\bibliography{refs}` past the halfway mark would otherwise truncate a
    whole paper.
    """
    return _COMMENT.sub(r'\1', s + '\n')


def collect_macros(src: str) -> dict[str, str]:
    """Zero-argument `\\newcommand`/`\\def` bodies, by macro name.

    Zero-arg only: general expansion is a halting problem on real source, while
    zero-arg covers the common `\\newcommand{\\ourmodel}{FooNet}` case that
    pylatexenc would otherwise drop, leaving holes in the prose. Self-referential
    and over-long bodies are skipped.
    """
    defs = {}
    pat = re.compile(r'\\(?:newcommand|renewcommand|providecommand)\s*\*?\s*'
                     r'\{?\\([a-zA-Z]+)\}?\s*(?=\{)|\\def\s*\\([a-zA-Z]+)\s*(?=\{)')
    for m in pat.finditer(src):
        name = m.group(1) or m.group(2)
        body, _ = _braced(src, m.end())
        if body and len(body) <= 200 and f'\\{name}' not in body:
            defs[name] = body
    return defs


def _apply_macros(s: str, defs: dict[str, str], limit: int = 2) -> tuple[str, int]:
    """Substitute collected macro bodies into `s` (definitions stay out of it)."""
    if not defs:
        return s, 0
    pat = re.compile(r'\\(' + '|'.join(map(re.escape, sorted(defs))) + r')(?![a-zA-Z])')
    total = 0
    for _ in range(limit):
        s, k = pat.subn(lambda m: defs[m.group(1)] + ' ', s)
        total += k
        if not k:
            break
    return s, total


def balance_blocks(s: str) -> tuple[str, int]:
    """Close brace groups that never close within their own paragraph block.

    A tokenizer that respects grouping will swallow the entire rest of a document
    into an unclosed argument, so `\\label{sec:intro` with no `}` costs every
    hardware sentence after it -- measured, not hypothesised: the
    `unbalanced-brace-mid-document` fixture loses its A100 sentence without this.
    TeX arguments essentially never span a blank line, so an unbalanced block is a
    truncated or mistyped argument and closing it locally is the conservative
    repair. Counting, not parsing: no loop over parser state, so it cannot hang.
    """
    out, repaired = [], 0
    # a capturing split alternates content, separator, content, ...; only the
    # even positions are content (testing for a leading newline would wrongly skip
    # a content block that merely starts on its own line)
    for i, block in enumerate(re.split(r'(\n[ \t]*\n)', s)):
        if i % 2 == 0 and block.strip():
            depth, esc = 0, False
            for ch in block:
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == '{':
                    depth += 1
                elif ch == '}':
                    depth = max(0, depth - 1)
            if depth:
                block += '}' * depth
                repaired += depth
        out.append(block)
    return ''.join(out), repaired


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
    """LaTeX string -> plain text with section sentinels, plus conversion stats.

    Everything between the pre-passes and the post-split is pylatexenc's work.
    """
    stats = {'had_bibliography': False, 'macros_expanded': 0, 'body_found': False,
             'braces_repaired': 0, 'convert_warnings': 0, 'convert_error': None}
    counter = _warning_counter()
    warned_before = counter.n
    s = _strip_comments(src)

    # Macros are defined in the preamble but must not be *read* as prose, so the
    # preamble is harvested and then discarded. Pre-2000 source sometimes has no
    # \begin{document} at all; then the whole file is the body.
    body = re.search(r'\\begin\s*\{document\}(.*?)(?:\\end\s*\{document\}|\Z)', s, re.S)
    if body:
        defs = collect_macros(s[:body.start()])
        s = body.group(1)
        stats['body_found'] = True
    else:
        defs = collect_macros(s)

    s, stats['macros_expanded'] = _apply_macros(s, defs)
    s, cut = _cut_bibliography(s)
    stats['had_bibliography'] = cut
    s = _DISPLAY_MATH.sub('\n\n', s)      # inline math survives; display does not
    s, stats['braces_repaired'] = balance_blocks(s)

    try:
        walker = LatexWalker(s, latex_context=_parse_context(), tolerant_parsing=True)
        nodes, _, _ = walker.get_latex_nodes()
        s = _converter().nodelist_to_text(nodes)
    except Exception as exc:              # pylatexenc is lenient, but never trusted
        stats['convert_error'] = type(exc).__name__
        stats['convert_warnings'] = counter.n - warned_before
        return '', stats

    stats['convert_warnings'] = counter.n - warned_before
    return _REF_LINE.sub(r'\n\n\1', s), stats


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

    `.bbl` must already be excluded by the caller (see `arxiv_source`); this is
    bibliography-exclusion layer 1.
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

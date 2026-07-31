"""arXiv LaTeX source -> `records.Paragraph` list, stdlib only.

Deliberately not a real TeX parser: what the matcher needs is *prose with word
boundaries*, not semantic fidelity, so this is a pragmatic de-TeXer whose correctness
is bought with fixtures (`tests/fixtures/latex_cases.yaml`) rather than a grammar.

**Off-the-shelf parsers were evaluated and rejected**, on measurements against this
module's own fixture suite rather than on taste. pylatexenc 2.11 passed 8/19 cases to
our 19/19 and took 74 ms/paper against our 4 ms on a realistic 22 KB paper (56 vs 3
single-core hours over 2.7M papers). Most of its misses are *policy* rather than
parsing -- it faithfully renders floats with captions, verbatim/listings, tabular and
the entire bibliography, because rendering is its job, whereas the measurement
requires dropping them -- so a parser-based pipeline would keep nearly all the code
below *and* add a dependency. Its hard gaps here: no `\input` resolution across tar
members, no user `\newcommand` expansion, and `math_mode='remove'` deletes `$8$`,
destroying device counts. The one thing it does better is unicode accents
(`Erd\H{o}s` -> `Erdős` vs our ASCII `Erdos`), which does not affect matching because
hardware names are ASCII.

Returns real `records.Paragraph` objects so `scan.scan_paragraphs` -- the shared
match/gate/cap core -- runs unchanged over both corpora.

Decisions that affect the measurement, all deliberate:

- **Bibliography exclusion in three layers.** S2ORC gets this free (`bibliography`
  is a separate object); arXiv must not lose it, or a reference titled
  "GPU-accelerated Monte Carlo..." becomes a false candidate. Layers: `.bbl` files
  are never read; the document is truncated at a bibliography macro occurring past
  half-way (the position guard stops a preamble `\\bibliographystyle` from
  decapitating the paper); and any surviving reference-shaped paragraph is dropped.
- **Floats are dropped with their captions** (figure/table/algorithm/listing), for
  comparability with S2ORC's GROBID body text, which largely excludes them. A
  caption alone also gives the LLM almost no usage context.
- **Inline math becomes a space, except when it is purely numeric.** `"we used $8$
  V100 GPUs"` must keep the 8: device counts are a headline estimand.
- **Whitespace is collapsed to single spaces.** `GATE_WINDOW_CHARS` (250) and
  `PASSAGE_CHAR_CAP` (2500) are character budgets, and LaTeX line-wrapping would
  otherwise spend 5-10% of them on newlines and indentation.

Every function returns a stats dict alongside its output; those land in the
`ingest_stats` product, which is the only way to spot the LaTeX converter silently
degrading across eras (a converter that decays over time manufactures exactly the
upward trend this project measures).
"""

import re

from accelscan.config import SPLIT_LONG_PARA_CHARS
from accelscan.records import MIN_PARA_CHARS, Paragraph

MAX_SOURCE_BYTES = 20_000_000
MAX_INCLUDE_DEPTH = 8
MAX_TITLE_CHARS = 120
BIB_POSITION_GUARD = 0.5      # bibliography macros before this fraction are ignored

# Sentinel for a section heading: NUL cannot occur in decoded TeX prose.
SEC_OPEN, SEC_CLOSE = '\x00SEC:', '\x00'

TEXT_EXTS = ('.tex', '.ltx', '.txt')

# Dropped whole, replaced by a paragraph break. Math, floats, code, verbatim.
DROP_ENVIRONMENTS = frozenset("""
    equation eqnarray align alignat gather multline flalign displaymath math
    array matrix pmatrix bmatrix vmatrix Vmatrix smallmatrix cases split
    figure figure* table table* tabular tabularx longtable wraptable wrapfigure
    subfigure subtable sidewaysfigure sidewaystable
    algorithm algorithmic algorithm2e algorithmicx pseudocode
    lstlisting listing verbatim Verbatim minted alltt code
    tikzpicture pgfpicture picture circuitikz forest
    thebibliography
""".split())

# Unwrapped to their contents: formatting that surrounds real prose.
UNWRAP_MACROS = frozenset("""
    emph textbf textit textsl texttt textsc textrm textsf textnormal textup
    mbox hbox text ensuremath mathrm textquote uline underline so
    subsubsubsection add caption* nolinkurl path
""".split())

SECTION_MACROS = ('chapter', 'section', 'subsection', 'subsubsection', 'paragraph',
                  'subparagraph')

# Dropped with their argument: citations, cross-references, notes.
DROP_WITH_ARG = frozenset("""
    cite citep citet citealp citealt citeauthor citeyear citenum nocite Cite
    ref eqref autoref cref Cref pageref label footnote footnotetext thanks
    index glossary marginpar bibliographystyle bibliography usepackage
    documentclass documentstyle input include includegraphics graphicspath
    hypersetup newcommand renewcommand providecommand def newtheorem
    setlength addtolength definecolor pagestyle thispagestyle vspace hspace
""".split())

_COMMENT = re.compile(r'(?<!\\)((?:\\\\)*)%[^\n]*\n[ \t]*')
_DISPLAY_MATH = re.compile(r'(?<!\\)\$\$.*?(?<!\\)\$\$|\\\[.*?\\\]', re.S)
_INLINE_MATH = re.compile(r'(?<!\\)\$([^$]{0,400}?)(?<!\\)\$|\\\((.{0,400}?)\\\)', re.S)
_NUMERIC_MATH = re.compile(r'^[\d\s.,]+$')
_BIB_STOP = re.compile(r'\\begin\s*\{thebibliography\}|\\bibliography\s*\{'
                       r'|\\printbibliography|\\bibliographystyle\s*\{')
_ENV_EDGE = re.compile(r'\\(begin|end)\s*\{([^}\n]{1,60})\}')
# Accents in two flavours. Symbol accents (\'e, \~n) may omit braces safely, but
# LETTER accents (\v \H \r \u \c \d \b \t) must require them: brace-optional
# would make '\c' eat the head of '\cite', '\r' of '\ref', '\u' of '\usepackage'
# and '\v' of '\varepsilon', silently corrupting prose into 'ite{x}' / 'ef{z}'.
_ACCENT_SYMBOL = re.compile(r'\\[`\'^"~=.]\s*\{?([a-zA-Z])\}?')
_ACCENT_LETTER = re.compile(r'\\[vHrucdbt]\s*\{([a-zA-Z])\}')
_CONTROL_WORD = re.compile(r'\\[a-zA-Z@]+\s*\*?')
_CONTROL_SYMBOL = re.compile(r'\\[^a-zA-Z@\s]')
_WS = re.compile(r'\s+')
# Escaped literals must survive both the control-symbol rule and the later global
# '&'/brace scrubbing, so they ride through as placeholders. Applied AFTER comment
# stripping, because '\\%' is an escaped backslash followed by a real comment.
_ESCAPED = {'%': '\x01', '&': '\x02', '_': '\x03', '#': '\x04', '$': '\x05'}
_UNESCAPE = {v: k for k, v in _ESCAPED.items()}
_ESCAPED_RE = re.compile(r'\\([%&_#$])')
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


def _braced(s: str, i: int) -> tuple[str, int]:
    """Content of the brace group starting at s[i]=='{', and the index after it.

    Returns ('', i) when unbalanced -- truncated source is normal on arXiv.
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

    Dropping only to EOL would glue words across the break; keeping the newline
    would invent paragraph boundaries. `\\%` is preserved.
    """
    return _COMMENT.sub(r'\1', s + '\n')


def collect_macros(src: str) -> dict[str, str]:
    """Zero-argument `\\newcommand`/`\\def` bodies, by macro name.

    Zero-arg only: general expansion is a halting problem on real source, while
    zero-arg covers the common `\\newcommand{\\ourmodel}{FooNet}` case that would
    otherwise leave holes in the prose. Self-referential and over-long bodies are
    skipped.
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


def _drop_environments(s: str, names=DROP_ENVIRONMENTS) -> tuple[str, list[str]]:
    """Remove `\\begin{env}...\\end{env}` wholesale via a nesting-aware scan.

    A non-greedy regex breaks on same-name nesting (an `align` inside an `align`),
    hence the stack.
    """
    out, dropped, pos, depth, target = [], [], 0, 0, None
    for m in _ENV_EDGE.finditer(s):
        kind, name = m.group(1), m.group(2).strip().rstrip('*')
        if depth == 0:
            if kind == 'begin' and name in names:
                out.append(s[pos:m.start()])
                target, depth = name, 1
                dropped.append(name)
        elif name == target:
            depth += 1 if kind == 'begin' else -1
            if depth == 0:
                out.append('\n\n')
                pos, target = m.end(), None
    out.append(s[pos:])
    return ''.join(out), dropped


def _strip_math(s: str) -> str:
    s = _DISPLAY_MATH.sub('\n\n', s)

    def inline(m):
        body = (m.group(1) if m.group(1) is not None else m.group(2)) or ''
        # keep bare numbers: "$8$ V100 GPUs" is a device count
        return body if _NUMERIC_MATH.match(body) else ' '

    return _INLINE_MATH.sub(inline, s)


def _drop_with_args(s: str) -> str:
    """Drop \\cite{...}, \\label{...}, \\footnote{...} and friends, argument included."""
    out, pos = [], 0
    pat = re.compile(r'\\([a-zA-Z@]+)\s*\*?')
    for m in pat.finditer(s):
        if m.start() < pos or m.group(1) not in DROP_WITH_ARG:
            continue
        out.append(s[pos:m.start()])
        j = m.end()
        while j < len(s) and s[j] in ' \n':
            j += 1
        if j < len(s) and s[j] == '[':                    # optional arg
            k = s.find(']', j)
            j = k + 1 if k != -1 else j
        while j < len(s) and s[j] == '{':
            _, j = _braced(s, j)
        out.append(' ')
        pos = j
    out.append(s[pos:])
    return ''.join(out)


def _unwrap(s: str, names=UNWRAP_MACROS, rounds: int = 3) -> str:
    pat = re.compile(r'\\(' + '|'.join(map(re.escape, sorted(names))) + r')\s*\*?\s*(?=\{)')
    for _ in range(rounds):
        out, pos, hits = [], 0, 0
        for m in pat.finditer(s):
            if m.start() < pos:
                continue
            body, end = _braced(s, m.end())
            if end == m.end():
                continue
            out.append(s[pos:m.start()])
            out.append(body)
            pos, hits = end, hits + 1
        out.append(s[pos:])
        s = ''.join(out)
        if not hits:
            break
    return s


def _mark_sections(s: str) -> str:
    """Replace section macros with a NUL-delimited sentinel + paragraph break."""
    out, pos = [], 0
    pat = re.compile(r'\\(' + '|'.join(SECTION_MACROS) + r')\s*\*?\s*(?=\{|\[)')
    for m in pat.finditer(s):
        if m.start() < pos:
            continue
        j = m.end()
        if j < len(s) and s[j] == '[':                    # short title form
            k = s.find(']', j)
            j = k + 1 if k != -1 else j
        title, end = _braced(s, j) if j < len(s) and s[j] == '{' else ('', j)
        out.append(s[pos:m.start()])
        clean = _WS.sub(' ', _CONTROL_WORD.sub(' ', title).replace('{', '')
                        .replace('}', '')).strip()[:MAX_TITLE_CHARS]
        out.append(f'\n\n{SEC_OPEN}{clean}{SEC_CLOSE}\n\n' if clean else '\n\n')
        pos = end
    out.append(s[pos:])
    return ''.join(out)


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
    """LaTeX string -> plain text with section sentinels, plus conversion stats."""
    stats = {'had_bibliography': False, 'macros_expanded': 0, 'dropped_envs': 0,
             'body_found': False, 'truncated_at': None}
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
    s, dropped = _drop_environments(s)
    stats['dropped_envs'] = len(dropped)
    s = _ESCAPED_RE.sub(lambda m: _ESCAPED[m.group(1)], s)
    s = _strip_math(s)
    s = _mark_sections(s)
    s = _drop_with_args(s)
    s = re.sub(r'\\href\s*\{[^}]*\}\s*(?=\{)', ' ', s)     # \href{url}{text} -> text
    s = re.sub(r'\\url\s*\{([^}]*)\}', r' \1 ', s)         # keep vendor domains
    s = _unwrap(s)
    s = _ACCENT_SYMBOL.sub(r'\1', s)
    s = _ACCENT_LETTER.sub(r'\1', s)
    s = _CONTROL_WORD.sub(' ', s)
    s = _CONTROL_SYMBOL.sub(' ', s)
    s = s.replace('~', ' ').replace('&', ' ').replace('{', ' ').replace('}', ' ')
    for ph, lit in _UNESCAPE.items():
        s = s.replace(ph, lit)
    # BUG 2 guard: a reference sitting directly under body prose used to be one
    # paragraph, so the reference filter deleted the prose with it.
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

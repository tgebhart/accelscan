"""Shared figure conventions for the analysis notebooks.

`setup_figures('<notebook>')` in a notebook's setup cell installs both behaviours
below by patching `plt.show`, so no plot cell needs to know about either. Every
figure in every notebook is closed with `plt.show()`, which is what makes one hook
sufficient — keep it that way when adding figures.

1. **Integer year ticks.** Matplotlib's default locator picks half-steps on short
   or awkward spans (2007.5, 2012.5), which reads as a nonsense date. Detection is
   by view range, not axis label: an axis counts as a year axis when its whole
   visible span sits inside [YEAR_LO, YEAR_HI] on a linear scale. That
   deliberately ignores labels, so "Median hardware age (years)" (0-15) is left
   alone while an unlabeled release-year axis is fixed, on either x or y.

2. **Powers-of-ten log axes.** A log axis gets ticks and grid lines only at powers
   of ten; matplotlib's minor ticks at 2..9 x 10^k read as unevenly spaced and, with
   a FuncFormatter attached, get labelled too.

3. **Percent tick labels.** Any axis whose label opens with "Percentage",
   "Proportion" or "Cumulative share" gets a `PercentFormatter`, so the unit sits
   on the ticks ("40%") and the axis label spells the word out. Axis labels must
   therefore never open with a bare "%" glyph. The 0-100 vs 0-1 scale is read off
   the axis, not assumed.

4. **PDF export.** Each figure is written to
   `output/analysis/<notebook>/<nn>_<slug>.pdf`, where the slug comes from the
   figure's suptitle or its axes titles — i.e. figures are named after what they
   plot, with no filename bookkeeping in the plot cells. The `nn` prefix
   preserves notebook order; it is assigned in show() order and reset by
   `setup_figures`, so a full top-to-bottom run numbers cleanly (re-running a
   single cell mid-notebook can bump a number, which is a dev-time artifact).
"""

import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import (LogLocator, MaxNLocator, NullFormatter,
                              NullLocator, PercentFormatter)

YEAR_LO, YEAR_HI = 1980, 2040
# An axis whose label opens with one of these carries a percentage/share, so the
# unit belongs on the tick labels ("40%") and the word, not the glyph, belongs in
# the axis label ("Percentage of papers", never "% of papers").
PCT_LABEL = re.compile(r'(?i)^\s*(percentage|proportion|cumulative share)\b')
FIG_ROOT = Path('output/analysis')
FIG_FORMAT = 'pdf'
SLUG_MAX = 70

_state = {'notebook': None, 'n': 0, 'saved': set()}


def _is_year_axis(axis) -> bool:
    lo, hi = sorted(axis.get_view_interval())
    scale = (axis.axes.get_xscale() if axis.axis_name == 'x' else axis.axes.get_yscale())
    return scale == 'linear' and YEAR_LO <= lo and hi <= YEAR_HI


def integerize_year_ticks(fig) -> None:
    """Force integer ticks on every year-like axis of `fig`."""
    for ax in fig.get_axes():
        for axis in (ax.xaxis, ax.yaxis):
            if _is_year_axis(axis):
                axis.set_major_locator(MaxNLocator(integer=True, nbins='auto'))
                axis.set_minor_locator(NullLocator())
                axis.grid(False, which='minor')


def clean_log_axes(fig) -> None:
    """Log axes get ticks and grid lines at powers of ten only.

    Matplotlib's log default adds minor ticks at 2..9 x 10^k, which read as
    unevenly spaced ticks and grid lines (and, with a FuncFormatter attached, get
    labelled too). Where fewer than two powers fall inside the view -- a range like
    2-24 GB -- the limits are snapped out to the enclosing decades instead of
    falling back to minor ticks, so the axis keeps at least two labels.
    """
    for ax in fig.get_axes():
        for axis, name in ((ax.xaxis, 'x'), (ax.yaxis, 'y')):
            if (ax.get_xscale() if name == 'x' else ax.get_yscale()) != 'log':
                continue
            lo, hi = sorted(axis.get_view_interval())
            if lo <= 0:
                continue
            if math.floor(math.log10(hi)) - math.ceil(math.log10(lo)) < 1:
                lo, hi = 10 ** math.floor(math.log10(lo)), 10 ** math.ceil(math.log10(hi))
                (ax.set_xlim if name == 'x' else ax.set_ylim)(lo, hi)
            axis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
            axis.set_minor_locator(NullLocator())
            axis.set_minor_formatter(NullFormatter())
            ax.grid(False, which='minor', axis=name)


def percent_tick_labels(fig) -> None:
    """Put a % on the ticks of every axis whose label declares a percentage.

    The scale comes from the label's wording, NOT from the data range: "Percentage
    of x" is 0-100, "Proportion/Cumulative share of x" is 0-1. Inferring it from
    the range is unsound -- a percentage axis topping out at 0.3% is
    indistinguishable from a proportion, and guessing wrong scales the tick labels
    by 100x.
    """
    for ax in fig.get_axes():
        for axis, label in ((ax.xaxis, ax.get_xlabel()), (ax.yaxis, ax.get_ylabel())):
            if not PCT_LABEL.match(label):
                continue
            xmax = 100 if label.lower().lstrip().startswith('percentage') else 1
            # Decimals from the visible span, not matplotlib's default guess, which
            # gives "0.0%" on one axis and "0%" on another in the same notebook.
            # Prevalence axes span a couple of percent and do need a decimal.
            lo, hi = sorted(axis.get_view_interval())
            span = (hi - lo) / xmax
            decimals = 0 if span >= 0.05 else 1 if span >= 0.005 else 2
            axis.set_major_formatter(PercentFormatter(xmax=xmax, decimals=decimals))


def spread_labels(ax, tol: float = 0.2, pad: float = 1.0) -> int:
    """Nudge overlapping point labels apart; returns how many were moved.

    For scatter labels created with `textcoords='offset points'`. Greedy and
    first-come-first-served: each label keeps its offset if it clears the ones
    already placed, otherwise it tries progressively farther offsets and finally
    takes whichever candidate overlaps least. `tol` is the fraction of the smaller
    label's area allowed to overlap, so slight touching passes and only genuine
    collisions are moved -- labels stay near their points rather than being flung
    across the panel.
    """
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    cands = [(4, 4), (4, -9), (-6, 7), (-6, -12), (7, 12), (7, -18),
             (-10, 15), (-10, -22), (14, 20), (14, -28)]

    def overlap(a, b) -> float:
        dx = min(a.x1, b.x1) - max(a.x0, b.x0) + 2 * pad
        dy = min(a.y1, b.y1) - max(a.y0, b.y0) + 2 * pad
        if dx <= 0 or dy <= 0:
            return 0.0
        return dx * dy / max(1e-9, min(a.width * a.height, b.width * b.height))

    placed, moved = [], 0
    for t in ax.texts:
        start = tuple(t.get_position())
        best, best_score = start, None
        for i, c in enumerate([start] + cands):
            t.set_position(c)
            score = max((overlap(t.get_window_extent(r), p) for p in placed), default=0.0)
            if score <= tol:
                best, best_score = c, score
                break
            if best_score is None or score < best_score:
                best, best_score = c, score
        t.set_position(best)
        if tuple(best) != start:
            moved += 1
        placed.append(t.get_window_extent(r))
    return moved


def _slug(fig) -> str:
    """Figure name from its titles: suptitle, else the distinct axes titles."""
    sup = fig._suptitle.get_text() if fig._suptitle else ''
    titles = [t for t in (sup, *(ax.get_title() for ax in fig.get_axes())) if t.strip()]
    seen = list(dict.fromkeys(titles))
    text = ' + '.join(seen[:2]) if not sup else sup
    s = re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')
    if len(s) > SLUG_MAX:                      # cut at a word boundary
        s = s[:SLUG_MAX].rsplit('-', 1)[0]
    return s or 'figure'


def save_figure(fig) -> Path | None:
    """Write `fig` to output/analysis/<notebook>/<nn>_<slug>.pdf."""
    if _state['notebook'] is None:
        return None
    d = FIG_ROOT / _state['notebook']
    d.mkdir(parents=True, exist_ok=True)
    _state['n'] += 1
    path = d / f"{_state['n']:02d}_{_slug(fig)}.{FIG_FORMAT}"
    fig.savefig(path, format=FIG_FORMAT, bbox_inches='tight')
    return path


def setup_figures(notebook: str | None = None) -> None:
    """Patch plt.show to fix year ticks and export each figure. Idempotent."""
    _state.update(notebook=notebook, n=0, saved=set())
    # Unwrap any hook we installed earlier and re-wrap, rather than returning
    # early: that keeps this idempotent (hooks never stack) while still picking up
    # edits to this module in a live kernel, where an early return would pin the
    # first version of show() for the life of the session.
    _show = getattr(plt.show, '_accelscan_wrapped', plt.show)

    def show(*a, **kw):
        for num in plt.get_fignums():
            fig = plt.figure(num)
            if id(fig) in _state['saved']:
                continue
            _state['saved'].add(id(fig))
            integerize_year_ticks(fig)
            clean_log_axes(fig)
            percent_tick_labels(fig)
            save_figure(fig)
        return _show(*a, **kw)

    show._accelscan_wrapped = _show
    plt.show = show


# Back-compat: the year fix alone, without PDF export.
def install_year_tick_fix() -> None:
    setup_figures(None)

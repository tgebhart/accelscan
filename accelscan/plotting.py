"""Shared figure conventions for the analysis notebooks.

Only one rule so far, but it applies to every figure in every notebook: a year
axis must never be labelled with fractions. Matplotlib's default locator picks
half-steps whenever a span is short or awkward (2007.5, 2012.5), which reads as
a nonsense date on a calendar-year axis.

Rather than annotating dozens of plot cells, `install_year_tick_fix()` patches
`plt.show` to integerize year axes just before draw. Detection is by view range,
not by axis label: an axis counts as a year axis when its whole visible span
sits inside [YEAR_LO, YEAR_HI] on a linear scale. That deliberately ignores
labels, so "Median hardware age (years)" (0-15) is left alone while an unlabeled
release-year axis is still fixed, on either x or y.
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

YEAR_LO, YEAR_HI = 1980, 2040


def _is_year_axis(axis) -> bool:
    lo, hi = sorted(axis.get_view_interval())
    return (axis.axes.get_xscale() == 'linear' if axis.axis_name == 'x'
            else axis.axes.get_yscale() == 'linear') and YEAR_LO <= lo and hi <= YEAR_HI


def integerize_year_ticks(fig=None) -> None:
    """Force integer ticks on every year-like axis of `fig` (default: all open)."""
    figs = [fig] if fig is not None else [plt.figure(n) for n in plt.get_fignums()]
    for f in figs:
        for ax in f.get_axes():
            for axis in (ax.xaxis, ax.yaxis):
                if _is_year_axis(axis):
                    axis.set_major_locator(MaxNLocator(integer=True, nbins='auto'))


def install_year_tick_fix() -> None:
    """Patch plt.show so year axes are integerized on every figure. Idempotent."""
    if getattr(plt.show, '_accelscan_year_fix', False):
        return
    _show = plt.show

    def show(*a, **kw):
        integerize_year_ticks()
        return _show(*a, **kw)

    show._accelscan_year_fix = True
    plt.show = show

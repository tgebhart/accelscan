"""Figure conventions that are silent when they break.

A missed tick formatter does not raise -- it ships a figure whose axis reads
"41" where it means "41%", or "2007.5" where it means a year. So the cases here
are the ones where the hook could skip an axis it was supposed to reach.
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

from accelscan.plotting import (_all_axes, clean_log_axes, integerize_year_ticks,
                               percent_tick_labels)


@pytest.fixture
def fig():
    f = plt.figure()
    yield f
    plt.close(f)


def test_all_axes_includes_secondary_axes(fig):
    """`fig.get_axes()` omits secondary axes -- they live in `child_axes`. Every
    hook iterates `_all_axes` for exactly this reason."""
    ax = fig.add_subplot()
    sec = ax.secondary_yaxis('right', functions=(lambda v: 2 * v, lambda v: v / 2))
    assert sec not in fig.get_axes(), 'matplotlib behaviour this helper exists for'
    assert sec in _all_axes(fig)
    assert ax in _all_axes(fig)


def test_percent_labels_reach_a_secondary_axis(fig):
    ax = fig.add_subplot()
    ax.plot([-1, 0, 1], [1.2, 0.5, 1.8])
    base = 4.33
    sec = ax.secondary_yaxis('right', functions=(lambda v, b=base: 100 * v / b,
                                                 lambda q, b=base: q * b / 100))
    sec.set_ylabel('Percentage of baseline output (4.33 papers/yr)')
    fig.canvas.draw()
    percent_tick_labels(fig)
    fig.canvas.draw()
    assert all(t.get_text().endswith('%')
               for t in sec.get_yticklabels() if t.get_text())
    # the primary axis carries papers, not a percentage, and must be left alone
    assert not any(t.get_text().endswith('%')
                   for t in ax.get_yticklabels() if t.get_text())


def test_percent_scale_comes_from_the_wording_not_the_range(fig):
    """"Percentage of x" is 0-100, "Proportion of x" is 0-1. Inferring from the
    data range would scale a 0.3%-max percentage axis by 100."""
    for label, val, want in (('Percentage of papers', 40.0, '40%'),
                             ('Proportion of papers', 0.4, '40%')):
        ax = fig.add_subplot()
        ax.plot([0, 1], [val * 0.5, val])
        ax.set_ylabel(label)
        fig.canvas.draw()
        percent_tick_labels(fig)
        fig.canvas.draw()
        texts = [t.get_text() for t in ax.get_yticklabels() if t.get_text()]
        assert want in texts, f'{label!r} produced {texts}'
        fig.clear()


def test_percent_labels_ignore_an_unrelated_label(fig):
    ax = fig.add_subplot()
    ax.plot([0, 1], [0.2, 0.9])
    ax.set_ylabel('Papers per year (change)')
    fig.canvas.draw()
    percent_tick_labels(fig)
    fig.canvas.draw()
    assert not any(t.get_text().endswith('%')
                   for t in ax.get_yticklabels() if t.get_text())


def test_year_ticks_are_integers_and_leave_non_year_axes_alone(fig):
    ax = fig.add_subplot()
    ax.plot([2007, 2008, 2009], [0.5, 1.5, 2.5])
    ax.set_xlim(2007, 2009)
    ax.set_ylim(0, 1.5)                   # a rate, not a year: decimals are correct
    integerize_year_ticks(fig)
    fig.canvas.draw()
    xs = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
    assert xs and not any('.' in t for t in xs), xs
    assert any('.' in t for t in
               [t.get_text() for t in ax.get_yticklabels() if t.get_text()])


def test_log_axis_keeps_only_powers_of_ten(fig):
    ax = fig.add_subplot()
    ax.plot([1, 2, 3], [1.0, 100.0, 10_000.0])
    ax.set_yscale('log')
    clean_log_axes(fig)
    fig.canvas.draw()
    minor = [t for t in ax.get_yticklabels(minor=True) if t.get_text()]
    assert not minor, f'minor log ticks left labelled: {[t.get_text() for t in minor]}'

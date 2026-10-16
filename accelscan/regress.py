"""pyfixest helpers for the citation-returns analysis (`notebooks/citations.ipynb`).

Estimation is **pyfixest's** (`pf.feols`, `pf.fepois`) -- this module holds only
what pyfixest does not: the polars/pandas hand-off, a guard against a clustering
degeneracy that pyfixest reports without warning, and result-tidying into polars
so the notebook's tables are polars like every other table in the repo.

Nothing here estimates anything. If a function in here starts fitting a model,
it is in the wrong file.

The clustering guard is the reason this module exists at all rather than the
notebook calling pyfixest bare. An interaction specification (`i(field, x)`,
one slope per field) clustered at the interacting group's own level returns
standard errors that are numerical noise -- on this sample, a field slope off
496 papers came back with an SE of 0.005 and a slope-equality chi2 of 5.5
million, versus 0.021 and 63 once clustered finer. pyfixest fits it happily and
says nothing, and the numbers look publishable, so the check has to be explicit
and it has to run before the estimate is read. See `assert_cluster_finer`.
"""

from __future__ import annotations

import numpy as np
import polars as pl

# The label a `field_top`-style column uses for its pooled tail; slope tables
# carry it through rather than dropping the pooled row.
POOLED = 'Other fields'


def to_pandas(df: pl.DataFrame, cols: list[str] | None = None):
    """polars -> pandas for pyfixest, without going through Arrow.

    `DataFrame.to_pandas()` needs pyarrow, which is not installed (and is a
    heavy dependency for the EC2 stage-1 image to carry). Every column handed to
    a formula here is a flat numeric or string, so a dict of numpy arrays is an
    exact and cheap conversion. Pass `cols` to hand over only what the formula
    references -- these frames are wide, and pyfixest copies what it is given.
    """
    import pandas as pd

    use = list(dict.fromkeys(cols)) if cols else df.columns
    missing = [c for c in use if c not in df.columns]
    if missing:
        raise ValueError(f'columns not in frame: {missing}')
    return pd.DataFrame({c: df[c].to_numpy() for c in use})


def assert_cluster_finer(df: pl.DataFrame, group: str, cluster: str) -> None:
    """Raise if every level of `group` sits inside a single `cluster` level.

    The degeneracy this catches: an interaction term `x * 1[group = g]` is
    non-zero only on group g's rows. If those rows all fall in one cluster, that
    cluster's partial score for the term IS the term's total score, which the
    regression's first-order condition sets to zero -- so the cluster-robust
    meat collapses to numerical noise and the SEs come back spuriously tiny.
    Clustering *finer* than the group is the fix: with field x year clusters, a
    field's term gets one non-zero score per year.

    Call this before any `i(group, x)` specification. pyfixest does not check
    it, and the degenerate output is not distinguishable from a precise estimate
    by inspection.
    """
    for col in (group, cluster):
        if col not in df.columns:
            raise ValueError(f'{col!r} not in frame')
    m = df.group_by(group).agg(pl.col(cluster).n_unique().alias('_c'))['_c'].max()
    if m is not None and int(m) <= 1:
        raise ValueError(
            f'each level of {group!r} sits inside a single {cluster!r}: every '
            f'interaction term would be confined to one cluster, driving the '
            f'cluster-robust SEs to numerical noise via the first-order '
            f'condition. Cluster on something finer than {group!r}.')


def tidy(model, *, keep: list[str] | None = None, label: str = '',
         pct: bool = True) -> pl.DataFrame:
    """A pyfixest model's coefficients as a polars frame.

    Adds `pct` = `100 * (exp(beta) - 1)`, the multiplicative reading of a
    coefficient on a log outcome, and the same transform of the CI bounds. With
    a `log1p` outcome that is a percentage change in `1 + y`, not in `y`, so the
    raw coefficient travels alongside it and is never replaced by it. Pass
    `pct=False` for an outcome that is not on a log scale (a percentile rank,
    say), where exponentiating would be meaningless.

    `spec`, `fe` and `cluster` ride on every row, so a results table cannot show
    two coefficients side by side without showing that they came from different
    specifications.
    """
    t = model.tidy()
    out = pl.DataFrame({
        'spec': [label or None] * len(t),
        'term': [str(i) for i in t.index],
        'beta': t['Estimate'].to_numpy(),
        'se': t['Std. Error'].to_numpy(),
        't': t['t value'].to_numpy(),
        'p': t['Pr(>|t|)'].to_numpy(),
        'lo': t['2.5%'].to_numpy(),
        'hi': t['97.5%'].to_numpy(),
    })
    if pct:
        out = out.with_columns(
            pct=100 * (pl.col('beta').exp() - 1),
            pct_lo=100 * (pl.col('lo').exp() - 1),
            pct_hi=100 * (pl.col('hi').exp() - 1))
    out = out.with_columns(
        n=pl.lit(int(model._N)),
        fe=pl.lit(str(getattr(model, '_fixef', '') or 'none')),
        cluster=pl.lit(vcov_label(model)),
        r2_within=pl.lit(float(getattr(model, '_r2_within', float('nan')))))
    return out.filter(pl.col('term').is_in(keep)) if keep else out


def vcov_label(model) -> str:
    """How this model's SEs were computed, for the results table."""
    kind = str(getattr(model, '_vcov_type', '?'))
    if kind != 'CRV':
        return kind
    by = getattr(model, '_clustervar', None)
    by = ' x '.join(by) if isinstance(by, (list, tuple)) else str(by or '?')
    g = getattr(model, '_G', None)
    n_g = f' ({max(g):,} clusters)' if isinstance(g, (list, tuple)) and g else ''
    return f'CRV1 {by}{n_g}'


def stack(models: dict, *, keep: list[str] | None = None,
          pct: bool = True) -> pl.DataFrame:
    """`tidy` over `{label: model}`, concatenated into one results table."""
    frames = [tidy(m, keep=keep, label=lab, pct=pct) for lab, m in models.items()]
    return pl.concat(frames) if frames else pl.DataFrame()


def slopes(model, group: str, x: str, *, pct: bool = True) -> pl.DataFrame:
    """Per-group slopes from an `i(group, x)` specification, one row per level.

    pyfixest names those coefficients `group::level:x`; this parses the level
    back out so the table can join to group counts and plot with real labels.
    Every level is present (pyfixest's `i()` omits no reference category when it
    interacts with a continuous variable), so each `beta` reads directly as that
    group's own slope rather than as a contrast against a base group.
    """
    t = tidy(model, pct=pct)
    out = (t.with_columns(level=pl.col('term')
                          .str.extract(rf'^{group}::(.+):{x}$', 1))
           .filter(pl.col('level').is_not_null()))
    if not out.height:
        raise ValueError(f'no i({group}, {x}) coefficients in this model; '
                         f'terms are {t["term"].to_list()[:5]}')
    return out


def equality_test(model, terms: list[str] | None = None) -> tuple[float, int, float]:
    """Wald test that the selected coefficients are EQUAL: (chi2, df, p).

    The heterogeneity test for a `slopes` specification. It contrasts each
    coefficient against the first rather than against zero, so a set of
    uniformly large but identical slopes is not flagged as heterogeneous --
    which a joint test of "all zero" would do.
    """
    names = [str(i) for i in model.coef().index]
    idx = [names.index(t) for t in terms] if terms else list(range(len(names)))
    if len(idx) < 2:
        raise ValueError('need >= 2 coefficients to test equality')
    r = np.zeros((len(idx) - 1, len(names)))
    for j, i in enumerate(idx[1:]):
        r[j, idx[0]], r[j, i] = -1.0, 1.0
    res = model.wald_test(R=r, q=np.zeros(r.shape[0]), distribution='chi2')
    return float(res['statistic']), r.shape[0], float(res['pvalue'])


def quantile_bins(df: pl.DataFrame, col: str, n_bins: int, *,
                  name: str = 'bin') -> tuple[pl.DataFrame, pl.DataFrame]:
    """Add a quantile-bin label for `col`, and return the per-bin summary.

    Data prep for the nonparametric dose-response, after which the regression is
    just `y ~ C(bin) | fe` in pyfixest with the lowest bin as reference. Ties are
    why the summary is returned rather than assumed: VRAM piles up on 8/12/16/24
    GB, so `allow_duplicates` can leave bins very unequal in size, and a bin's
    support belongs next to its coefficient. Labels are zero-padded so string
    sorting and numeric order agree.
    """
    d = df.drop_nulls([col]).with_columns(
        **{name: pl.col(col).qcut(n_bins,
                                  labels=[f'{i:02d}' for i in range(n_bins)],
                                  allow_duplicates=True).cast(pl.String)})
    summary = (d.group_by(name)
               .agg(n_bin=pl.len(), x_mean=pl.col(col).mean(),
                    x_lo=pl.col(col).min(), x_hi=pl.col(col).max())
               .sort(name))
    return d, summary


def dose_table(model, summary: pl.DataFrame, *, name: str = 'bin') -> pl.DataFrame:
    """Bin coefficients joined to their support, with the reference bin as zero.

    pyfixest omits the lowest bin as the `C()` reference; it belongs in the table
    as an explicit zero so a dose-response plot starts at the baseline rather
    than at the second bin.
    """
    t = (tidy(model)
         .with_columns(_lvl=pl.col('term').str.extract(r'\[T\.(.+)\]$', 1))
         .filter(pl.col('_lvl').is_not_null())
         .drop(name, strict=False)
         .rename({'_lvl': name}))
    ref_level = summary[name][0]
    ref = t.head(1).with_columns(
        term=pl.lit(f'C({name})[{ref_level}] (reference)'),
        **{name: pl.lit(ref_level)},
        **{c: pl.lit(0.0) for c in ('beta', 'se', 'lo', 'hi', 'pct',
                                    'pct_lo', 'pct_hi')},
        **{c: pl.lit(None, dtype=pl.Float64) for c in ('t', 'p')})
    return pl.concat([ref, t]).join(summary, on=name, how='left').sort(name)


def delta_citations(beta, baseline: float):
    """A log1p-outcome coefficient re-expressed in citations.

    The regressions estimate `log(1 + c)`, so a coefficient is a proportional
    effect on `1 + c` and is unit-free. To put it back in citations it has to be
    evaluated somewhere on the citation scale:

        delta_c = (1 + baseline) * (exp(beta) - 1)

    i.e. "a paper sitting at `baseline` citations is predicted to have this many
    more, per unit of the regressor". Use a median rather than a mean for
    `baseline` -- the citation distribution's mean is pulled far above the
    typical paper (39 vs 13 on this sample), so a mean-based conversion reports
    an effect no ordinary paper experiences.

    The transform is monotone in `beta`, so applying it to a CI's endpoints
    gives that CI's image exactly, and applying it to axis tick positions gives
    a valid secondary axis. It is **not** an average effect on citations across
    the sample: effects are proportional, so a paper with 200 citations gets far
    more than one with 5, and this reports the conversion at one chosen point.

    Either argument may be a polars expression, so the same function serves a
    scalar coefficient, a numpy array of axis ticks, and a `with_columns` call
    that converts a whole slope table at each row's own baseline.
    """
    expr = (pl.Expr, pl.Series)
    if isinstance(beta, expr) or isinstance(baseline, expr):
        b = beta if isinstance(beta, expr) else pl.lit(beta)
        return (1 + baseline) * (b.exp() - 1)
    return (1 + baseline) * np.expm1(np.asarray(beta, dtype=float))

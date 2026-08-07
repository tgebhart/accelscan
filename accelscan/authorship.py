"""Stage 4b: author-level GPU access, output, and citations.

The paper-level inequality analyses (Gini/Theil/p90-vs-median in `capacity.ipynb`)
asked how unequally *works* are equipped. This asks the same of *people*, and then
whether an author's output changes after a discrete jump in the accelerator
capacity they report.

Two steps, one SLURM job (`slurm/s2_authorship.txt`):

`--step crosswalk` — explode the papers table's `authors` list into
(corpusid, authorid) rows, part-by-part, writing each part straight to S3. Keyed
by Semantic Scholar *release*, not registry version: it depends only on the
metadata snapshot, so it can run before/independently of any re-extraction.

`--step panel` — the author-year panel and the jumps table, from
crosswalk x capacity x paper_flags x citations.

**Identification, stated honestly.** Access is chosen, not assigned. Nothing here
recovers a causal effect: the estimand is "how do trajectories change around a
capacity jump, relative to matched peers who had not yet jumped". Anticipation
(the A100 arrives *because* a big project is starting) inflates the estimate with
no pre-trend signature, which is why `event_study` reports the full tau window
including pre-periods, why `--step panel` records `coauthor_sourced`, and why the
notebook runs placebo pseudo-jumps.

**Known measurement bias, both directions.** Semantic Scholar author-id *splits*
truncate careers: a split at the jump year reads as an output drop, attenuating
any post-jump effect. *Merges* inflate output and can fabricate jumps; the
>MAX_PAPERS_PER_YEAR filter is a crude merge screen and its bite is reported.

  python -m accelscan.authorship --step crosswalk
  python -m accelscan.authorship --step panel
  python -m accelscan.authorship --step panel --count-policy explicit
"""

import argparse
import io
import sys
import warnings

import numpy as np
import polars as pl

from accelscan.config import BUCKET, PAPERS_PREFIX, S2_RELEASE_ID
from accelscan.paths import (S2ORC, Corpus, analytic_base, author_analytic_key,
                            authorship_crosswalk_parts, capacity_key,
                            get_corpus, inventory_glob, s3_uri)

# --- population filters ----------------------------------------------------
MIN_CAREER_PAPERS = 3
MIN_CAREER_SPAN = 3           # years, inclusive
MAX_FIRST_YEAR = 2021         # need some post-period to observe
MAX_PAPERS_PER_YEAR = 50      # merged-profile screen
MIN_FIELD_PREVALENCE = 0.01   # never-GPU controls only from accelerator-active fields

# --- event definitions -----------------------------------------------------
JUMP_RATIO = 4.0              # primary; {2, 8} as robustness
EVENT_TYPES = ('ratio_jump', 'dc_first', 'gpu_first')
CITATION_EVENT_CUTOFF = 2019  # i_3 window complete through ~2022

# --- event study -----------------------------------------------------------
TAU_MIN, TAU_MAX = -5, 5
BASE_TAUS = (-5, -4, -3, -2)  # tau=-1 excluded: anticipation
N_CONTROLS = 5
BOOTSTRAP_B = 300
SEED = 42

OUTCOMES = ('log_papers', 'n_accel', 'n_nonaccel', 'cit_i3_mean')


# ===========================================================================
# step 1: (corpusid, authorid) crosswalk
# ===========================================================================

def _fulltext_ids(so: dict, c: Corpus) -> pl.DataFrame:
    """Full-text paper ids, from the (registry-independent) inventory prefix.

    `extra_columns`/`missing_columns` for the same reason as in `denominator`:
    inventory parts are overwritten in place, so the prefix can briefly hold two
    schema shapes at once.
    """
    return (pl.scan_parquet(inventory_glob(c), storage_options=so,
                            extra_columns='ignore', missing_columns='insert')
            .select(c.key).unique().collect()
            .with_columns(fulltext=pl.lit(True)))


def explode_authors(part: pl.DataFrame, fulltext: pl.DataFrame,
                    key: str = 'corpusid') -> pl.DataFrame:
    """One papers part -> (paper, author) rows, byline position preserved.

    Row-count invariant: exactly one output row per NON-NULL author-list element.
    Two distinct things produce a dropped row and they must not be conflated --
    an empty/null `authors` list (a paper with no byline recorded), and a null
    element *inside* a list (a real byline slot whose author id is missing; ~0.1%
    of elements in the 2026-07-28 release). Counting both and asserting the
    arithmetic is what would catch a silently mangled `authors` column, which
    would otherwise multiply or drop output for every downstream author-year.

    `n_authors` is the recorded byline length, including slots whose id is null,
    so fractional attribution divides by team size rather than by the number of
    ids that happen to have resolved.
    """
    n_elements = int(part['authors'].list.len().fill_null(0).sum())
    ex = (part
          .with_columns(
              n_authors=pl.col('authors').list.len().cast(pl.UInt16),
              field=pl.col('s2fieldsofstudy').list.first().struct.field('category'))
          .drop('s2fieldsofstudy')
          .explode('authors', empty_as_null=True)
          .rename({'authors': 'authorid'})
          # position is assigned BEFORE dropping null ids, so a missing id does
          # not shift the author behind it into first place
          .with_columns(author_pos=pl.int_range(pl.len()).over(key).cast(pl.UInt16)))
    n_empty_lists = int((part['authors'].list.len().fill_null(0) == 0).sum())
    n_null_ids = ex['authorid'].null_count() - n_empty_lists
    out = (ex.filter(pl.col('authorid').is_not_null())
           .with_columns(year=pl.col('year').cast(pl.Int16))
           .join(fulltext, on=key, how='left')
           .with_columns(pl.col('fulltext').fill_null(False)))
    assert out.height == n_elements - n_null_ids, (
        f'exploded {out.height:,} rows from {n_elements:,} author-list elements '
        f'less {n_null_ids:,} null ids')
    if n_null_ids:
        print(f'    {n_null_ids:,} null author ids in non-empty bylines '
              f'({100 * n_null_ids / max(n_elements, 1):.3f}%)', file=sys.stderr)
    return out.select(key, 'authorid', 'author_pos', 'n_authors', 'year',
                      'field', 'fulltext')


def build_crosswalk(so: dict, client, c: Corpus = S2ORC,
                    release_id: str = S2_RELEASE_ID, n_parts: int | None = None,
                    local_out: str | None = None, overwrite: bool = False) -> int:
    """Write one crosswalk part per papers part. Peak memory = one part.

    Restartable at part granularity, the same contract stage 1/2 get from their
    `.done` markers: an already-written part is skipped, so re-running the SLURM
    script to redo only the panel step costs a listing rather than 7e8 rows.
    Pass `overwrite=True` after a new S2 release lands under the same id.
    """
    from accelscan.s3 import list_keys
    out_prefix = authorship_crosswalk_parts(c, release_id)
    done = set() if overwrite else {
        k.rsplit('/', 1)[-1]
        for k in list_keys(out_prefix + '/', suffix='.parquet', client=client)}
    keys = list_keys(PAPERS_PREFIX, suffix='.parquet', client=client)
    if n_parts:
        keys = keys[:n_parts]
    if len(done) >= len(keys):
        print(f'crosswalk already complete: {len(done)} parts under {out_prefix}',
              file=sys.stderr)
        return 0
    fulltext = _fulltext_ids(so, c)
    print(f'{fulltext.height:,} full-text papers | {len(done)} parts already written',
          file=sys.stderr)

    total = 0
    for i, k in enumerate(keys):
        if f'part_{i:04d}.parquet' in done:
            continue
        part = pl.read_parquet(s3_uri(k), storage_options=so,
                               columns=[c.key, 'year', 'authors', 's2fieldsofstudy'])
        x = explode_authors(part, fulltext, key=c.key)
        del part
        name = f'part_{i:04d}.parquet'
        if local_out:
            from pathlib import Path
            d = Path(local_out); d.mkdir(parents=True, exist_ok=True)
            x.write_parquet(d / name)
        else:
            buf = io.BytesIO(); x.write_parquet(buf)
            client.put_object(Bucket=BUCKET, Key=f'{out_prefix}/{name}',
                              Body=buf.getvalue())
        total += x.height
        print(f'  [crosswalk] {i+1}/{len(keys)} parts, {x.height:,} rows '
              f'({total:,} total)', file=sys.stderr)
        del x
    return total


# ===========================================================================
# step 2: author-year panel
# ===========================================================================

def _iter_parts(paths: list[str], so: dict, cols: list[str], label: str):
    """Read crosswalk parts one at a time. `paths` are full URIs or local paths,
    so the panel step can be dev-run against a local `--local-crosswalk` dir."""
    for i, k in enumerate(paths):
        yield pl.read_parquet(k, storage_options=so, columns=cols)
        if (i + 1) % 10 == 0 or i + 1 == len(paths):
            print(f'  [{label}] {i+1}/{len(paths)} parts', file=sys.stderr)


def hash_pool(authorid: pl.Expr, rate: float, seed: int = SEED) -> pl.Expr:
    """Deterministic authorid subsample, identical across parts and reruns.

    The never-GPU control pool has to be bounded *before* per-author statistics
    are computed -- there are ~10^8 author ids and the panel only ever needs a
    few times the jumper count. Hashing the id (rather than sampling rows) keeps
    every one of a selected author's papers, which is what makes the downstream
    career statistics correct for the sampled authors.
    """
    return authorid.hash(seed=seed).mod(1_000_000) < int(rate * 1_000_000)


def author_paper_rows(paths: list[str], so: dict, authors: pl.DataFrame,
                      key: str = 'corpusid', label: str = 'rows') -> pl.DataFrame:
    """All crosswalk rows belonging to `authors` (one column: authorid)."""
    cols = [key, 'authorid', 'n_authors', 'year', 'field', 'fulltext']
    keep = [p.join(authors, on='authorid', how='semi')
            for p in _iter_parts(paths, so, cols, label)]
    return pl.concat([k for k in keep if k.height])


def author_year_panel(rows: pl.DataFrame, key: str = 'corpusid') -> pl.DataFrame:
    """(author, paper) rows joined to per-paper access/outcomes -> author-year.

    `rows` must already carry `accel_used`, `reported_fp32_gflops`,
    `reported_vram_gb`, `reported_tensor_gflops`, `any_datacenter`,
    `any_consumer`, `i_3`. The panel is **densified** over each author's active
    span so zero-output years exist -- without that, the event study would
    silently drop the very years the effect should show up in.
    """
    rows = rows.filter(pl.col('year').is_not_null())
    used = pl.col('accel_used').fill_null(False)

    ay = rows.group_by('authorid', 'year').agg(
        n_papers=pl.len(),
        n_fulltext=pl.col('fulltext').fill_null(False).sum(),
        n_accel=used.sum(),
        n_solo=(pl.col('n_authors') == 1).sum(),
        cit_i3_total=pl.col('i_3').sum(),
        cit_i3_mean=pl.col('i_3').mean(),
        access_fp32_max=pl.col('reported_fp32_gflops').filter(used).max(),
        access_vram_max=pl.col('reported_vram_gb').filter(used).max(),
        access_tensor_max=pl.col('reported_tensor_gflops').filter(used).max(),
        access_frac_sum=(pl.col('reported_fp32_gflops').filter(used)
                         / pl.col('n_authors').filter(used).cast(pl.Float64)).sum(),
        any_datacenter=pl.col('any_datacenter').filter(used).fill_null(False).any(),
        any_consumer=pl.col('any_consumer').filter(used).fill_null(False).any(),
    ).with_columns(n_nonaccel=pl.col('n_papers') - pl.col('n_accel'))

    span = ay.group_by('authorid').agg(
        y0=pl.col('year').min().cast(pl.Int32),
        y1=pl.col('year').max().cast(pl.Int32),
        n_career_papers=pl.col('n_papers').sum(),
        max_papers_year=pl.col('n_papers').max(),
        ever_gpu=(pl.col('n_accel').sum() > 0),
    )
    span = span.join(
        rows.filter(pl.col('field').is_not_null())
        .group_by('authorid')
        .agg(modal_field=pl.col('field').mode().sort().first()),
        on='authorid', how='left')

    grid = (span.select('authorid', year=pl.int_ranges('y0', pl.col('y1') + 1))
            .explode('year', empty_as_null=True).with_columns(pl.col('year').cast(pl.Int16)))
    panel = (grid.join(ay, on=['authorid', 'year'], how='left')
             .join(span, on='authorid', how='left')
             .with_columns(
                 *[pl.col(cc).fill_null(0) for cc in
                   ('n_papers', 'n_fulltext', 'n_accel', 'n_nonaccel', 'n_solo',
                    'cit_i3_total', 'access_frac_sum')],
                 *[pl.col(cc).fill_null(False) for cc in
                   ('any_datacenter', 'any_consumer')],
                 career_year=(pl.col('year').cast(pl.Int32) - pl.col('y0') + 1))
             .sort('authorid', 'year'))

    # Running-max access stock (no decay: hardware bought stays available). Nulls
    # mean "no reported access", never zero, so the cum-max runs over a 0-filled
    # copy and the zero is mapped back to null.
    stock = pl.col('access_fp32_max').fill_null(0.0).cum_max().over('authorid')
    w3 = (pl.col('access_fp32_max').fill_null(0.0)
          .rolling_max(window_size=3, min_samples=1).over('authorid'))
    return panel.with_columns(
        access_fp32_stock=pl.when(stock > 0).then(stock).otherwise(None),
        access_fp32_stock_w3=pl.when(w3 > 0).then(w3).otherwise(None),
    )


def apply_activity_filters(panel: pl.DataFrame) -> pl.DataFrame:
    """Drop short careers, careers with no post-period, and merged profiles."""
    before = panel['authorid'].n_unique()
    keep = (panel.group_by('authorid').agg(pl.first('n_career_papers'),
                                           pl.first('max_papers_year'),
                                           pl.first('y0'), pl.first('y1'))
            .filter((pl.col('n_career_papers') >= MIN_CAREER_PAPERS)
                    & (pl.col('y1') - pl.col('y0') + 1 >= MIN_CAREER_SPAN)
                    & (pl.col('y0') <= MAX_FIRST_YEAR)
                    & (pl.col('max_papers_year') <= MAX_PAPERS_PER_YEAR)))
    out = panel.join(keep.select('authorid'), on='authorid', how='semi')
    print(f'activity filters: {before:,} -> {keep.height:,} authors', file=sys.stderr)
    return out


def find_jumps(panel: pl.DataFrame, jump_ratio: float = JUMP_RATIO) -> pl.DataFrame:
    """First qualifying event per (author, event_type).

    - `ratio_jump` (primary): reported FP32 this year is >= `jump_ratio` x the
      author's prior career max. Requires prior access, so it is a *jump*, not
      adoption.
    - `dc_first`: first year a datacenter-segment part appears (procurement-
      driven, so less anticipatory than a ratio jump).
    - `gpu_first`: first accelerator-using paper (adoption).
    """
    p = panel.sort('authorid', 'year')
    prior = pl.col('access_fp32_stock').shift(1).over('authorid')
    p = p.with_columns(
        prior_stock=prior,
        prior_dc=pl.col('any_datacenter').shift(1).cum_max().over('authorid'),
        prior_accel=pl.col('n_accel').shift(1).cum_max().over('authorid'),
    )
    cond = {
        'ratio_jump': (pl.col('access_fp32_max').is_not_null()
                       & pl.col('prior_stock').is_not_null()
                       & (pl.col('access_fp32_max')
                          >= jump_ratio * pl.col('prior_stock'))),
        'dc_first': (pl.col('any_datacenter')
                     & ~pl.col('prior_dc').fill_null(False)),
        'gpu_first': ((pl.col('n_accel') > 0)
                      & (pl.col('prior_accel').fill_null(0) == 0)),
    }
    out = []
    for et in EVENT_TYPES:
        e = (p.filter(cond[et]).group_by('authorid').agg(
                event_year=pl.col('year').min())
             .join(p.select('authorid', 'year', 'access_fp32_max', 'prior_stock',
                            'career_year', 'modal_field'),
                   left_on=['authorid', 'event_year'], right_on=['authorid', 'year'],
                   how='left')
             .with_columns(
                 event_type=pl.lit(et),
                 event_fp32=pl.col('access_fp32_max'),
                 pre_max_fp32=pl.col('prior_stock'),
                 jump_ratio=pl.col('access_fp32_max') / pl.col('prior_stock'),
                 career_year_at_event=pl.col('career_year'),
                 field=pl.col('modal_field'))
             .select('authorid', 'event_type', 'event_year', 'pre_max_fp32',
                     'event_fp32', 'jump_ratio', 'field', 'career_year_at_event'))
        out.append(e)
    jumps = pl.concat(out)

    pre = pre_papers_mean(panel, jumps.select('authorid', 'event_year').unique())
    return jumps.join(pre, on=['authorid', 'event_year'], how='left')


def pre_papers_mean(panel: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    """Mean n_papers over the baseline window tau in BASE_TAUS, per (author, event_year).

    This is both a matching covariate and the demeaning baseline, so it excludes
    tau=-1 exactly as `event_study` does.
    """
    lo, hi = min(BASE_TAUS), max(BASE_TAUS)
    return (events.join(panel.select('authorid', 'year', 'n_papers'),
                        on='authorid', how='inner')
            .with_columns(tau=pl.col('year').cast(pl.Int32)
                          - pl.col('event_year').cast(pl.Int32))
            .filter(pl.col('tau').is_between(lo, hi))
            .group_by('authorid', 'event_year')
            .agg(pre_papers_mean=pl.col('n_papers').mean(),
                 n_pre_years=pl.len()))


def jump_papers(rows: pl.DataFrame, jumps: pl.DataFrame, panel: pl.DataFrame,
                key: str = 'corpusid') -> pl.DataFrame:
    """Per jump: the papers that carried it, team size, and coauthor provenance.

    `coauthor_sourced` is the load-bearing diagnostic for "the treatment is
    really a well-equipped collaborator": it is true when some coauthor of a
    jump paper already held at least half the jump's capacity in the prior year.
    Every coauthor of an accelerator-using paper is in the panel by construction
    (the panel's GPU-ever population is defined as exactly those authors), so
    this comparison is exact rather than a lower bound.
    """
    ev = jumps.select('authorid', 'event_type', 'event_year', 'event_fp32')
    mine = (rows.filter(pl.col('accel_used').fill_null(False))
            .select('authorid', key, 'year', 'n_authors', 'reported_fp32_gflops')
            .join(ev, on='authorid', how='inner')
            .filter(pl.col('year') == pl.col('event_year')))
    # the papers at the author's max for that year are the ones that carried it
    mine = mine.filter(
        pl.col('reported_fp32_gflops').is_null().not_()
        & (pl.col('reported_fp32_gflops')
           >= pl.col('event_fp32').fill_null(0) - 1e-9)
        | pl.col('event_fp32').is_null())

    agg = mine.group_by('authorid', 'event_type').agg(
        jump_corpusids=pl.col(key).unique().sort(),
        n_jump_papers=pl.col(key).n_unique(),
        min_team_size=pl.col('n_authors').min())

    stock = panel.select('authorid', 'year', 'access_fp32_stock')
    co = (mine.select('authorid', 'event_type', 'event_year', key, 'event_fp32')
          .join(rows.select(key, 'authorid').unique().rename({'authorid': 'coauthorid'}),
                on=key, how='inner')
          .filter(pl.col('coauthorid') != pl.col('authorid'))
          .with_columns(prior_year=(pl.col('event_year').cast(pl.Int32) - 1)
                        .cast(pl.Int16))
          .join(stock.rename({'authorid': 'coauthorid', 'year': 'prior_year'}),
                on=['coauthorid', 'prior_year'], how='left')
          .group_by('authorid', 'event_type').agg(
              coauthor_prior_max=pl.col('access_fp32_stock').max(),
              event_fp32=pl.col('event_fp32').first()))
    co = co.with_columns(
        coauthor_sourced=(pl.col('coauthor_prior_max')
                          >= pl.col('event_fp32') / 2).fill_null(False)
    ).drop('event_fp32')
    return agg.join(co, on=['authorid', 'event_type'], how='left').with_columns(
        pl.col('coauthor_sourced').fill_null(False))


# ===========================================================================
# event study
# ===========================================================================

def _career_bin(e: pl.Expr) -> pl.Expr:
    return (e.clip(1, 40) // 5).cast(pl.Int32)


def match_controls(panel: pl.DataFrame, jumps: pl.DataFrame, *,
                   variant: str = 'riskset', n_controls: int = N_CONTROLS,
                   seed: int = SEED) -> pl.DataFrame:
    """(authorid, event_year, control_id) pairs; pseudo-event = jumper's year.

    `riskset` — GPU-ever authors who have NOT yet jumped by the jumper's event
    year (never jump, or jump strictly later). This is the defensible comparison:
    everyone in it is on an accelerator-using trajectory.

    `plus_never_gpu` — the same pool plus never-GPU authors from
    accelerator-active fields. Read as an upper bound: never-GPU authors differ
    from jumpers on everything, not just timing.

    Strata: field x career-age bin x pre-period output quartile, with quartiles
    cut on the jumpers' own distribution so the bins are comparable.
    """
    if variant not in ('riskset', 'plus_never_gpu'):
        raise ValueError(f'unknown control variant {variant!r}')

    author = panel.group_by('authorid').agg(
        pl.first('modal_field'), pl.first('ever_gpu'), pl.first('y0'), pl.first('y1'),
        pl.first('control_sample') if 'control_sample' in panel.columns
        else pl.lit('gpu_ever').alias('control_sample'))

    jumped = jumps.select('authorid', jumped_year=pl.col('event_year'))
    pool = author.join(jumped, on='authorid', how='left')
    pool = pool.filter(pl.col('ever_gpu') if variant == 'riskset'
                       else pl.lit(True))

    # A control needs the whole baseline window and at least one post year, so
    # eligibility depends on the pseudo-event year. Build the candidate x
    # event-year grid one event year at a time: event years number in the dozens,
    # so this is far smaller than a cross join.
    lo = min(BASE_TAUS)
    years = sorted(jumps['event_year'].drop_nulls().unique().to_list())
    frames = []
    for y in years:
        elig = pool.filter(
            (pl.col('y0') <= y + lo) & (pl.col('y1') >= y + 1)
            & (pl.col('jumped_year').is_null() | (pl.col('jumped_year') > y)))
        if elig.height:
            frames.append(elig.with_columns(event_year=pl.lit(y, dtype=pl.Int16)))
    if not frames:
        return pl.DataFrame(schema={'authorid': pl.Utf8, 'event_year': pl.Int16,
                                    'control_id': pl.Utf8})
    elig = pl.concat(frames)

    epm = pre_papers_mean(panel, elig.select('authorid', 'event_year'))
    elig = (elig.join(epm, on=['authorid', 'event_year'], how='inner')
            .filter(pl.col('n_pre_years') >= 2))

    jj = jumps.select('authorid', 'event_year', 'field', 'career_year_at_event',
                      'pre_papers_mean').drop_nulls('pre_papers_mean')
    cuts = jj['pre_papers_mean'].quantile
    qs = [cuts(q) for q in (0.25, 0.5, 0.75)]

    def strat(df: pl.DataFrame, field: str, career: str, ppm: str) -> pl.DataFrame:
        return df.with_columns(
            stratum=pl.concat_str([
                pl.col(field).fill_null('?'),
                _career_bin(pl.col(career).cast(pl.Int32)).cast(pl.Utf8),
                sum((pl.col(ppm) > q).cast(pl.Int32) for q in qs).cast(pl.Utf8),
            ], separator='|'))

    jj = strat(jj, 'field', 'career_year_at_event', 'pre_papers_mean')
    elig = elig.with_columns(
        career_at=(pl.lit(1) + pl.col('event_year').cast(pl.Int32) - pl.col('y0')))
    elig = strat(elig, 'modal_field', 'career_at', 'pre_papers_mean')

    rng = np.random.default_rng(seed)
    pairs = []
    by_stratum = {(k[0], k[1]): g for k, g in
                  elig.group_by(['stratum', 'event_year'], maintain_order=True)}
    for (st, y), g in jj.group_by(['stratum', 'event_year'], maintain_order=True):
        cands = by_stratum.get((st, y))
        if cands is None or not cands.height:
            continue
        ids = cands['authorid'].to_numpy()
        for a in g['authorid'].to_list():
            take = ids[ids != a]
            if not take.size:
                continue
            pick = rng.choice(take, size=min(n_controls, take.size), replace=False)
            pairs += [{'authorid': a, 'event_year': y, 'control_id': str(p)}
                      for p in pick]
    if not pairs:
        return pl.DataFrame(schema={'authorid': pl.Utf8, 'event_year': pl.Int16,
                                    'control_id': pl.Utf8})
    return pl.DataFrame(pairs).with_columns(pl.col('event_year').cast(pl.Int16))


def _demeaned(panel: pl.DataFrame, events: pl.DataFrame, outcome: str,
              id_col: str = 'authorid') -> pl.DataFrame:
    """(id_col, event_year, tau, y_tilde): outcome minus own baseline mean."""
    d = (events.join(panel.select(id_col, 'year', outcome), on=id_col, how='inner')
         .with_columns(tau=pl.col('year').cast(pl.Int32)
                       - pl.col('event_year').cast(pl.Int32))
         .filter(pl.col('tau').is_between(TAU_MIN, TAU_MAX)))
    base = (d.filter(pl.col('tau').is_in(list(BASE_TAUS)))
            .group_by(id_col, 'event_year')
            .agg(base=pl.col(outcome).mean()))
    return (d.join(base, on=[id_col, 'event_year'], how='inner')
            .with_columns(y_tilde=pl.col(outcome).cast(pl.Float64) - pl.col('base'))
            .select(id_col, 'event_year', 'tau', 'y_tilde'))


def add_outcomes(panel: pl.DataFrame) -> pl.DataFrame:
    return panel.with_columns(
        log_papers=(1 + pl.col('n_papers')).cast(pl.Float64).log())


def event_study(panel: pl.DataFrame, jumps: pl.DataFrame,
                controls: pl.DataFrame, outcome: str, *,
                b: int = BOOTSTRAP_B, seed: int = SEED) -> pl.DataFrame:
    """theta_tau = mean over jumpers of (own-demeaned outcome - matched control mean).

    Bootstrap is a **block** bootstrap over jumpers (a resampled jumper brings its
    own controls), because the jumper is the unit of independence.

    Mechanical caveat at tau=0: the jump *is* a paper, so output there is >=1 by
    construction. Read tau>=1, and read `n_nonaccel` as the least contaminated
    outcome.
    """
    if outcome not in panel.columns:
        raise ValueError(f'{outcome!r} not in panel; call add_outcomes first')
    ev = jumps.select('authorid', 'event_year').unique()
    treated = _demeaned(panel, ev, outcome)

    cev = controls.select(target='authorid', control_id='control_id',
                          event_year='event_year')
    cd = _demeaned(panel, cev.select(authorid='control_id', event_year='event_year')
                   .unique(), outcome)
    cm = (cev.join(cd.rename({'authorid': 'control_id'}),
                   on=['control_id', 'event_year'], how='inner')
          .group_by('target', 'event_year', 'tau')
          .agg(ctrl=pl.col('y_tilde').mean()))

    d = (treated.join(cm.rename({'target': 'authorid'}),
                      on=['authorid', 'event_year', 'tau'], how='inner')
         .with_columns(diff=pl.col('y_tilde') - pl.col('ctrl')))
    if not d.height:
        return pl.DataFrame(schema={'tau': pl.Int32, 'theta': pl.Float64,
                                    'lo': pl.Float64, 'hi': pl.Float64,
                                    'n': pl.Int64})

    # one row per jumper, one column per tau -- so the bootstrap can resample
    # jumpers (rows) as blocks. A tau with no support anywhere is simply absent.
    wide = d.pivot('tau', index='authorid', values='diff', aggregate_function='mean')
    present = [t for t in range(TAU_MIN, TAU_MAX + 1) if str(t) in wide.columns]
    m = wide.select([str(t) for t in present]).to_numpy().astype(float)

    rng = np.random.default_rng(seed)
    n = (~np.isnan(m)).sum(axis=0)
    boots = np.empty((b, m.shape[1]))
    # A tau can have no support at all (thin tails of the window), and a bootstrap
    # resample can miss every jumper that supports a given tau. Both leave an
    # all-NaN column, and NaN out is the right answer -- the alternative is
    # inventing a number for a tau nothing was observed at. So the numpy warnings
    # are suppressed rather than the reductions being guarded.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        theta = np.nanmean(m, axis=0)
        for i in range(b):
            idx = rng.integers(0, m.shape[0], m.shape[0])
            boots[i] = np.nanmean(m[idx], axis=0)
        lo = np.nanpercentile(boots, 2.5, axis=0)
        hi = np.nanpercentile(boots, 97.5, axis=0)
    return pl.DataFrame({'tau': present, 'theta': theta, 'lo': lo, 'hi': hi, 'n': n})


# ===========================================================================
# assembly
# ===========================================================================

def _accel_active_fields(so: dict, base: str) -> list[str]:
    """Fields whose accelerator prevalence clears MIN_FIELD_PREVALENCE."""
    den = pl.read_parquet(f'{s3_uri(base)}/denominator.parquet', storage_options=so,
                          columns=['corpusid', 'field'])
    flags = pl.read_parquet(f'{s3_uri(base)}/paper_flags.parquet',
                            storage_options=so, columns=['corpusid', 'accel_used'])
    d = den.join(flags, on='corpusid', how='left')
    prev = (d.group_by('field').agg(n=pl.len(),
                                    used=pl.col('accel_used').fill_null(False).sum())
            .filter((pl.col('field').is_not_null())
                    & (pl.col('used') / pl.col('n') >= MIN_FIELD_PREVALENCE)))
    return prev['field'].to_list()


def build_panel(so: dict, client, c: Corpus, release_id: str, rv: str,
                prompt_version: str, model_tag: str, count_policy: str,
                never_gpu_rate: float, jump_ratio: float,
                never_gpu_multiplier: int = 5,
                n_parts: int | None = None,
                local_crosswalk: str | None = None
                ) -> tuple[pl.DataFrame, pl.DataFrame]:
    from accelscan.s3 import list_keys
    base = analytic_base(c, rv, prompt_version, model_tag)
    if local_crosswalk:
        from pathlib import Path
        # `part_*` and not `*`: the panel's own outputs are parquet too, and a
        # dev run that points --local-crosswalk at its --local-out directory
        # would otherwise feed the jumps table back in as a crosswalk part
        keys = sorted(str(p) for p in Path(local_crosswalk).glob('part_*.parquet'))
    else:
        keys = [s3_uri(k) for k in
                list_keys(authorship_crosswalk_parts(c, release_id) + '/',
                          suffix='.parquet', client=client)]
    if not keys:
        raise FileNotFoundError(
            f'no crosswalk parts under '
            f'{local_crosswalk or authorship_crosswalk_parts(c, release_id)} '
            f'— run `--step crosswalk` first')
    if n_parts:
        keys = keys[:n_parts]

    flags = pl.read_parquet(f'{s3_uri(base)}/paper_flags.parquet',
                            storage_options=so,
                            columns=[c.key, 'accel_any', 'accel_used'])
    cap = pl.read_parquet(
        s3_uri(capacity_key(c, rv, prompt_version, model_tag, count_policy)),
        storage_options=so,
        columns=[c.key, 'reported_fp32_gflops', 'reported_vram_gb',
                 'reported_tensor_gflops', 'any_datacenter', 'any_consumer'])
    cits = pl.read_parquet(f'{s3_uri(base)}/citations.parquet', storage_options=so,
                           columns=[c.key, 'i_3'])
    print(f'flags {flags.height:,} | capacity {cap.height:,} | citations '
          f'{cits.height:,}', file=sys.stderr)

    # --- pass A: GPU-ever authors (every author of an accelerator-using paper)
    accel = flags.filter(pl.col('accel_used').fill_null(False)).select(c.key)
    gpu_ever = pl.concat([
        p.join(accel, on=c.key, how='semi').select('authorid').unique()
        for p in _iter_parts(keys, so, [c.key, 'authorid'], 'gpu-ever')
    ]).unique()
    print(f'GPU-ever authors: {gpu_ever.height:,}', file=sys.stderr)

    # --- pass B: bounded never-GPU candidate pool (hash-sampled ids)
    pool = pl.concat([
        p.filter(hash_pool(pl.col('authorid'), never_gpu_rate))
        .select('authorid').unique()
        for p in _iter_parts(keys, so, ['authorid'], 'pool')
    ]).unique().join(gpu_ever, on='authorid', how='anti')
    print(f'never-GPU candidate pool: {pool.height:,} (rate {never_gpu_rate})',
          file=sys.stderr)

    panel_authors = pl.concat([
        gpu_ever.with_columns(control_sample=pl.lit('gpu_ever')),
        pool.with_columns(control_sample=pl.lit('never_gpu')),
    ])

    # --- pass C: every paper of every panel author
    rows = author_paper_rows(keys, so, panel_authors.select('authorid'), c.key)
    print(f'author-paper rows: {rows.height:,}', file=sys.stderr)
    rows = (rows.join(flags, on=c.key, how='left')
            .join(cap, on=c.key, how='left')
            .join(cits, on=c.key, how='left'))
    # accel_used is a full-text-only signal: a paper outside the full-text corpus
    # was never scanned, so it must not be counted as non-accelerator evidence of
    # access. It still counts toward n_papers (output), which is the point.
    rows = rows.with_columns(
        accel_used=pl.col('accel_used').fill_null(False) & pl.col('fulltext'))

    panel = author_year_panel(rows, c.key)
    panel = panel.join(panel_authors, on='authorid', how='left')
    panel = apply_activity_filters(panel)

    # never-GPU controls: 5x the jumper-pool count per field x first-decade
    active = _accel_active_fields(so, base)
    panel = _trim_never_gpu(panel, active, never_gpu_multiplier, seed=SEED)

    panel = panel.with_columns(
        release_id=pl.lit(release_id), count_policy=pl.lit(count_policy))

    jumps = find_jumps(panel.filter(pl.col('control_sample') == 'gpu_ever'),
                       jump_ratio)
    jumps = jumps.join(jump_papers(rows, jumps, panel, c.key),
                       on=['authorid', 'event_type'], how='left')
    jumps = jumps.with_columns(count_policy=pl.lit(count_policy),
                               jump_ratio_threshold=pl.lit(jump_ratio))
    _check_invariants(panel, jumps, jump_ratio)
    return panel, jumps


def _trim_never_gpu(panel: pl.DataFrame, active_fields: list[str],
                    multiplier: int = 5, seed: int = SEED) -> pl.DataFrame:
    """Keep <= `multiplier` never-GPU authors per GPU-ever author, per stratum.

    Strata are field x first-decade. Shortfalls are printed rather than silently
    tolerated: a stratum with too few never-GPU candidates makes control variant
    (b) unbalanced exactly where it matters.
    """
    a = panel.group_by('authorid').agg(
        pl.first('modal_field'), pl.first('y0'), pl.first('control_sample')
    ).with_columns(decade=(pl.col('y0') // 10) * 10)
    target = (a.filter(pl.col('control_sample') == 'gpu_ever')
              .group_by('modal_field', 'decade').agg(need=multiplier * pl.len()))
    cands = a.filter((pl.col('control_sample') == 'never_gpu')
                     & pl.col('modal_field').is_in(active_fields))
    rng = np.random.default_rng(seed)
    keep, short = [], 0
    tmap = {(r['modal_field'], r['decade']): r['need'] for r in target.iter_rows(named=True)}
    for k, g in cands.group_by(['modal_field', 'decade']):
        need = tmap.get((k[0], k[1]), 0)
        if not need:
            continue
        ids = g['authorid'].to_numpy()
        if ids.size < need:
            short += 1
        pick = rng.choice(ids, size=min(need, ids.size), replace=False)
        keep.append(pl.DataFrame({'authorid': [str(x) for x in pick]}))
    kept = (pl.concat(keep) if keep
            else pl.DataFrame(schema={'authorid': pl.Utf8}))
    n_gpu = a.filter(pl.col('control_sample') == 'gpu_ever').height
    print(f'never-GPU controls kept: {kept.height:,} for {n_gpu:,} GPU-ever authors '
          f'({kept.height / max(n_gpu, 1):.2f}x of the {multiplier}x target; '
          f'{short} of {target.height} strata short)', file=sys.stderr)
    return pl.concat([
        panel.filter(pl.col('control_sample') == 'gpu_ever'),
        panel.filter(pl.col('control_sample') != 'gpu_ever')
        .join(kept, on='authorid', how='semi'),
    ]).sort('authorid', 'year')


def _check_invariants(panel: pl.DataFrame, jumps: pl.DataFrame,
                      jump_ratio: float) -> None:
    """Fail loudly rather than publish a panel that cannot be right."""
    assert panel.select(pl.struct('authorid', 'year').is_duplicated().any()).item() \
        is False, 'duplicate (authorid, year) in the panel'
    assert panel.filter(pl.col('access_fp32_max') == 0).height == 0, \
        'access must be null, never 0, when there is no reported access'
    assert panel.filter(pl.col('n_accel') > pl.col('n_fulltext')).height == 0, \
        'n_accel exceeds n_fulltext'
    assert panel.filter(pl.col('n_fulltext') > pl.col('n_papers')).height == 0, \
        'n_fulltext exceeds n_papers'
    # stock is a running max, so it can never fall within an author
    fell = panel.sort('authorid', 'year').filter(
        pl.col('access_fp32_stock')
        < pl.col('access_fp32_stock').shift(1).over('authorid'))
    assert fell.height == 0, f'access stock fell for {fell.height} author-years'
    bad = jumps.filter((pl.col('event_type') == 'ratio_jump')
                       & (pl.col('jump_ratio') < jump_ratio - 1e-9))
    assert bad.height == 0, f'{bad.height} ratio_jumps below the threshold'
    print(f'invariants ok: {panel.height:,} author-years, '
          f'{panel["authorid"].n_unique():,} authors, {jumps.height:,} events',
          file=sys.stderr)


def _put(client, df: pl.DataFrame, key: str) -> None:
    buf = io.BytesIO(); df.write_parquet(buf)
    client.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())
    print(f'wrote s3://{BUCKET}/{key} ({df.height:,} rows)', file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', required=True, choices=['crosswalk', 'panel'])
    ap.add_argument('--model-tag', default='qwen3-14b')
    ap.add_argument('--prompt-version', default='p1')
    ap.add_argument('--count-policy', default='floor1', choices=['floor1', 'explicit'])
    ap.add_argument('--release-id', default=S2_RELEASE_ID)
    ap.add_argument('--jump-ratio', type=float, default=JUMP_RATIO)
    # The pool is hash-bounded before per-author statistics exist, so the rate
    # has to be set high enough to reach the multiplier target: the release holds
    # ~9e7 distinct author ids against ~3e6 GPU-ever authors, so 5x coverage needs
    # roughly 15e6/9e7 ~ 0.17 before the accelerator-active-field restriction
    # takes its cut. Lower it if the panel pass runs short of memory -- the cost
    # is only that control arm (b), already an upper bound, gets thinner.
    ap.add_argument('--never-gpu-rate', type=float, default=0.20,
                    help='authorid hash-sample rate bounding the never-GPU pool')
    ap.add_argument('--never-gpu-multiplier', type=int, default=5,
                    help='never-GPU controls kept per GPU-ever author, per stratum')
    ap.add_argument('--n-parts', type=int, help='dev: cap parts read')
    ap.add_argument('--local-crosswalk',
                    help='dev: read crosswalk parts from a local directory')
    ap.add_argument('--overwrite', action='store_true',
                    help='crosswalk: rewrite parts that already exist')
    ap.add_argument('--local-out')
    ap.add_argument('--corpus', default='s2orc', choices=['s2orc'],
                    help='arXiv carries no author ids, only name strings')
    args = ap.parse_args()
    c = get_corpus(args.corpus)

    from accelscan.s3 import make_s3_client, storage_options
    so, client = storage_options(), make_s3_client()

    if args.step == 'crosswalk':
        n = build_crosswalk(so, client, c, args.release_id, args.n_parts,
                            args.local_out, args.overwrite)
        print(f'crosswalk: {n:,} (paper, author) rows', file=sys.stderr)
        return

    from accelscan.registry import load_registry
    rv = load_registry().version
    panel, jumps = build_panel(so, client, c, args.release_id, rv,
                               args.prompt_version, args.model_tag,
                               args.count_policy, args.never_gpu_rate,
                               args.jump_ratio, args.never_gpu_multiplier,
                               args.n_parts, args.local_crosswalk)
    outputs = {f'author_year_{args.count_policy}': panel,
               f'author_jumps_{args.count_policy}': jumps}
    if args.local_out:
        from pathlib import Path
        d = Path(args.local_out); d.mkdir(parents=True, exist_ok=True)
        for name, df in outputs.items():
            df.write_parquet(d / f'{name}.parquet')
            print(f'wrote {d}/{name}.parquet ({df.height:,})', file=sys.stderr)
    else:
        for name, df in outputs.items():
            _put(client, df, author_analytic_key(c, rv, args.prompt_version,
                                                 args.model_tag, name))


if __name__ == '__main__':
    main()

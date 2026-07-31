"""arXiv category taxonomy: primary_category -> field label and display name.

Reads `registry/arxiv_categories.yaml`, which is *generated* from
https://arxiv.org/category_taxonomy by `scripts/build_arxiv_taxonomy.py` (same
convention as the hardware registry: scraped, not hand-authored). Three tables:

- `groups`     — group -> archives, e.g. Physics -> [astro-ph, cond-mat, hep-th, ...]
- `legacy`     — pre-2007 archive -> the modern archive that absorbed it
                 (`cmp-lg` -> `cs`), so a 1995 paper lands in the same field as its
                 modern equivalent instead of dropping out
- `categories`  — subject class -> display name (`cs.LG` -> "Machine Learning"),
                 for figure labels in the fine-grained breakdowns
"""

from functools import lru_cache
from pathlib import Path

import yaml

CATEGORIES_PATH = Path('registry/arxiv_categories.yaml')


@lru_cache(maxsize=4)
def load_taxonomy(path: str | Path = CATEGORIES_PATH) -> dict:
    """Parsed taxonomy with legacy archives resolved to their modern group."""
    spec = yaml.safe_load(Path(path).read_text())
    archive_to_group = {archive: group
                        for group, archives in spec['groups'].items()
                        for archive in archives}
    legacy = spec.get('legacy') or {}
    for old, modern in legacy.items():
        if modern not in archive_to_group:
            raise ValueError(f'legacy archive {old!r} maps to unknown archive {modern!r}')
        archive_to_group.setdefault(old, archive_to_group[modern])
    return {'version': spec.get('version'), 'groups': spec['groups'],
            'legacy': legacy, 'categories': spec.get('categories') or {},
            'archive_to_group': archive_to_group}


def load_category_map(path: str | Path = CATEGORIES_PATH) -> dict[str, str]:
    """archive -> field label, legacy archives included."""
    return load_taxonomy(path)['archive_to_group']


def archive_of(primary_category: str) -> str:
    """'cs.LG' -> 'cs'; 'hep-th' -> 'hep-th'; 'astro-ph.GA' -> 'astro-ph'."""
    return (primary_category or '').strip().split('.')[0]


def field_of(primary_category: str, mapping: dict[str, str] | None = None) -> str | None:
    """Field label (arXiv group) for a primary category, or None if unknown.

    None rather than an 'Other' bucket on purpose: an unmapped archive is a gap in
    the generated taxonomy that `tests/test_arxiv_meta.py` should catch, not
    something to paper over in a published figure.
    """
    mapping = mapping if mapping is not None else load_category_map()
    return mapping.get(archive_of(primary_category))


def category_name(primary_category: str, path: str | Path = CATEGORIES_PATH) -> str | None:
    """'cs.LG' -> 'Machine Learning'. Falls back to the code's own display name."""
    cats = load_taxonomy(path)['categories']
    return cats.get((primary_category or '').strip())


def primary_category_of(categories: str | list[str]) -> str | None:
    """First listed category is arXiv's primary one.

    The Kaggle snapshot stores `categories` as a space-separated string
    ('cs.LG cs.AI stat.ML'), not a list.
    """
    parts = categories.split() if isinstance(categories, str) else list(categories or [])
    return parts[0] if parts else None

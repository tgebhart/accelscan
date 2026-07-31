"""arXiv category taxonomy: primary_category -> field label.

Loaded from `registry/arxiv_categories.yaml` (data, not code -- same convention as
the hardware registry). Kept separate from `arxiv_source` so that
`scripts/build_arxiv_metadata.py`, `meta.py` and the notebooks share one mapping.
"""

from functools import lru_cache
from pathlib import Path

import yaml

CATEGORIES_PATH = Path('registry/arxiv_categories.yaml')


@lru_cache(maxsize=1)
def load_category_map(path: str | Path = CATEGORIES_PATH) -> dict[str, str]:
    """archive -> field label, e.g. {'cs': 'Computer Science', 'hep-th': 'Physics'}."""
    spec = yaml.safe_load(Path(path).read_text())
    return {archive: group
            for group, archives in spec['groups'].items()
            for archive in archives}


def archive_of(primary_category: str) -> str:
    """'cs.LG' -> 'cs'; 'hep-th' -> 'hep-th'; 'math.AG' -> 'math'."""
    return (primary_category or '').strip().split('.')[0]


def field_of(primary_category: str, mapping: dict[str, str] | None = None) -> str | None:
    """arXiv field label for a primary category, or None if the archive is unknown.

    None rather than a fallback bucket on purpose: an unmapped archive is a gap in
    `registry/arxiv_categories.yaml` that a test should catch, not something to
    paper over with 'Other' in a published figure.
    """
    mapping = mapping if mapping is not None else load_category_map()
    return mapping.get(archive_of(primary_category))


def primary_category_of(categories: str | list[str]) -> str | None:
    """First listed category is arXiv's primary one.

    The Kaggle snapshot stores `categories` as a space-separated string
    ('cs.LG cs.AI stat.ML'), not a list.
    """
    if isinstance(categories, str):
        parts = categories.split()
    else:
        parts = list(categories or [])
    return parts[0] if parts else None

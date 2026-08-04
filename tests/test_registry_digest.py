"""The registry version must cover the generated files, not just the hand file."""

from accelscan.scripts.registry_digest import compute, load


def test_digest_matches_committed():
    cur, rec = compute(), load()
    assert rec is not None, ('registry/digest.json is missing; run '
                            'python -m accelscan.scripts.registry_digest --write')
    assert rec['version'] == cur['version'], (
        f'digest pins v{rec["version"]} but hardware.yaml says '
        f'v{cur["version"]}: re-pin with registry_digest --write')
    changed = [f for f, h in cur['files'].items() if rec['files'].get(f) != h]
    assert not changed, (
        f'registry content changed without a version bump: {changed}. Matching '
        f'may differ from what v{rec["version"]} outputs were built with -- bump '
        f'`version` in registry/hardware.yaml and re-run registry_digest --write.')
    assert set(rec['files']) == set(cur['files']), 'registry file set changed'

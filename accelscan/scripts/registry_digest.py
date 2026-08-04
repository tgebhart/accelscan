"""Pin the registry's content to its version.

`version` lives in `registry/hardware.yaml`, but matching is decided by that file
*plus* `registry/generated/*.yaml` -- so a re-scrape could change every alias
without touching the version, and version-namespaced outputs would then mix two
registries under one name. This writes `registry/digest.json` (version + a sha256
per registry file), and `tests/test_registry_digest.py` fails on any content
change the version does not acknowledge.

    python -m accelscan.scripts.registry_digest            # verify
    python -m accelscan.scripts.registry_digest --write    # after a version bump
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

REGISTRY_DIR = Path('registry')
DIGEST_PATH = REGISTRY_DIR / 'digest.json'


def registry_files(registry_dir: Path = REGISTRY_DIR) -> list[Path]:
    return [registry_dir / 'hardware.yaml',
            *sorted((registry_dir / 'generated').glob('*.yaml'))]


def compute(registry_dir: Path = REGISTRY_DIR) -> dict:
    hand = yaml.safe_load((registry_dir / 'hardware.yaml').read_text())
    return {
        'version': str(hand['version']),
        'files': {str(p.relative_to(registry_dir)):
                  hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in registry_files(registry_dir)},
    }


def load() -> dict | None:
    return json.loads(DIGEST_PATH.read_text()) if DIGEST_PATH.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', action='store_true')
    # for a change that provably cannot affect matching (provenance metadata in a
    # generated file's header, a comment) while the version is still unreleased
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    cur, rec = compute(), load()

    if not args.write:
        if cur == rec:
            print(f'registry digest ok (v{cur["version"]})')
            return
        raise SystemExit('registry digest MISMATCH: registry content changed. Bump '
                         '`version` in registry/hardware.yaml, then re-run with '
                         '--write.')
    if (rec and not args.force and rec['version'] == cur['version']
            and rec['files'] != cur['files']):
        raise SystemExit(f'refusing to re-pin v{cur["version"]}: its content '
                         f'changed, so the version must change too (matching may '
                         f'differ, and outputs are namespaced by version).')
    DIGEST_PATH.write_text(json.dumps(cur, indent=2, sort_keys=True) + '\n')
    print(f'wrote {DIGEST_PATH} (v{cur["version"]})', file=sys.stderr)


if __name__ == '__main__':
    main()

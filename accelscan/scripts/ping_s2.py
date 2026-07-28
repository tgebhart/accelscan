"""Ping the Semantic Scholar Graph API with the key from .env.

Reads S2_API_KEY from the environment, ./.env, or ~/.config/accelscan.env
(first hit wins) and fetches one known paper.

Usage: python -m accelscan.scripts.ping_s2
"""

import os
import sys
import time
from pathlib import Path

import requests

GRAPH_URL = 'https://api.semanticscholar.org/graph/v1/paper/CorpusID:2314124'
FIELDS = 'title,year,citationCount,externalIds'


def load_api_key() -> str:
    if os.environ.get('S2_API_KEY'):
        print('key source: environment')
        return os.environ['S2_API_KEY']
    for env_path in (Path('.env'), Path.home() / '.config/accelscan.env'):
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith('S2_API_KEY=') and line.split('=', 1)[1]:
                    key = line.split('=', 1)[1].strip().strip('"\'')
                    print(f'key source: {env_path} ({key[:6]}..., {len(key)} chars)')
                    return key
    sys.exit('S2_API_KEY not found in environment, ./.env, or ~/.config/accelscan.env')


def main() -> None:
    key = load_api_key()
    for attempt in range(3):
        r = requests.get(GRAPH_URL, params={'fields': FIELDS},
                         headers={'x-api-key': key}, timeout=30)
        print(f'status: {r.status_code}')
        if r.status_code != 429:
            break
        print('429 (rate limited) — retrying in 2s')
        time.sleep(2)
    print(r.json())
    r.raise_for_status()


if __name__ == '__main__':
    main()

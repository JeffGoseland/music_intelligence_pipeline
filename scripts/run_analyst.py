#!/usr/bin/env python3
"""Phase 4: natural-language question; uses Grok (x.ai) if XAI_API_KEY is set."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from src.analyst import run_analyst  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print(
            'Usage: python3 scripts/run_analyst.py "Your question"',
            file=sys.stderr,
        )
        print(
            'Example: python3 scripts/run_analyst.py "Find calm music"',
            file=sys.stderr,
        )
        return 1
    query = " ".join(sys.argv[1:])
    try:
        song_ids, explanation = run_analyst(query, max_songs=20)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1
    print(explanation)
    print()
    if song_ids:
        for sid in song_ids:
            print(sid)
    else:
        print("(No songs matched the requested tags.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

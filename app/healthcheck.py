"""Container HEALTHCHECK.

Deliberately hits /health, not /ready: the first start downloads ~1.3 GB and Docker
would kill the container mid-download if healthiness required a loaded model. Readiness
is a separate signal callers poll themselves.
"""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request


def main() -> int:
    port = os.getenv("PORT", "8000")
    url = f"http://127.0.0.1:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:  # noqa: S310 - localhost
            return 0 if response.status == 200 else 1
    except (urllib.error.URLError, OSError) as exc:
        print(f"healthcheck failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Content hash of the agent package, stored with every result.

The reference runs (arc_reference) must use exactly the agent version that the
own benchmark evaluated and that was frozen.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def agent_version() -> str:
    h = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:12]

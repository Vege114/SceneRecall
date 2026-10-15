"""Atomic publication with bounded tolerance for Windows file-sharing locks."""
from __future__ import annotations

import os
from pathlib import Path
from time import sleep

_WINDOWS = os.name == "nt"
_REPLACE_DELAYS = (0.01, 0.02, 0.04, 0.08, 0.16)


def atomic_replace(source: Path | str, target: Path | str) -> None:
    """Keep the destination intact while a Windows reader/scanner releases it."""
    for attempt in range(len(_REPLACE_DELAYS) + 1):
        try:
            os.replace(source, target)
            return
        except OSError as exc:
            if (not _WINDOWS or getattr(exc, "winerror", None) not in {5, 32, 33}
                    or attempt == len(_REPLACE_DELAYS)):
                raise
            # Permanent access denial is surfaced after at most 310 ms.
            sleep(_REPLACE_DELAYS[attempt])

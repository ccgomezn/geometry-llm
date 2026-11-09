"""Append dated entries to the results log."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "RESULTS.md"


def append_entry(title: str, body_md: str, figures: Sequence[Path] = (), path: Path = RESULTS) -> None:
    lines = [f"\n## {datetime.now():%Y-%m-%d %H:%M} · {title}\n", body_md.strip() + "\n"]
    for f in figures:
        rel = os.path.relpath(Path(f), Path(path).parent)
        lines.append(f"\n![{Path(f).stem}]({rel})\n")
    with open(path, "a") as fh:
        fh.write("".join(lines))

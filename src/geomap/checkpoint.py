"""Resumable steps for long runs on machines that can disappear (Colab), plus optional Slack progress messages.

df_step: compute a DataFrame once, save it, and read it back on later runs.
notify: post a short text to the Slack incoming webhook in $SLACK_WEBHOOK_URL; a no-op without it, never raises."""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Callable

import pandas as pd


def df_step(path, fn: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    path = Path(path)
    if path.exists():
        return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    df = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")  # write then rename: a disconnect mid-write leaves no half file
    df.to_parquet(tmp) if path.suffix == ".parquet" else df.to_csv(tmp, index=False)
    tmp.replace(path)
    return df


def notify(text: str) -> bool:
    url = os.environ.get("SLACK_WEBHOOK_URL")
    if not url:
        return False
    try:
        req = urllib.request.Request(url, data=json.dumps({"text": text}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return 200 <= r.status < 300
    except Exception as e:  # progress messages must never stop a run
        print(f"[notify] failed: {e}", flush=True)
        return False

import json

import pandas as pd

from geomap import checkpoint
from geomap.checkpoint import df_step, notify


def test_df_step_computes_once_then_reads_back(tmp_path):
    calls = []

    def fn():
        calls.append(1)
        return pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    p = tmp_path / "sub" / "step.parquet"
    a = df_step(p, fn)
    b = df_step(p, fn)
    assert len(calls) == 1
    pd.testing.assert_frame_equal(a, b)


def test_df_step_csv(tmp_path):
    p = tmp_path / "step.csv"
    df_step(p, lambda: pd.DataFrame({"a": [0.5]}))
    assert df_step(p, lambda: (_ for _ in ()).throw(AssertionError("recomputed"))).a.iloc[0] == 0.5


def test_notify_without_webhook_is_a_noop(monkeypatch):
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    assert notify("hello") is False


def test_notify_posts_text_and_never_raises(monkeypatch):
    sent = {}

    class Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout):
        sent["url"], sent["body"] = req.full_url, json.loads(req.data)
        return Resp()

    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.example/abc")
    monkeypatch.setattr(checkpoint.urllib.request, "urlopen", fake_urlopen)
    assert notify("step done") is True
    assert sent == {"url": "https://hooks.example/abc", "body": {"text": "step done"}}

    def boom(req, timeout):
        raise OSError("network down")

    monkeypatch.setattr(checkpoint.urllib.request, "urlopen", boom)
    assert notify("step done") is False

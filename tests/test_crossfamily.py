import pytest

from geomap.crossfamily import overall_cross_family, verdict_cross_family, wilson

REACHES = "reaches the answer position (the read-out ignores it)"


def test_wilson_matches_the_papers_interval():
    # 12B, paper: probe 86% of 104 held-out decisive questions -> [78%, 91%]
    p, lo, hi = wilson(89, 104)
    assert round(p, 2) == 0.86 and round(lo, 2) == 0.78 and round(hi, 2) == 0.91


def test_wilson_edges():
    assert wilson(0, 10)[1] == 0.0
    assert wilson(10, 10)[2] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        wilson(1, 0)


def test_replicates_when_all_three_hold():
    assert verdict_cross_family(0.40, REACHES, (0.86, 0.78, 0.91), (0.32, 0.24, 0.41)) == "replicates"


def test_failure_does_not_replicate_when_model_is_mostly_right():
    v = verdict_cross_family(0.62, REACHES, (0.90, 0.83, 0.94), (0.60, 0.51, 0.69))
    assert v == "the failure does not replicate"


def test_partial_when_intervals_overlap():
    v = verdict_cross_family(0.40, REACHES, (0.60, 0.50, 0.69), (0.45, 0.36, 0.55))
    assert v.startswith("partial") and "probe" in v


def test_partial_when_flow_verdict_differs():
    v = verdict_cross_family(0.30, "lost before the answer position", (0.86, 0.78, 0.91), (0.32, 0.24, 0.41))
    assert v.startswith("partial") and "flow" in v


def test_overall():
    assert overall_cross_family({"qwen3-8b": "replicates", "llama3.1-8b": "replicates"}) == \
        "the account holds across families"
    assert "qwen3-8b" in overall_cross_family({"qwen3-8b": "replicates", "llama3.1-8b": "partial (probe)"})
    assert "Qwen3-8B alone" in overall_cross_family({"qwen3-8b": "replicates"})

"""Cross-family check (docs/preregistration_cross_family.md): the preregistered criteria, per model and overall."""

import math

REACHES = "reaches the answer position (the read-out ignores it)"


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float, float]:
    """Share k/n with its Wilson 95% interval."""
    if n <= 0:
        raise ValueError("n must be positive")
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, mid - half), min(1.0, mid + half)


def verdict_cross_family(dec_acc: float, flow_verdict: str, probe: tuple, model: tuple) -> str:
    """dec_acc: clean accuracy on all decisive questions; flow_verdict: the flow rule's verdict; probe, model: (share,
    lo, hi) on the held-out decisive questions."""
    if dec_acc >= 0.5:
        return "the failure does not replicate"
    failed = []
    if not flow_verdict.startswith(REACHES):
        failed.append("flow")
    if not probe[1] > model[2]:
        failed.append("probe")
    return "replicates" if not failed else f"partial ({', '.join(failed)})"


def overall_cross_family(verdicts: dict[str, str]) -> str:
    names = {"qwen3-8b": "Qwen3-8B", "llama3.1-8b": "Llama 3.1 8B"}
    ok = [k for k, v in verdicts.items() if v == "replicates"]
    if len(verdicts) == 1:
        (k, v), = verdicts.items()
        return f"{v} (on {names.get(k, k)} alone; the other model could not be run)"
    if len(ok) == len(verdicts):
        return "the account holds across families"
    if ok:
        return "replicates in " + ", ".join(ok) + " only"
    return "does not replicate in either model"

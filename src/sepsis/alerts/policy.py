"""Versioned alert policy: a deterministic state machine, separate from the model.

States: ``monitoring``, ``alerted``, ``acknowledged``, ``cooldown`` and
``data_unavailable``.

* monitoring: count consecutive scored hours with score >= threshold; when
  the count reaches ``consecutive`` an alert is emitted -> alerted.
* alerted / acknowledged: no further alerts while the score stays above
  threshold. When it falls below threshold the policy re-arms (monitoring),
  or enters cooldown if the cooldown window opened by the alert is still
  running. Acknowledgement only records simulated review; it never feeds back
  into the model and does not change alert timing.
* cooldown: no alert until ``cooldown_h`` hours after the last emitted alert.
* data_unavailable: missing/invalid input. Never treated as low risk; the
  consecutive counter is reset and the condition is surfaced.

The same ``step`` function drives batch evaluation and the replay worker, and
its state is a plain dict so it can be persisted and recovered after restarts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

STATES = ("monitoring", "alerted", "acknowledged", "cooldown", "data_unavailable")


@dataclass(frozen=True)
class AlertPolicy:
    threshold: float
    consecutive: int = 1
    cooldown_h: int = 0

    def __post_init__(self):
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        if self.consecutive < 1 or self.cooldown_h < 0:
            raise ValueError("invalid consecutive/cooldown")

    @property
    def version(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return f"policy-{hashlib.sha256(payload.encode()).hexdigest()[:10]}"

    def to_dict(self) -> dict:
        return {**asdict(self), "policy_version": self.version}

    @classmethod
    def from_dict(cls, d: dict) -> "AlertPolicy":
        return cls(threshold=float(d["threshold"]), consecutive=int(d.get("consecutive", 1)),
                   cooldown_h=int(d.get("cooldown_h", 0)))


def initial_state() -> dict:
    return {"state": "monitoring", "consec": 0, "cooldown_until": None, "last_alert_hour": None}


def step(policy: AlertPolicy, state: dict, hour: int, score: float | None, status: str) -> tuple[dict, bool]:
    """Advance one hour. Returns (new_state, alert_emitted). Pure function."""
    s = dict(state)
    in_cooldown = s["cooldown_until"] is not None and hour < s["cooldown_until"]
    if status != "scored" or score is None or score != score:  # NaN check
        s["consec"] = 0
        s["state"] = "data_unavailable"
        return s, False
    above = score >= policy.threshold

    if s["state"] in ("alerted", "acknowledged"):
        if above:
            return s, False
        s["state"] = "cooldown" if in_cooldown else "monitoring"
        s["consec"] = 0
        return s, False

    if s["state"] == "cooldown" or (s["state"] == "data_unavailable" and in_cooldown):
        if in_cooldown:
            s["state"] = "cooldown"
            s["consec"] = 0
            return s, False
    s["state"] = "monitoring"
    s["consec"] = s["consec"] + 1 if above else 0
    if s["consec"] >= policy.consecutive:
        s["state"] = "alerted"
        s["consec"] = 0
        s["last_alert_hour"] = hour
        s["cooldown_until"] = hour + policy.cooldown_h if policy.cooldown_h > 0 else None
        return s, True
    return s, False


def acknowledge(state: dict) -> dict:
    s = dict(state)
    if s["state"] == "alerted":
        s["state"] = "acknowledged"
    return s


def run_policy(policy: AlertPolicy, hours, scores, statuses) -> list[int]:
    """Batch helper: emitted alert hours for one stay (rows in hour order)."""
    st = initial_state()
    out = []
    for h, sc, stt in zip(hours, scores, statuses):
        st, emitted = step(policy, st, int(h), None if sc != sc else float(sc), stt)
        if emitted:
            out.append(int(h))
    return out

"""
Tests for the analyzer's SOCIAL / SIGNAL section (Brain v3.5 / World W4).

The analyzer script runs its analysis at import time, so we exec only the
function-definition prefix (everything before ``import argparse``) to get
``_compute_social_metrics`` in isolation, then exercise it on synthetic logs.

Author: Karan Vasa
"""

import pathlib

import numpy as np
import pandas as pd
import pytest

_PATH = pathlib.Path("scripts/analyze_logs.py").resolve()
_PREFIX = _PATH.read_text(encoding="utf-8").split("\nimport argparse")[0]
# The script's prefix uses __file__ (to put the repo root on sys.path); provide
# it so the exec mirrors a normal import of the function definitions.
_NS: dict = {"__name__": "analyze_logs_partial", "__file__": str(_PATH)}
exec(compile(_PREFIX, str(_PATH), "exec"), _NS)
_compute_social_metrics = _NS["_compute_social_metrics"]


def _df(rows):
    return pd.DataFrame(rows)


class TestSocialMetrics:
    def test_no_signal_no_proximity_is_empty(self):
        df = _df(
            [
                {"action": "EAT", "agent_id": 1},
                {"action": "MOVE_FORWARD", "agent_id": 1},
            ]
        )
        out = _compute_social_metrics(df)
        assert out["signal_actions"] == 0
        assert "proximity_response" not in out

    def test_signal_count_and_rate(self):
        rows = [{"action": "SIGNAL", "agent_id": 1}] * 3 + [
            {"action": "WAIT", "agent_id": 1}
        ] * 7
        out = _compute_social_metrics(_df(rows))
        assert out["signal_actions"] == 3
        assert out["signal_rate_pct"] == pytest.approx(30.0)
        assert out["signallers"] == 1

    def test_signal_entropy_even_vs_concentrated(self):
        # Even: two agents signal equally → entropy ~1.0 normalised
        even = _df(
            [{"action": "SIGNAL", "agent_id": 1}] * 5
            + [{"action": "SIGNAL", "agent_id": 2}] * 5
        )
        out_even = _compute_social_metrics(even)
        assert out_even["signal_entropy_norm"] == pytest.approx(1.0, abs=1e-6)

        # Concentrated: one agent does almost all signalling → low entropy
        conc = _df(
            [{"action": "SIGNAL", "agent_id": 1}] * 19
            + [{"action": "SIGNAL", "agent_id": 2}] * 1
        )
        out_conc = _compute_social_metrics(conc)
        assert out_conc["signal_entropy_norm"] < 0.5

    def test_interaction_kind_counts_as_signal(self):
        rows = [
            {"action": "MOVE_FORWARD", "agent_id": 1, "interaction_kind": "signal"},
            {"action": "WAIT", "agent_id": 1, "interaction_kind": ""},
        ]
        out = _compute_social_metrics(_df(rows))
        assert out["signal_actions"] == 1

    def test_proximity_response_buckets(self):
        # obs_75 is nearest_agent_proximity under the v2 layout
        rows = []
        # alone (prox 0): mostly MOVE
        rows += [{"action": "MOVE_FORWARD", "agent_id": 1, "obs_75": 0.0}] * 8
        # close (prox > 0.5): mostly SIGNAL
        rows += [{"action": "SIGNAL", "agent_id": 1, "obs_75": 0.9}] * 6
        out = _compute_social_metrics(_df(rows))
        resp = {r["bucket"]: r for r in out["proximity_response"]}
        assert resp["alone"]["top_action"] == "MOVE_FORWARD"
        assert resp["close"]["top_action"] == "SIGNAL"
        assert resp["close"]["signal_pct"] == pytest.approx(100.0)
        # agents signal more in company
        assert out["mean_prox_when_signalling"] > out["mean_prox_overall"]

    def test_handles_missing_action_column(self):
        out = _compute_social_metrics(pd.DataFrame({"x": [1, 2]}))
        assert out == {"signal_actions": 0}

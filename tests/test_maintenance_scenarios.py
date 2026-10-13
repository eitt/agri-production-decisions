"""Verify physical case reuse and recovery rather than file-existence skipping."""
import json

import numpy as np
import pytest

from agriopt.model import ProblemData
from agriopt.maintenance import scenarios


@pytest.fixture
def tiny_suite(tmp_path, monkeypatch):
    config = {
        "experiment": {"instances": ["I01_micro"]},
        "maintenance": {"scenarios": {
            "interval_weeks": [2], "cost_cop_ha_pass": [250000],
            "max_idle_weeks": [None], "hard_idle_weeks": [1],
            "hard_idle_interval_weeks": 2,
        }},
    }
    input_file = tmp_path / "frozen_input.csv"
    input_file.write_text("value\n1\n", encoding="utf-8")
    data = ProblemData(
        ["a", "b"], np.array([1000.0]), np.array([0]), ["test"],
        np.array([[0.5, 0.4]]), np.array([[10.0] * 8, [12.0] * 8]),
        np.eye(2) * 0.02, np.array([2, 2]), np.array(["a", "b"]),
        np.array([0, 0]), np.ones((2, 8), dtype=bool),
        eligible=np.ones((1, 2), dtype=bool),
    )
    monkeypatch.setattr(scenarios, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(scenarios, "load_config", lambda _: config)
    monkeypatch.setattr(scenarios, "load_inputs", lambda *_: (None, None, None, None, [input_file]))
    monkeypatch.setattr(scenarios, "make_instance", lambda *_args, **_kwargs: (data, {"instance": "tiny"}))
    return config, input_file


def test_saved_physical_cases_resume_and_corrupt_case_recomputes(tiny_suite, monkeypatch):
    directory = scenarios.run_scenarios("unused.yaml", "tiny")
    progress = json.loads((directory / "progress.json").read_text())
    assert progress["complete"] and progress["completed"] == 2
    assert progress["infeasible"] == 1
    cases = sorted((directory / "cases").glob("*.json"))
    feasible = next(path for path in cases if json.loads(path.read_text())["result"]["feasible"])
    original = feasible.read_bytes()

    calls = []
    original_anchor = scenarios.economic_anchor

    def recording_anchor(*args):
        calls.append(args[1])
        return original_anchor(*args)

    monkeypatch.setattr(scenarios, "economic_anchor", recording_anchor)
    assert scenarios.run_scenarios("unused.yaml", "tiny") == directory
    assert calls == []
    assert feasible.read_bytes() == original

    saved = json.loads(feasible.read_text())
    saved["result"]["net_income_mcop"] += 1.0
    feasible.write_text(json.dumps(saved))
    scenarios.run_scenarios("unused.yaml", "tiny")
    assert calls
    assert feasible.read_bytes() == original

    feasible.write_text("{truncated")
    calls.clear()
    scenarios.run_scenarios("unused.yaml", "tiny")
    assert calls
    assert feasible.read_bytes() == original


def test_changed_inputs_reject_same_run_without_overwriting(tiny_suite):
    _, input_file = tiny_suite
    directory = scenarios.run_scenarios("unused.yaml", "tiny")
    before = (directory / "protocol.json").read_bytes()
    input_file.write_text("value\n2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="use a new run_id"):
        scenarios.run_scenarios("unused.yaml", "tiny")
    assert (directory / "protocol.json").read_bytes() == before

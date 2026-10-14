"""Durability gates: incomplete files never count as completed experiments."""
import numpy as np
import os
import pandas as pd
import pytest

from agriopt.maintenance import inputs, instances, runner
from agriopt.maintenance.persistence import atomic_json, read_json, run_lock
from agriopt.maintenance.reporting import build_report, _paired_inference
from agriopt.model import ProblemData


@pytest.fixture
def small_run(tmp_path, monkeypatch):
    data = ProblemData(["A", "B"], np.array([100.]), np.array([0]), ["M"],
                       np.array([[1., .8]]), np.full((2, 8), 1000.), np.eye(2) * .1,
                       np.array([2, 2]), np.array(["a", "b"]), np.array([0, 0]), np.ones((2, 8), dtype=bool))
    source = tmp_path / "inputs.csv"
    source.write_text("frozen input\n", encoding="utf-8")
    cfg = {"_root": str(tmp_path), "_config_path": str(tmp_path / "config.yaml"),
           "project": {"run_id": "tiny"}, "data": {},
           "maintenance": {"interval_weeks": 4, "cost_cop_ha_pass": 250000, "max_idle_weeks": None},
           "experiment": {"instances": ["I01_micro"], "methods": ["nsga2"], "seeds": [1, 2],
                          "budget_mode": "generations", "population": 4, "generations": 2, "workers": 1}}
    monkeypatch.setattr(runner, "load_config", lambda _: cfg)
    monkeypatch.setattr(inputs, "load_inputs", lambda config, root: (None, None, None, None, [source]))
    monkeypatch.setattr(instances, "make_instance", lambda *args, **kwargs: (data, {"instance": "I01_micro"}))
    return tmp_path, cfg, source


def test_atomic_replace_preserves_previous_file_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "result.json"
    atomic_json(path, {"old": 1})
    import agriopt.maintenance.persistence as persistence
    def fail(*args):
        raise OSError("simulated crash before replacement")
    monkeypatch.setattr(persistence.os, "replace", fail)
    with pytest.raises(OSError):
        atomic_json(path, {"new": 2})
    assert read_json(path) == {"old": 1}
    assert not list(tmp_path.glob("*.tmp"))


def test_run_lock_refuses_second_live_writer(tmp_path):
    with run_lock(tmp_path):
        with pytest.raises(RuntimeError, match="locked by PID"):
            with run_lock(tmp_path):
                pass
    assert not (tmp_path / ".run.lock").exists()


def test_run_resume_preserves_completed_tasks_and_immutable_inputs(small_run):
    root, cfg, source = small_run
    run = runner.run_experiment("irrelevant.yaml")
    assert runner.status(run)["all_completed"]
    assert (run / "final" / "maintenance.csv").is_file()
    states = [read_json(path) for path in sorted((run / "tasks").glob("*/status.json"))]
    checksums = [state["result_sha256"] for state in states]
    runner.run_experiment("irrelevant.yaml")
    resumed = [read_json(path) for path in sorted((run / "tasks").glob("*/status.json"))]
    assert [item["attempts"] for item in resumed] == [1, 1]
    assert [item["result_sha256"] for item in resumed] == checksums
    source.write_text("changed input\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Cannot resume"):
        runner.run_experiment("irrelevant.yaml")
    assert runner.status(run)["all_completed"]


def test_corrupt_completed_result_is_recomputed_not_skipped(small_run):
    run = runner.run_experiment("irrelevant.yaml")
    result = next((run / "tasks").glob("*/result.json"))
    result.write_text('{"truncated":', encoding="utf-8")
    assert not runner.status(run)["all_completed"]
    build_report(run)
    assert not (run / "final").exists()
    archived = next((run / "invalidated").glob("*/invalidation.json"))
    assert not read_json(archived)["complete"]
    assert read_json(archived.parent / "report.json")["complete"]
    runner.run_experiment("irrelevant.yaml")
    assert runner.status(run)["all_completed"]
    assert read_json(result.parent / "status.json")["attempts"] == 2
    assert read_json(result)["audit"]["passed"]
    preserved = list((run / "invalidated" / result.parent.name).glob("*/result.json"))
    assert len(preserved) == 1
    assert preserved[0].read_text(encoding="utf-8") == '{"truncated":'


def test_task_failure_keeps_completed_results_and_preliminary_outputs(small_run, monkeypatch):
    from agriopt.maintenance import search
    original = search.run_search
    fail_once = {2}
    def sometimes_fail(data, policy, method, seed, settings, checkpoint_path, progress=None):
        if seed in fail_once:
            fail_once.remove(seed)
            raise RuntimeError("simulated task interruption")
        return original(data, policy, method, seed, settings, checkpoint_path, progress=progress)
    monkeypatch.setattr(search, "run_search", sometimes_fail)
    with pytest.raises(RuntimeError, match="tasks failed"):
        runner.run_experiment("irrelevant.yaml")
    root, _, _ = small_run
    run = root / "outputs" / "maintenance" / "tiny"
    current = runner.status(run)
    assert current["counts"]["completed"] == 1
    assert current["counts"]["failed"] == 1
    assert (run / "preliminary" / "summary.csv").exists()
    assert not (run / "final").exists()
    runner.run_experiment("irrelevant.yaml")
    assert runner.status(run)["all_completed"]
    first = read_json(run / "tasks" / "I01_micro__nsga2__1" / "status.json")
    second = read_json(run / "tasks" / "I01_micro__nsga2__2" / "status.json")
    assert first["attempts"] == 1 and second["attempts"] == 2


def test_paired_holm_family_excludes_exploratory_proxy():
    records, results = [], {}
    for instance in ("I01_micro", "I02_small"):
        for method in ("surrogate_local", "nsga2", "random_local", "proxy_local"):
            for seed in range(4):
                value = 1.1 if method == "surrogate_local" else 1.0
                records.append({"instance": instance, "method": method, "seed": seed, "hypervolume": value})
                results[(instance, method, seed)] = {"ideal": [0, 0], "span": [1, 1], "budget_mode": "generations"}
    contrasts = _paired_inference(pd.DataFrame(records), results, list(range(4)))
    primary = [row for row in contrasts if row["primary_contrast"]]
    assert len(primary) == 4
    assert all(row["holm_family_size"] == 4 for row in primary)
    assert all(row["holm_p"] >= row["sign_flip_p"] for row in primary)
    assert all(row["holm_p"] is None for row in contrasts if not row["primary_contrast"])


def test_incomplete_paired_block_has_no_inferential_pvalue():
    records = [{"instance": "I01_micro", "method": method, "seed": seed, "hypervolume": 1.}
               for method in ("surrogate_local", "nsga2") for seed in (1, 2)]
    results = {("I01_micro", method, seed): {"ideal": [0, 0], "span": [1, 1], "budget_mode": "generations"}
               for method in ("surrogate_local", "nsga2") for seed in (1, 2)}
    row = _paired_inference(pd.DataFrame(records), results, [1, 2, 3])[0]
    assert not row["complete_seed_block"]
    assert row["sign_flip_p"] is None and row["holm_p"] is None


def test_wall_clock_restarted_block_is_descriptive():
    records = [{"instance": "I01_micro", "method": method, "seed": seed, "hypervolume": 1.}
               for method in ("surrogate_local", "nsga2") for seed in (1, 2, 3)]
    results = {("I01_micro", method, seed): {"ideal": [0, 0], "span": [1, 1], "budget_mode": "wall_clock",
                                              "restarts": int(method == "surrogate_local" and seed == 1)}
               for method in ("surrogate_local", "nsga2") for seed in (1, 2, 3)}
    row = _paired_inference(pd.DataFrame(records), results, [1, 2, 3])[0]
    assert row["complete_seed_block"] and not row["uninterrupted_timed_block"]
    assert row["sign_flip_p"] is None and row["bootstrap_ci_low"] is None


def test_resuming_refuses_a_live_orphan_worker(small_run):
    run = runner.run_experiment("irrelevant.yaml")
    path = next((run / "tasks").glob("*/status.json"))
    state = read_json(path)
    state.update(status="running", pid=os.getpid())
    atomic_json(path, state)
    (path.parent / "completion.json").unlink()
    with pytest.raises(RuntimeError, match="Previous worker.*still active"):
        runner.run_experiment("irrelevant.yaml")


def test_holm_family_keeps_suppressed_timed_restart_contrasts():
    records, results = [], {}
    for instance in ("I01_micro", "I02_small"):
        for method in ("surrogate_local", "nsga2", "random_local"):
            for seed in range(4):
                records.append({"instance": instance, "method": method, "seed": seed,
                                "hypervolume": 1.1 if method == "surrogate_local" else 1.0})
                results[(instance, method, seed)] = {"ideal": [0, 0], "span": [1, 1],
                    "budget_mode": "wall_clock", "restarts": int(instance == "I01_micro" and
                    method == "surrogate_local" and seed == 0)}
    rows = _paired_inference(pd.DataFrame(records), results, list(range(4)))
    assert len(rows) == 4
    assert all(row["holm_family_size"] == 4 for row in rows)
    suppressed = [row for row in rows if row["instance"] == "I01_micro"]
    assert all(row["holm_p"] is None for row in suppressed)
    eligible = [row for row in rows if row["instance"] == "I02_small"]
    assert all(row["sign_flip_p"] == pytest.approx(.125) for row in eligible)
    assert all(row["holm_p"] == pytest.approx(.5) for row in eligible)


def test_committed_receipt_reconciles_an_older_progress_state(small_run):
    run = runner.run_experiment("irrelevant.yaml")
    path = next((run / "tasks").glob("*/status.json"))
    complete = read_json(path)
    stale = dict(complete, status="running", generation=0, pid=os.getpid(), result_sha256=None)
    atomic_json(path, stale)
    viewed = next(item for item in runner.status(run)["tasks"] if item["key"] == complete["key"])
    assert viewed["status"] == "completed" and viewed["generation"] == 2
    assert viewed["result_sha256"] == complete["result_sha256"]
    runner.run_experiment("irrelevant.yaml")
    repaired = read_json(path)
    assert repaired["status"] == "completed" and repaired["attempts"] == 1
    assert repaired["result_sha256"] == complete["result_sha256"]


def test_completed_files_without_a_receipt_are_not_skipped(small_run):
    run = runner.run_experiment("irrelevant.yaml")
    path = next((run / "tasks").glob("*/completion.json"))
    task_folder = path.parent
    path.unlink()
    assert not runner.status(run)["all_completed"]
    runner.run_experiment("irrelevant.yaml")
    assert runner.status(run)["all_completed"]
    assert read_json(task_folder / "status.json")["attempts"] == 2
    assert read_json(path)["result_sha256"] == read_json(task_folder / "status.json")["result_sha256"]


def test_completion_receipt_must_match_the_immutable_protocol(small_run):
    run = runner.run_experiment("irrelevant.yaml")
    path = next((run / "tasks").glob("*/completion.json"))
    receipt = read_json(path)
    receipt["protocol_sha256"] = "different protocol"
    atomic_json(path, receipt)
    assert not runner.status(run)["all_completed"]


@pytest.mark.parametrize("changed", ["configuration", "source"])
def test_incompatible_resume_preserves_all_previous_task_bytes(small_run, monkeypatch, changed):
    _, config, _ = small_run
    run = runner.run_experiment("irrelevant.yaml")
    previous = {path: path.read_bytes() for path in (run / "tasks").glob("*/*.json")}
    if changed == "configuration":
        config["experiment"]["generations"] += 1
    else:
        original = runner._sources()
        monkeypatch.setattr(runner, "_sources", lambda: dict(original, **{"main.py": "changed"}))
    with pytest.raises(RuntimeError, match="Cannot resume"):
        runner.run_experiment("irrelevant.yaml")
    assert previous == {path: path.read_bytes() for path in previous}

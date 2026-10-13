"""Recover the scientific state, rather than merely skipping finished files."""
import pickle

import numpy as np
import pytest

from agriopt.model import ProblemData
from agriopt.maintenance.model import Policy, audit_calendar
from agriopt.maintenance.search import common, run_search


def tiny_data():
    weeks = np.arange(20)
    return ProblemData(
        crop_names=["short", "long"], areas=np.array([1_000., 1_500.]),
        farm_ids=np.array([0, 0]), municipalities=["A", "B"],
        yields=np.array([[1., 1.2], [1.1, .8]]),
        prices=np.vstack([1_000. + 100. * np.sin(weeks), 1_400. + 80. * np.cos(weeks)]),
        covariance=np.array([[.1, .02], [.02, .2]]), maturity=np.array([3, 5]),
        botanical_family=np.array(["A", "B"]), sales_group=np.array([0, 1]),
        sow_allowed=np.ones((2, 20), dtype=bool), setup_weeks=1, same_family_extra_rest=1,
    )


def settings():
    return {"budget_mode": "generations", "population": 8, "generations": 8,
            "budget_seconds": 1., "search": {"neighbors": 4, "min_training_samples": 2,
            "retrain_every": 2, "n_estimators": 4, "checkpoint_every_generations": 1}}


class PlannedInterruption(Exception):
    pass


@pytest.mark.parametrize("method", ["nsga2", "surrogate_local"])
def test_fixed_generation_checkpoint_recovers_population_rng_and_learning(tmp_path, method):
    data, policy = tiny_data(), Policy()
    uninterrupted = run_search(data, policy, method, 701, settings(), tmp_path / "whole.pkl")

    def interrupt_after_committed_generation(result):
        if result["generation"] == 4:
            raise PlannedInterruption()

    split_path = tmp_path / "split.pkl"
    with pytest.raises(PlannedInterruption):
        run_search(data, policy, method, 701, settings(), split_path, interrupt_after_committed_generation)
    with split_path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    assert checkpoint["generation"] == 4
    assert checkpoint["status"] == "interrupted"
    assert len(checkpoint["population"]) == 8
    if method == "surrogate_local":
        assert checkpoint["model"] is not None
        assert checkpoint["lastfit"] >= 2
        assert checkpoint["X"] and checkpoint["Y"]

    resumed = run_search(data, policy, method, 701, settings(), split_path)
    assert resumed["front"] == uninterrupted["front"]
    assert resumed["decisions"] == uninterrupted["decisions"]
    assert resumed["evaluations"] == uninterrupted["evaluations"]
    assert resumed["generation"] == 8
    assert resumed["status"] == "completed"
    assert len(resumed["restarts"]) == 1
    assert not resumed["uninterrupted"]
    assert [row["front"] for row in resumed["history"]] == [row["front"] for row in uninterrupted["history"]]
    for left, right in zip(resumed["logs"], uninterrupted["logs"]):
        for key in ("prior_labels", "fit_labels", "prediction", "target", "generation"):
            assert left[key] == right[key]
    with (tmp_path / "whole.pkl").open("rb") as handle:
        expected_state = pickle.load(handle)
    with split_path.open("rb") as handle:
        actual_state = pickle.load(handle)
    assert actual_state["rng_state"] == expected_state["rng_state"]
    assert actual_state["gate_state"] == expected_state["gate_state"]
    for actual, expected in zip(actual_state["population"], expected_state["population"]):
        np.testing.assert_array_equal(actual.crops, expected.crops)
        np.testing.assert_array_equal(actual.waits, expected.waits)
        np.testing.assert_array_equal(actual.objectives, expected.objectives)
    # Re-running a completed task returns its result without duplicate history.
    again = run_search(data, policy, method, 701, settings(), split_path)
    assert again["history"] == resumed["history"]


def test_checkpoints_reject_changed_scientific_inputs_and_corruption(tmp_path):
    path = tmp_path / "state.pkl"
    data = tiny_data()
    config = settings()
    config["generations"] = 0
    run_search(data, Policy(), "nsga2", 3, config, path)
    with pytest.raises(ValueError, match="does not match"):
        run_search(data, Policy(cost_cop_ha_pass=500_000), "nsga2", 3, config, path)
    data.prices[0, 0] += 1
    with pytest.raises(ValueError, match="does not match"):
        run_search(data, Policy(), "nsga2", 3, config, path)
    path.write_bytes(b"truncated checkpoint")
    with pytest.raises(pickle.UnpicklingError):
        run_search(data, Policy(), "nsga2", 3, config, path)
    assert path.read_bytes() == b"truncated checkpoint"


def test_wall_clock_resume_keeps_already_spent_budget(tmp_path):
    path = tmp_path / "wall.pkl"
    config = settings()
    config.update(budget_mode="wall_clock", budget_seconds=.02)

    def interrupt_initial(result):
        raise PlannedInterruption()

    with pytest.raises(PlannedInterruption):
        run_search(tiny_data(), Policy(), "nsga2", 2, config, path, interrupt_initial)
    with path.open("rb") as handle:
        state = pickle.load(handle)
    # Model a committed task that has exhausted its scientific time allowance.
    state["elapsed_s"] = config["budget_seconds"]
    with path.open("wb") as handle:
        pickle.dump(state, handle)
    resumed = run_search(tiny_data(), Policy(), "nsga2", 2, config, path)
    assert resumed["generation"] == 0
    assert resumed["elapsed_s"] >= config["budget_seconds"]
    assert resumed["stop_reason"] == "search_time_budget"
    assert resumed["restarts"][0]["used_search_seconds"] == config["budget_seconds"]


def test_strict_gap_search_never_admits_infeasible_zero_or_random_calendars(tmp_path):
    data, policy = tiny_data(), Policy(interval_weeks=4, max_idle_weeks=3)
    population, _, _ = common(data, policy, 19, 8)
    assert all(audit_calendar(data, x.events, policy)["max_idle_weeks"] <= 3 for x in population)
    config = settings()
    config["generations"] = 3
    result = run_search(data, policy, "random_search", 19, config, tmp_path / "strict.pkl")
    assert result["infeasible_candidates"] > 0
    for decision in result["decisions"]:
        assert decision["max_idle_weeks"] <= 3
        assert decision["max_untreated_weeks"] < 4
        assert decision["cultivation"]


def test_exact_resume_retains_completed_epsilon_and_rebuilds_solver(tmp_path):
    pytest.importorskip("pyscipopt")
    source = tiny_data()
    data = ProblemData(
        crop_names=source.crop_names, areas=source.areas[:1], farm_ids=source.farm_ids[:1],
        municipalities=source.municipalities[:1], yields=source.yields[:1], prices=source.prices[:, :10],
        covariance=source.covariance, maturity=source.maturity, botanical_family=source.botanical_family,
        sales_group=source.sales_group, sow_allowed=source.sow_allowed[:, :10],
        setup_weeks=source.setup_weeks, same_family_extra_rest=source.same_family_extra_rest,
    )
    config = settings()
    config["exact"] = {"epsilon_fractions": [.25, .5], "per_phase_time_limit_s": 5., "memory_limit_mb": 128}
    uninterrupted = run_search(data, Policy(interval_weeks=4), "exact", 21, config, tmp_path / "exact_whole.pkl")
    assert uninterrupted["completed_epsilon_subproblems"] == 2

    def interrupt_after_first_epsilon(result):
        if result["completed_epsilon_subproblems"] == 1:
            raise PlannedInterruption()

    path = tmp_path / "exact.pkl"
    with pytest.raises(PlannedInterruption):
        run_search(data, Policy(interval_weeks=4), "exact", 21, config, path, interrupt_after_first_epsilon)
    with path.open("rb") as handle:
        state = pickle.load(handle)
    assert state["phase"] == 1
    first_log = state["logs"][0]
    result = run_search(data, Policy(interval_weeks=4), "exact", 21, config, path)
    assert result["completed_epsilon_subproblems"] == 2
    assert result["logs"][0] == first_log
    assert [row["fraction"] for row in result["logs"]] == [.25, .5]
    assert "tree is rebuilt" in result["resume_scope"]
    for decision in result["decisions"]:
        np.testing.assert_allclose(
            decision["objectives"],
            [-decision["net_income_mcop"], decision["risk"]], rtol=1e-10,
        )


def test_late_generation_keeps_prior_front_without_clipped_timestamp(tmp_path, monkeypatch):
    import agriopt.maintenance.search as search

    clock = [0.]
    monkeypatch.setattr(search.time, "perf_counter", lambda: clock[0])
    original_selection = search.environmental_selection

    def selection_finishes_after_deadline(*args):
        selected = original_selection(*args)
        clock[0] += 2.
        return selected

    monkeypatch.setattr(search, "environmental_selection", selection_finishes_after_deadline)
    config = settings()
    config.update(budget_mode="wall_clock", budget_seconds=1.)
    initial_front = []

    def observe(result):
        if not initial_front:
            initial_front.extend(result["front"])

    result = search.run_search(tiny_data(), Policy(), "nsga2", 12, config, tmp_path / "late.pkl", observe)
    assert result["elapsed_s"] == 2.
    assert result["front"] == initial_front
    assert len(result["history"]) == 1
    assert result["history"][0]["elapsed_s"] == 0.
    assert result["logs"][-1]["discarded_generation"]
    assert result["logs"][-1]["elapsed_s"] == 2.

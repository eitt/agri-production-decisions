"""Independent small-calendar checks for the whole-plot maintenance model."""
from itertools import combinations

import numpy as np
import pytest

from agriopt.model import ProblemData
from agriopt.nsga2 import Chromosome
from agriopt.maintenance.model import (
    Policy, audit_calendar, build_exact_model, economic_anchor, encode_events,
    evaluate, maintenance_schedule,
)


def tiny_data(T=9, extra_rest=1):
    allowed = np.zeros((2, T), dtype=bool)
    allowed[:, [week for week in (0, 3, 6) if week < T]] = True
    return ProblemData(
        crop_names=["A", "B"], areas=np.array([100.0]), farm_ids=np.array([0]),
        municipalities=["Test"], yields=np.array([[1.0, 1.4]]),
        prices=np.array([np.linspace(1000, 2000, T), np.linspace(2000, 1000, T)]),
        covariance=np.array([[0.04, 0.01], [0.01, 0.06]]), maturity=np.array([2, 2]),
        botanical_family=np.array([0, 1]), sales_group=np.array([0, 1]),
        sow_allowed=allowed, setup_weeks=1, same_family_extra_rest=extra_rest,
    )


def independent_calendars(data, policy):
    """Enumerate subsets with pairwise calendar checks and a weekly care counter.

    This oracle does not use the maintenance decoder, audit, DP or SCIP event
    enumeration.  Its risk and cost calculations use a separate weekly loop.
    """
    events = []
    for crop in range(data.K):
        for sow in range(data.T):
            harvest = sow + data.maturity[crop]
            if data.sow_allowed[crop, sow] and harvest < data.T:
                events.append({"lot": 0, "crop": crop, "sow": sow,
                               "harvest": int(harvest), "area_m2": data.areas[0]})
    feasible = []
    for size in range(len(events) + 1):
        for selection in combinations(events, size):
            selection = sorted(selection, key=lambda row: row["sow"])
            bad = False
            for a, b in combinations(selection, 2):
                release = a["harvest"] + data.setup_weeks
                if data.botanical_family[a["crop"]] == data.botanical_family[b["crop"]]:
                    release += data.same_family_extra_rest
                if b["sow"] < release:
                    bad = True
                    break
            if bad:
                continue
            cropped = [any(e["sow"] <= week < e["harvest"] for e in selection)
                       for week in range(data.T)]
            idle, untreated, longest, passes = 0, 0, 0, 0
            for busy in cropped:
                idle = 0 if busy else idle + 1
                longest = max(longest, idle)
                untreated = 0 if busy else untreated + 1
                if untreated == policy.interval_weeks:
                    passes += 1
                    untreated = 0
            if policy.max_idle_weeks is not None and longest > policy.max_idle_weeks:
                continue
            exposure = np.zeros((data.K, data.T))
            for e in selection:
                exposure[e["crop"], e["harvest"]] += (
                    data.event_yield(0, e["crop"], e["sow"]) * data.areas[0]
                    * data.prices[e["crop"], e["harvest"]] / data.monetary_scale
                )
            gross = float(exposure.sum())
            cost = passes * data.areas[0] * policy.cost_cop_ha_pass / (10_000 * data.monetary_scale)
            risk = float(sum(exposure[:, week] @ data.covariance @ exposure[:, week]
                             for week in range(data.T)))
            feasible.append((selection, gross - cost, risk))
    return feasible


@pytest.mark.parametrize("interval", [1, 2, 4])
@pytest.mark.parametrize("max_gap", [None, 1, 3])
def test_economic_dp_equals_independent_exhaustive_calendars(interval, max_gap):
    data = tiny_data()
    policy = Policy(interval, 250_000, max_gap)
    expected = max(net for _, net, _ in independent_calendars(data, policy))
    anchor = economic_anchor(data, policy)
    assert anchor.income_mcop == pytest.approx(expected, abs=1e-10)
    audited = audit_calendar(data, anchor.events, policy)
    assert anchor.risk == pytest.approx(audited["risk"])
    assert anchor.objectives.tolist() == pytest.approx([-expected, anchor.risk])


def test_dp_retains_earlier_family_history_for_long_extra_rest():
    data = tiny_data(T=12, extra_rest=5)
    data.sow_allowed[:] = False
    data.sow_allowed[0, [0, 6, 9]] = True
    data.sow_allowed[1, [3, 6]] = True
    policy = Policy(3, 500_000)
    expected = max(net for _, net, _ in independent_calendars(data, policy))
    assert economic_anchor(data, policy).income_mcop == pytest.approx(expected, abs=1e-10)
    invalid = [
        {"lot": 0, "crop": 0, "sow": 0, "harvest": 2},
        {"lot": 0, "crop": 1, "sow": 3, "harvest": 5},
        {"lot": 0, "crop": 0, "sow": 6, "harvest": 8},
    ]
    with pytest.raises(ValueError, match="botanical-family rest"):
        audit_calendar(data, invalid, policy)


def test_decoder_uses_all_family_releases_and_roundtrip_preserves_dates():
    data = tiny_data(T=12, extra_rest=5)
    data.sow_allowed[:] = True
    genes = Chromosome(np.array([[0, 1, 0]]), np.zeros((1, 3), int), np.full((1, 3), 0.1))
    policy = Policy()
    evaluated = evaluate(data, genes, policy)
    assert [e["sow"] for e in evaluated.events] == [0, 3, 8]
    assert (evaluated.area_frac == 1).all()
    assert all(e["area_m2"] == data.areas[0] for e in evaluated.events)
    encoded = encode_events(data, evaluated.events, policy)
    assert encoded.events == evaluated.events
    assert encoded.objectives == pytest.approx(evaluated.objectives)


@pytest.mark.parametrize("T,passes,last_week", [(7, 0, None), (8, 1, 7), (16, 2, 15)])
def test_empty_calendar_care_boundary_and_negative_net(T, passes, last_week):
    data = tiny_data(T)
    policy = Policy(8, 250_000)
    maintenance = maintenance_schedule(data, [], policy)
    assert len(maintenance) == passes
    if maintenance:
        assert maintenance[-1]["week"] == last_week
    audited = audit_calendar(data, [], policy)
    assert audited["max_idle_weeks"] == T
    assert audited["max_untreated_weeks"] == min(T, 7)
    assert audited["gross_income_mcop"] == 0
    assert audited["net_income_mcop"] == pytest.approx(-passes * 0.0025)


def test_cultivation_resets_care_and_no_care_overlaps_crop():
    data = tiny_data(T=12)
    events = [{"lot": 0, "crop": 0, "sow": 3, "harvest": 5}]
    audited = audit_calendar(data, events, Policy(4))
    assert [row["week"] for row in audited["maintenance"]] == [8]
    assert audited["max_untreated_weeks"] == 3
    assert audited["max_idle_weeks"] == 7
    assert audited["occupancy"] == pytest.approx(2 / 12)


def test_hard_gap_includes_boundaries_and_maintenance_does_not_reset_it():
    data = tiny_data(T=16)
    with pytest.raises(ValueError, match="initial and terminal"):
        audit_calendar(data, [], Policy(4, 0, 8))
    data.sow_allowed[:] = False
    data.sow_allowed[0, 6] = True
    event = [{"lot": 0, "crop": 0, "sow": 6, "harvest": 8}]
    with pytest.raises(ValueError, match="max_idle_weeks=7"):
        encode_events(data, event, Policy(4, 0, 7))
    with pytest.raises(ValueError, match="No feasible crop calendar for lot 0"):
        economic_anchor(data, Policy(4, 0, 7))
    assert audit_calendar(data, event, Policy(4, 0, 8))["max_idle_weeks"] == 8


def test_partial_plots_and_coupled_demand_are_not_silently_repaired():
    data = tiny_data()
    event = [{"lot": 0, "crop": 0, "sow": 0, "harvest": 2, "area_m2": 50}]
    with pytest.raises(ValueError, match="whole area"):
        encode_events(data, event, Policy())
    data.demand = np.ones((2, data.T))
    with pytest.raises(ValueError, match="coupled demand"):
        economic_anchor(data, Policy())


@pytest.mark.parametrize("arguments", [
    {"interval_weeks": 0}, {"interval_weeks": 2.5},
    {"cost_cop_ha_pass": -1}, {"cost_cop_ha_pass": float("nan")},
    {"max_idle_weeks": -1}, {"max_idle_weeks": 2.5},
])
def test_policy_rejects_invalid_units_and_bounds(arguments):
    with pytest.raises(ValueError):
        Policy(**arguments)


@pytest.mark.parametrize("max_gap", [None, 1, 3])
def test_scip_net_endpoint_equals_exhaustive_and_dp(max_gap):
    pytest.importorskip("pyscipopt")
    data = tiny_data()
    policy = Policy(2, 250_000, max_gap)
    exhaustive = independent_calendars(data, policy)
    expected = max(net for _, net, _ in exhaustive)
    built = build_exact_model(data, policy, np.array([-expected, 0.]), np.ones(2), seed=7, mip_gap=0)
    model = built["model"]
    try:
        model.setObjective(built["net"], "maximize")
        model.setRealParam("limits/time", 10.0)
        model.optimize()
        assert str(model.getStatus()) == "optimal"
        assert model.getObjVal() == pytest.approx(expected, abs=1e-7)
        assert economic_anchor(data, policy).income_mcop == pytest.approx(model.getObjVal(), abs=1e-7)
        solution = model.getBestSol()
        events = []
        for event in built["events"]:
            area = model.getSolVal(solution, built["z"][event.idx])
            selected = model.getSolVal(solution, built["y"][event.idx]) > 0.5
            assert area == pytest.approx(data.areas[event.lot] if selected else 0)
            if selected:
                events.append({"lot": event.lot, "crop": event.crop, "sow": event.sow,
                               "harvest": event.harvest, "area_m2": area})
        assert audit_calendar(data, events, policy)["net_income_mcop"] == pytest.approx(expected, abs=1e-7)
    finally:
        model.freeProb()


def test_scip_epsilon_risk_matches_independent_whole_plot_front():
    pytest.importorskip("pyscipopt")
    data = tiny_data()
    policy = Policy(2, 250_000)
    exhaustive = independent_calendars(data, policy)
    maximum = max(net for _, net, _ in exhaustive)
    # A floor away from a discrete income boundary avoids ambiguous tolerance.
    floor = 0.55 * maximum
    expected_risk = min(risk for _, net, risk in exhaustive if net >= floor)
    built = build_exact_model(data, policy, np.array([-maximum, 0]), np.array([maximum, .1]), seed=9, mip_gap=0)
    model = built["model"]
    try:
        model.chgLhs(built["floor"], floor)
        model.setRealParam("limits/time", 10.0)
        model.optimize()
        assert str(model.getStatus()) == "optimal"
        assert model.getObjVal() * .1 == pytest.approx(expected_risk, abs=1e-6)
    finally:
        model.freeProb()

"""Controlled whole-plot searches with resumable, atomic checkpoints.

Checkpoints are trusted local pickle files, not an exchange format.  Heuristics
commit complete generations; a hard kill may lose the current generation.  The
search clock includes algorithm work and excludes checkpoint/report I/O.  Its
cumulative value is restored, rather than granting a new time budget on restart.
SCIP resumes completed epsilon subproblems and feasible calendars, not its tree.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile
import time
from typing import Callable

import numpy as np

from ..metrics import hypervolume_2d, nondominated, normalize
from ..model import ProblemData
from ..nsga2 import (
    Chromosome, crossover, environmental_selection, infer_max_slots,
    mutate, rank_and_crowding, tournament,
)
from .model import (
    Policy, audit_calendar, build_exact_model, economic_anchor, encode_events,
    evaluate, maintenance_schedule,
)

METHODS = ("exact", "nsga2", "random_search", "random_local", "proxy_local", "surrogate_local")
CHECKPOINT_VERSION = 1


class Archive:
    def __init__(self, individuals: list[Chromosome]):
        self.items: list[Chromosome] = []
        self.add(individuals)

    def add(self, individuals: list[Chromosome]) -> None:
        pool = self.items + individuals
        if not pool:
            return
        points = nondominated(np.asarray([x.objectives for x in pool]))
        by_point = {tuple(x.objectives): x for x in pool}
        self.items = [by_point[tuple(p)] for p in points]

    def points(self) -> np.ndarray:
        return np.asarray([x.objectives for x in self.items], dtype=float).reshape(-1, 2)


def _fresh(data: ProblemData, policy: Policy, rng: np.random.Generator, slots: int) -> Chromosome:
    crops = rng.integers(-1, data.K, size=(data.L, slots))
    crops[rng.random(crops.shape) < rng.uniform(.1, .9)] = -1
    waits = rng.integers(0, data.T, size=crops.shape)
    mask = rng.random(crops.shape) < .8
    waits[mask] = np.minimum(rng.geometric(.25, size=int(mask.sum())) - 1, data.T - 1)
    return evaluate(data, Chromosome(crops, waits, np.ones(crops.shape)), policy)


def _feasible_candidate(data, policy, make_candidate, fallback, counters=None):
    """Reject infeasible hard-gap candidates; never admit a penalised calendar."""
    for _ in range(8):
        try:
            return make_candidate()
        except ValueError:
            if counters is not None:
                counters["infeasible_candidates"] += 1
    if counters is not None:
        counters["feasible_fallbacks"] += 1
    return fallback.clone()


def common(data: ProblemData, policy: Policy, seed: int, population_size: int = 40):
    """Identical feasible initial population and scaling for all six methods."""
    if population_size < 2:
        raise ValueError("population_size must be at least two")
    anchor = economic_anchor(data, policy)
    slots = infer_max_slots(data)
    rng = np.random.default_rng(seed)
    zero = Chromosome(np.full((data.L, slots), -1, int), np.zeros((data.L, slots), int), np.ones((data.L, slots)))
    try:
        boundary = evaluate(data, zero, policy)
    except ValueError:
        # A strict maximum uncultivated gap can make zero cultivation infeasible.
        boundary = anchor.clone()
    pop = [boundary, anchor]
    while len(pop) < population_size:
        if len(pop) % 2:
            def candidate():
                x = anchor.clone()
                x.crops[rng.random(x.crops.shape) < rng.uniform(.05, .95)] = -1
                return evaluate(data, x, policy)
        else:
            def candidate():
                return _fresh(data, policy, rng, slots)
        pop.append(_feasible_candidate(data, policy, candidate, anchor))
    ideal = np.array([-anchor.income_mcop, 0.])
    # The no-cultivation maintenance cost is a valid conservative scaling bound
    # even when that boundary is excluded by the optional hard-gap constraint.
    zero_cost = sum(float(row["cost_cop"]) for row in maintenance_schedule(data, [], policy)) / data.monetary_scale
    span = np.maximum([anchor.income_mcop + zero_cost, anchor.risk], 1e-9)
    return pop, ideal, span


def _features(data, x, ideal, span):
    ev = x.events or []
    busy = sum(e["area_m2"] * (e["harvest"] - e["sow"]) for e in ev) / (data.areas.sum() * data.T)
    return [
        (-x.income_mcop - ideal[0]) / span[0], x.risk / span[1], busy,
        len(ev) / (data.L * max(1, data.T / 10)), float((x.crops >= 0).mean()),
        float(x.waits.mean() / data.T), float(x.waits.std() / data.T),
        len({e["crop"] for e in ev}) / data.K,
    ]


def _identity(data, policy, method, seed, settings):
    digest = hashlib.sha256()
    for name in ("areas", "farm_ids", "yields", "prices", "covariance", "maturity", "botanical_family", "sales_group", "sow_allowed", "eligible", "yield_by_sow_week", "demand"):
        value = getattr(data, name)
        if value is None:
            digest.update(b"None")
        else:
            array = np.asarray(value)
            digest.update(str(array.shape).encode())
            digest.update(str(array.dtype).encode())
            digest.update(json.dumps(array.tolist(), sort_keys=True).encode() if array.dtype.hasobject else array.tobytes())
    digest.update(json.dumps([data.crop_names, data.municipalities, data.setup_weeks, data.same_family_extra_rest, data.monetary_scale]).encode())
    return {"data_sha256": digest.hexdigest(), "policy": asdict(policy), "method": method, "seed": int(seed), "settings": settings}


def _atomic_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        if hasattr(os, "O_DIRECTORY"):
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _decision(data, policy, individual):
    audit = audit_calendar(data, individual.events or [], policy)
    expected = np.array([-audit["net_income_mcop"], audit["risk"]])
    if not np.allclose(individual.objectives, expected, rtol=1e-10, atol=1e-7):
        raise ValueError("checkpoint objectives disagree with its physical crop calendar")
    if not np.allclose([individual.income_mcop, individual.risk], [audit["net_income_mcop"], audit["risk"]], rtol=1e-10, atol=1e-7):
        raise ValueError("checkpoint income or risk disagrees with its physical crop calendar")
    decoded = evaluate(data, individual.clone(), policy)
    fields = ("lot", "crop", "sow", "harvest", "area_m2")
    recorded_calendar = [[row[key] for key in fields] for row in audit["cultivation"]]
    decoded_calendar = [[row[key] for key in fields] for row in decoded.events or []]
    if recorded_calendar != decoded_calendar:
        raise ValueError("checkpoint genes do not reconstruct its physical crop calendar")
    return {
        "crops": individual.crops.tolist(), "waits": individual.waits.tolist(),
        "area_frac": individual.area_frac.tolist(), "objectives": individual.objectives.tolist(),
        "events": individual.events or [], "cultivation": audit["cultivation"], "maintenance": audit["maintenance"],
        "gross_income_cop": audit["gross_income_mcop"] * data.monetary_scale,
        "maintenance_cost_cop": audit["maintenance_cost_mcop"] * data.monetary_scale,
        "net_income_cop": audit["net_income_mcop"] * data.monetary_scale,
        "net_income_mcop": audit["net_income_mcop"], "risk": audit["risk"],
        "occupied_area_week_share": audit["occupancy"], "max_idle_weeks": audit["max_idle_weeks"],
        "max_untreated_weeks": audit["max_untreated_weeks"],
    }


def _validate_state(data, policy, state, population_size):
    try:
        if len(state["population"]) != population_size or not state["archive"]:
            raise ValueError("checkpoint population or archive is incomplete")
        ideal, span = np.asarray(state["ideal"]), np.asarray(state["span"])
        if ideal.shape != (2,) or span.shape != (2,) or not np.isfinite(ideal).all() or not np.isfinite(span).all() or np.any(span <= 0):
            raise ValueError("checkpoint scaling is invalid")
        if not np.isfinite(state["elapsed_s"]) or state["elapsed_s"] < 0 or state["generation"] < 0:
            raise ValueError("checkpoint budget counters are invalid")
        if len(state["X"]) != len(state["Y"]) or state["lastfit"] > len(state["Y"]):
            raise ValueError("checkpoint surrogate labels are incomplete")
        for individual in state["population"] + state["archive"]:
            if individual.objectives is None or not np.isfinite(individual.objectives).all():
                raise ValueError("checkpoint contains unevaluated or nonfinite individuals")
            _decision(data, policy, individual)
        points = np.asarray([x.objectives for x in state["archive"]])
        expected_front = nondominated(points)
        if points.shape != expected_front.shape or not np.allclose(points, expected_front, rtol=1e-10, atol=1e-7):
            raise ValueError("checkpoint archive is not a duplicate-free nondominated front")
        test_rng = np.random.default_rng()
        test_rng.bit_generator.state = state["rng_state"]
        test_rng.bit_generator.state = state["gate_state"]
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("checkpoint scientific state is incomplete or malformed") from exc


def _result(data, policy, state, decision_cache):
    decisions = []
    for x in state["archive"]:
        key = (x.crops.tobytes(), x.waits.tobytes(), tuple(x.objectives))
        if key not in decision_cache:
            decision_cache[key] = _decision(data, policy, x)
        decisions.append(decision_cache[key])
    return {
        "method": state["identity"]["method"], "seed": state["identity"]["seed"],
        "front": [x.objectives.tolist() for x in state["archive"]], "decisions": decisions,
        "history": state["history"], "logs": state["logs"], "ideal": state["ideal"].tolist(), "span": state["span"].tolist(),
        "status": state["status"], "stop_reason": state.get("stop_reason"), "generation": state["generation"],
        "completed_epsilon_subproblems": state.get("phase", 0), "evaluations": state["evaluations"],
        "elapsed_s": state["elapsed_s"], "checkpoint_io_s": state["checkpoint_io_s"], "progress_io_s": state["progress_io_s"],
        "budget_mode": state["identity"]["settings"]["budget_mode"], "restarts": state["restarts"],
        "uninterrupted": not bool(state["restarts"]), "infeasible_candidates": state["infeasible_candidates"],
        "feasible_fallbacks": state["feasible_fallbacks"],
        "surrogate_training_samples": len(state["Y"]), "surrogate_fitted_samples": state["lastfit"],
        "clock_definition": "cumulative algorithm wall-clock time; checkpoint and progress-report I/O excluded",
        "resume_scope": "completed epsilon subproblems and feasible warm starts; SCIP tree is rebuilt" if state["identity"]["method"] == "exact" else "full algorithm state at committed generations",
    }


def run_search(data: ProblemData, policy: Policy, method: str, seed: int, settings: dict,
               checkpoint_path: str | Path, progress: Callable[[dict], None] | None = None) -> dict:
    """Run or resume one immutable task. Never silently reset a bad checkpoint."""
    if method not in METHODS:
        raise ValueError(f"unknown maintenance method: {method}")
    settings = json.loads(json.dumps(settings))
    mode = settings.get("budget_mode", "generations")
    mode = {"wallclock": "wall_clock"}.get(mode, mode)
    if mode not in ("generations", "wall_clock"):
        raise ValueError("budget_mode must be generations or wall_clock")
    settings["budget_mode"] = mode
    population_size = int(settings.get("population", 40))
    generations = int(settings.get("generations", 150))
    budget = float(settings.get("budget_seconds", settings.get("wallclock_limit_s", 60.)))
    if population_size < 2 or generations < 0 or not np.isfinite(budget) or budget <= 0:
        raise ValueError("invalid population, generations or budget_seconds")
    identity = _identity(data, policy, method, seed, settings)
    path = Path(checkpoint_path)
    decision_cache = {}
    if path.exists():
        with path.open("rb") as handle:
            state = pickle.load(handle)
        if state.get("version") != CHECKPOINT_VERSION or state.get("identity") != identity:
            raise ValueError("checkpoint does not match data, policy, method, seed or settings; use a new run-id")
        _validate_state(data, policy, state, population_size)
        if state["status"] == "completed":
            return _result(data, policy, state, decision_cache)
        state["restarts"].append({"from_generation": state["generation"], "from_phase": state.get("phase", 0), "previous_status": state["status"], "used_search_seconds": state["elapsed_s"]})
        state["status"] = "running"
    else:
        pop, ideal, span = common(data, policy, seed, population_size)
        archive = Archive(pop)
        state = {
            "version": CHECKPOINT_VERSION, "identity": identity, "population": pop, "archive": archive.items,
            "ideal": ideal, "span": span, "rng_state": np.random.default_rng(seed + 100000).bit_generator.state,
            "gate_state": np.random.default_rng(seed + 200000).bit_generator.state,
            "generation": 0, "phase": 0, "history": [{"elapsed_s": 0., "front": archive.points().tolist(), "generation": 0}],
            "logs": [], "evaluations": 0, "X": [], "Y": [], "model": None, "lastfit": 0,
            "refinement_elapsed_s": 0., "elapsed_s": 0., "checkpoint_io_s": 0., "progress_io_s": 0.,
            "status": "running", "restarts": [], "infeasible_candidates": 0, "feasible_fallbacks": 0,
        }

    def commit():
        tick = time.perf_counter()
        _atomic_checkpoint(path, state)
        state["checkpoint_io_s"] += time.perf_counter() - tick
        if progress is not None:
            tick = time.perf_counter()
            progress(_result(data, policy, state, decision_cache))
            state["progress_io_s"] += time.perf_counter() - tick

    # Persist the initial feasible front and the fact that this task was restarted.
    try:
        commit()
        if method == "exact":
            _run_exact(data, policy, seed, settings, state, commit)
        else:
            _run_heuristic(data, policy, method, seed, settings, state, commit)
    except BaseException:
        # The checkpoint already holds the last complete generation/epsilon.
        # A failing callback may have reported that boundary successfully.
        with path.open("rb") as handle:
            committed = pickle.load(handle)
        if state.get("_active_tick") is not None:
            committed["elapsed_s"] = max(committed["elapsed_s"], state["elapsed_s"] + time.perf_counter() - state["_active_tick"])
        committed["status"] = "interrupted"
        _atomic_checkpoint(path, committed)
        raise
    commit()
    return _result(data, policy, state, decision_cache)


def _run_heuristic(data, policy, method, seed, settings, state, commit):
    config = settings.get("search", {})
    mode = settings["budget_mode"]
    budget = float(settings.get("budget_seconds", settings.get("wallclock_limit_s", 60.)))
    generations = int(settings.get("generations", 150))
    population_size = int(settings.get("population", 40))
    calls = int(config.get("refinements_per_generation", 2))
    neighbors_n = int(config.get("neighbors", 8))
    fraction = float(config.get("refinement_budget_fraction", .30))
    min_samples = int(config.get("min_training_samples", 8))
    retrain = int(config.get("retrain_every", 8))
    window = int(config.get("training_window", 128))
    exploration = float(config.get("exploration_probability", .25))
    checkpoint_every = int(config.get("checkpoint_every_generations", 1))
    if calls < 0 or neighbors_n < 1 or not 0 <= fraction <= 1 or min_samples < 1 or retrain < 1 or window < 1 or not 0 <= exploration <= 1 or checkpoint_every < 1:
        raise ValueError("invalid maintenance search settings")
    rng = np.random.default_rng()
    rng.bit_generator.state = state["rng_state"]
    gate = np.random.default_rng()
    gate.bit_generator.state = state["gate_state"]
    archive = Archive(state["archive"])
    pop = state["population"]
    ideal, span = state["ideal"], state["span"]
    tick = time.perf_counter()
    state["_active_tick"] = tick

    def elapsed():
        return state["elapsed_s"] + time.perf_counter() - tick

    def available():
        return mode != "wall_clock" or elapsed() < budget

    def hv():
        return hypervolume_2d(normalize(archive.points(), ideal, span))

    while available() and (mode != "generations" or state["generation"] < generations):
        safety = settings.get("safety_limit_s")
        if safety is not None and elapsed() >= float(safety):
            state["status"] = "interrupted"
            state["stop_reason"] = "safety_limit"
            break
        previous_archive = list(archive.items)
        previous_population = pop
        previous_learning = (len(state["X"]), state["model"], state["lastfit"], state["refinement_elapsed_s"])
        previous_rng, previous_gate = rng.bit_generator.state, gate.bit_generator.state
        previous_logs = len(state["logs"])
        rank_and_crowding(pop)
        children = []
        complete = True
        for _ in range((population_size + 1) // 2):
            if not available():
                complete = False
                break
            if method == "random_search":
                pair = [_feasible_candidate(data, policy, lambda: _fresh(data, policy, rng, pop[0].crops.shape[1]), pop[0], state) for _ in range(2)]
            else:
                pair = list(crossover(tournament(pop, rng), tournament(pop, rng), rng))
                for i, child in enumerate(pair):
                    mutate(child, data, rng, data.T - 1, p_area=0)
                    try:
                        pair[i] = evaluate(data, child, policy)
                    except ValueError:
                        state["infeasible_candidates"] += 1
                        state["feasible_fallbacks"] += 1
                        pair[i] = pop[int(rng.integers(len(pop)))].clone()
            state["evaluations"] += len(pair)
            if available():
                children.extend(pair)
            else:
                complete = False
                break
        if children:
            archive.add(children)
        if method in ("random_local", "proxy_local", "surrogate_local") and children:
            for _ in range(calls):
                if not available() or (mode == "wall_clock" and state["refinement_elapsed_s"] >= fraction * budget):
                    break
                local_tick = time.perf_counter()
                candidates = pop + children
                if method == "surrogate_local" and len(state["Y"]) >= min_samples and (state["model"] is None or len(state["Y"]) - state["lastfit"] >= retrain):
                    # A missing sklearn dependency is an error, never random fallback.
                    from sklearn.ensemble import ExtraTreesRegressor
                    state["model"] = ExtraTreesRegressor(n_estimators=int(config.get("n_estimators", 32)), max_depth=int(config.get("max_depth", 6)), min_samples_leaf=int(config.get("min_samples_leaf", 2)), random_state=seed, n_jobs=1).fit(np.asarray(state["X"][-window:]), np.asarray(state["Y"][-window:]))
                    state["lastfit"] = len(state["Y"])
                prediction = None
                if method == "proxy_local":
                    index = int(np.argmax([_features(data, x, ideal, span)[2] for x in candidates]))
                elif method == "surrogate_local" and state["model"] is not None and gate.random() >= exploration:
                    predictions = state["model"].predict(np.asarray([_features(data, x, ideal, span) for x in candidates]))
                    index = int(np.argmax(predictions))
                    prediction = float(predictions[index])
                else:
                    index = int(gate.integers(len(candidates)))
                candidate = candidates[index]
                feature = _features(data, candidate, ideal, span)
                before, prior = hv(), len(state["Y"])
                neighbors = []
                for j in range(neighbors_n):
                    if not available():
                        complete = False
                        break
                    neighbor = candidate.clone()
                    lot = int(gate.integers(data.L))
                    slot = int(gate.integers(neighbor.crops.shape[1]))
                    operation = j % 4
                    if operation == 0:
                        neighbor.waits[lot, slot] = max(0, neighbor.waits[lot, slot] - int(gate.integers(1, 9)))
                    elif operation == 1:
                        neighbor.waits[lot, slot] = min(data.T - 1, neighbor.waits[lot, slot] + int(gate.integers(1, 9)))
                    elif operation == 2:
                        neighbor.crops[lot, slot] = int(gate.integers(-1, data.K))
                        neighbor.waits[lot, slot] = int(min(gate.geometric(.25) - 1, data.T - 1))
                    else:
                        neighbor.crops[lot, slot] = -1
                    try:
                        evaluate(data, neighbor, policy)
                    except ValueError:
                        state["infeasible_candidates"] += 1
                        continue
                    state["evaluations"] += 1
                    if available():
                        neighbors.append(neighbor)
                    else:
                        complete = False
                if neighbors:
                    archive.add(neighbors)
                    children.extend(neighbors)
                gain = max(0., hv() - before)
                duration = time.perf_counter() - local_tick
                state["refinement_elapsed_s"] += duration
                state["X"].append(feature)
                state["Y"].append(gain)
                state["logs"].append({"prior_labels": prior, "fit_labels": state["lastfit"], "prediction": prediction, "target": gain, "duration_s": duration, "elapsed_s": elapsed(), "generation": state["generation"] + 1})
        if children:
            pop = environmental_selection(pop + children, population_size)
        completed_elapsed = state["elapsed_s"] + time.perf_counter() - tick
        in_budget = mode != "wall_clock" or completed_elapsed <= budget
        state["elapsed_s"] = completed_elapsed
        if in_budget:
            state["generation"] += 1
            state["population"], state["archive"] = pop, archive.items
            state["rng_state"], state["gate_state"] = rng.bit_generator.state, gate.bit_generator.state
            state["history"].append({"elapsed_s": completed_elapsed, "front": archive.points().tolist(), "generation": state["generation"], "complete_generation": complete, "in_budget": True})
        else:
            # A front is published only after a fully completed in-budget
            # generation.  Deadline overruns do not receive a clipped timestamp.
            pop, archive = previous_population, Archive(previous_archive)
            state["population"], state["archive"] = pop, archive.items
            labels, state["model"], state["lastfit"], state["refinement_elapsed_s"] = previous_learning
            del state["X"][labels:]
            del state["Y"][labels:]
            rng.bit_generator.state, gate.bit_generator.state = previous_rng, previous_gate
            state["rng_state"], state["gate_state"] = previous_rng, previous_gate
            for record in state["logs"][previous_logs:]:
                record["discarded_generation"] = True
            state["logs"].append({"discarded_generation": True, "generation": state["generation"] + 1, "elapsed_s": completed_elapsed, "reason": "generation_finished_after_search_deadline"})
        state["_active_tick"] = None
        if state["generation"] % checkpoint_every == 0 or not in_budget:
            commit()
        tick = time.perf_counter()
        state["_active_tick"] = tick
    # Charge a tiny final loop check only; never count checkpoint/report latency.
    state["elapsed_s"] += time.perf_counter() - tick
    state["_active_tick"] = None
    if state["status"] != "interrupted":
        state["status"] = "completed"
        state["stop_reason"] = "generations" if mode == "generations" else "search_time_budget"


def _run_exact(data, policy, seed, settings, state, commit):
    config = settings.get("exact", {})
    fractions = [float(v) for v in config.get("epsilon_fractions", [.25, .5, .75, .9, .1])]
    if not fractions or any(not 0 <= value <= 1 for value in fractions):
        raise ValueError("epsilon_fractions must lie in [0, 1]")
    mode = settings["budget_mode"]
    budget = float(settings.get("budget_seconds", settings.get("wallclock_limit_s", 60.)))
    phase_limit = float(config.get("per_phase_time_limit_s", 60.))
    if not np.isfinite(phase_limit) or phase_limit <= 0:
        raise ValueError("per_phase_time_limit_s must be positive")
    if int(config.get("threads", 1)) != 1:
        raise ValueError("the controlled maintenance exact reference uses threads=1")
    if state["phase"] >= len(fractions) or (mode == "wall_clock" and state["elapsed_s"] >= budget):
        state["status"] = "completed"
        state["stop_reason"] = "epsilon_subproblems" if state["phase"] >= len(fractions) else "search_time_budget"
        return
    archive = Archive(state["archive"])
    tick = time.perf_counter()
    state["_active_tick"] = tick
    built = build_exact_model(data, policy, state["ideal"], state["span"], seed,
                              memory_mb=int(config.get("memory_limit_mb", 512)),
                              mip_gap=float(config.get("mip_gap", .0001)))
    model, events = built["model"], built["events"]
    y, z, exposure, care = built["y"], built["z"], built["exposure"], built["care"]
    net, rv, floor = built["net"], built["rv"], built["floor"]

    def warm(x):
        solution = model.createSol()
        active = {(e["lot"], e["crop"], e["sow"]) for e in x.events or []}
        interventions = {(row["lot"], row["week"]) for row in maintenance_schedule(data, x.events or [], policy)}
        for event in events:
            value = int((event.lot, event.crop, event.sow) in active)
            model.setSolVal(solution, y[event.idx], value)
            model.setSolVal(solution, z[event.idx], float(data.areas[event.lot]) * value)
        revenues = np.zeros((data.K, data.T))
        for event in x.events or []:
            revenues[event["crop"], event["harvest"]] += event["production_kg"] * event["price_cop_kg"] / data.monetary_scale
        for key, variable in exposure.items():
            model.setSolVal(solution, variable, float(revenues[key]))
        for key, variable in care.items():
            model.setSolVal(solution, variable, int(key in interventions))
        model.setSolVal(solution, net, x.income_mcop)
        model.setSolVal(solution, rv, x.risk / float(state["span"][1]))
        model.addSol(solution)

    try:
        initial_phase = state["phase"]
        for index in range(initial_phase, len(fractions)):
            elapsed = state["elapsed_s"] + time.perf_counter() - tick
            if mode == "wall_clock" and budget - elapsed <= .05:
                break
            if index > initial_phase:
                model.freeTransform()
            fraction = fractions[index]
            threshold = -state["ideal"][0] - (1 - fraction) * state["span"][0]
            model.chgLhs(floor, float(threshold))
            for x in state["population"] + archive.items:
                if x.income_mcop >= threshold - 1e-8:
                    warm(x)
            limit = phase_limit if mode == "generations" else max(.001, (budget - state["elapsed_s"] - (time.perf_counter() - tick)) / (len(fractions) - index) * .90)
            model.setRealParam("limits/time", float(limit))
            model.optimize()
            search_elapsed = state["elapsed_s"] + time.perf_counter() - tick
            gap = float(model.getGap())
            record = {"fraction": fraction, "status": str(model.getStatus()), "gap": gap if np.isfinite(gap) else None,
                      "elapsed_s": search_elapsed, "solutions": int(model.getNSols()), "net_floor": float(threshold), "solver_time_limit_s": limit}
            previous_archive = list(archive.items)
            if model.getNSols() > 0:
                solution = model.getBestSol()
                physical = [{"lot": event.lot, "crop": event.crop, "sow": event.sow, "harvest": event.harvest, "area_m2": float(data.areas[event.lot])}
                            for event in events if model.getSolVal(solution, y[event.idx]) > .5]
                x = encode_events(data, physical, policy)
                record["physical_objectives"] = x.objectives.tolist()
                record["floor_valid"] = bool(x.income_mcop >= threshold - 1e-6)
                record["accepted_in_budget"] = bool(mode != "wall_clock" or state["elapsed_s"] + time.perf_counter() - tick <= budget)
                if record["floor_valid"] and record["accepted_in_budget"]:
                    archive.add([x])
                    record["accepted_in_budget"] = bool(mode != "wall_clock" or state["elapsed_s"] + time.perf_counter() - tick <= budget)
                    if not record["accepted_in_budget"]:
                        archive = Archive(previous_archive)
            state["logs"].append(record)
            state["phase"] = index + 1
            state["elapsed_s"] += time.perf_counter() - tick
            state["archive"] = archive.items
            record["elapsed_s"] = state["elapsed_s"]
            if mode != "wall_clock" or state["elapsed_s"] <= budget:
                state["history"].append({"elapsed_s": state["elapsed_s"], "front": archive.points().tolist(), "epsilon_index": index, "fraction": fraction, "in_budget": True})
            else:
                archive = Archive(previous_archive)
                state["archive"] = archive.items
                record["accepted_in_budget"] = False
                record["discarded_after_deadline"] = True
            state["_active_tick"] = None
            commit()
            tick = time.perf_counter()
            state["_active_tick"] = tick
        state["elapsed_s"] += time.perf_counter() - tick
        state["_active_tick"] = None
        state["status"] = "completed"
        state["stop_reason"] = "epsilon_subproblems" if state["phase"] == len(fractions) else "search_time_budget"
    finally:
        model.freeProb()

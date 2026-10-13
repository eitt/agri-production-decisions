"""Whole-plot crop planning with explicit maintenance of idle land.

Cultivation occupies weeks ``[sow, harvest)``.  Preparation and botanical
family release dates constrain the next planting, but do not count as
cultivation.  An idle gap of ``g`` weeks requires ``floor(g / H)`` maintenance
passes.  Costs are hypothetical COP per hectare per pass; net receipts are
not profit because production costs are not included.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isfinite
from numbers import Integral
from typing import Any

import numpy as np

from ..model import ProblemData, objectives_from_production
from ..nsga2 import Chromosome, infer_max_slots


@dataclass(frozen=True)
class Policy:
    """Maintenance frequency, cost and optional cultivation gap limit.

    ``max_idle_weeks`` limits time without cultivation, including the initial
    and terminal gaps.  Maintenance does not reset that separate limit.
    ``interval_weeks`` limits untreated idle land to at most ``H - 1`` weeks.
    """

    interval_weeks: int = 8
    cost_cop_ha_pass: float = 250_000.0
    max_idle_weeks: int | None = None

    def __post_init__(self) -> None:
        if (not isinstance(self.interval_weeks, Integral)
                or isinstance(self.interval_weeks, bool) or self.interval_weeks < 1):
            raise ValueError("interval_weeks must be a positive integer")
        if not isfinite(float(self.cost_cop_ha_pass)) or self.cost_cop_ha_pass < 0:
            raise ValueError("cost_cop_ha_pass must be finite and nonnegative")
        if self.max_idle_weeks is not None and (
            not isinstance(self.max_idle_weeks, Integral)
            or isinstance(self.max_idle_weeks, bool) or self.max_idle_weeks < 0
        ):
            raise ValueError("max_idle_weeks must be a nonnegative integer or None")


def _check_scope(data: ProblemData) -> None:
    if data.demand is not None:
        raise ValueError(
            "The maintenance experiment assumes no coupled demand constraints; "
            "per-lot economic decomposition must not be applied to coupled demand."
        )
    if not np.isfinite(data.prices).all() or np.any(data.prices < 0):
        raise ValueError("The maintenance model requires finite nonnegative prices")


def _integer(value: Any, name: str) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not isfinite(number) or number != int(number):
        raise ValueError(f"{name} must be an integer")
    return int(number)


def _calendar(data: ProblemData, events: list[dict]) -> list[list[dict]]:
    """Validate supplied cultivation events without silently repairing them."""
    _check_scope(data)
    by_lot: list[list[dict]] = [[] for _ in range(data.L)]
    for source in events:
        try:
            lot = _integer(source["lot"], "lot")
            crop = _integer(source["crop"], "crop")
            sow = _integer(source["sow"], "sow")
            harvest = _integer(source["harvest"], "harvest")
        except KeyError as exc:
            raise ValueError(f"Cultivation event is missing {exc.args[0]}") from exc
        if not 0 <= lot < data.L or not 0 <= crop < data.K:
            raise ValueError("Cultivation event has a lot or crop outside the instance")
        area = float(source.get("area_m2", data.areas[lot]))
        if not np.isfinite(area) or not np.isclose(area, data.areas[lot], rtol=1e-10, atol=1e-8):
            raise ValueError(f"Cultivation must use the whole area of lot {lot}")
        eligible = data.eligible is None or bool(data.eligible[lot, crop])
        if (not eligible or not data.can_sow(crop, sow)
                or harvest != sow + int(data.maturity[crop]) or not harvest < data.T):
            raise ValueError("Cultivation violates eligibility, sowing calendar, or maturity")
        yld = data.event_yield(lot, crop, sow)
        if not np.isfinite(yld) or yld <= 0:
            raise ValueError("Cultivation must have a positive finite yield")
        by_lot[lot].append({
            "lot": lot,
            "farm": int(data.farm_ids[lot]),
            "municipality": data.municipalities[lot],
            "crop": crop,
            "crop_name": data.crop_names[crop],
            "sow": sow,
            "harvest": harvest,
            "area_m2": float(data.areas[lot]),
            "yield_kg_m2": float(yld),
            "production_kg": float(yld * data.areas[lot]),
            "price_cop_kg": float(data.prices[crop, harvest]),
        })

    for lot, sequence in enumerate(by_lot):
        sequence.sort(key=lambda row: (row["sow"], row["harvest"], row["crop"]))
        previous_harvest = -int(data.setup_weeks)
        family_release: dict[Any, int] = {}
        for slot, event in enumerate(sequence):
            family = data.botanical_family[event["crop"]]
            earliest = max(0, previous_harvest + int(data.setup_weeks),
                           family_release.get(family, 0))
            if event["sow"] < earliest:
                raise ValueError(f"Lot {lot} violates occupancy, preparation, or botanical-family rest")
            event["slot"] = slot
            previous_harvest = event["harvest"]
            family_release[family] = previous_harvest + int(data.setup_weeks) + int(data.same_family_extra_rest)
    return by_lot


def _maintenance(data: ProblemData, by_lot: list[list[dict]], policy: Policy) -> list[dict]:
    rows: list[dict] = []
    for lot, sequence in enumerate(by_lot):
        start = 0
        for event in sequence + [{"sow": data.T, "harvest": data.T}]:
            for week in range(start + policy.interval_weeks - 1, event["sow"], policy.interval_weeks):
                rows.append({
                    "lot": lot,
                    "week": int(week),
                    "area_m2": float(data.areas[lot]),
                    "activity": "maintenance",
                    "cost_cop": float(data.areas[lot] * policy.cost_cop_ha_pass / 10_000.0),
                })
            start = event["harvest"]
    return rows


def maintenance_schedule(data: ProblemData, events: list[dict], policy: Policy) -> list[dict]:
    """Return the minimum canonical care schedule for a valid crop calendar.

    The land is assumed clean at the horizon start and at harvest.  A pass
    occurs in each Hth idle week, so gaps of H-1, H and 2H weeks need zero,
    one and two passes respectively.  Passes may occur during preparation
    or family rest, but never during cultivation.
    """
    return _maintenance(data, _calendar(data, events), policy)


def audit_calendar(data: ProblemData, events: list[dict], policy: Policy) -> dict:
    """Independently reconstruct objectives, care, occupancy and both gap limits.

    All monetary quantities use ``data.monetary_scale`` (normally million COP).
    A hard gap violation raises ValueError; no cultivation is inserted by this
    audit or by maintenance.  The risk is the sum of contemporaneous monetary
    exposure quadratic forms, not annual profit variance.
    """
    by_lot = _calendar(data, events)
    cultivation = [event for sequence in by_lot for event in sequence]
    care = _maintenance(data, by_lot, policy)
    cropped = np.zeros((data.L, data.T), dtype=bool)
    cared = np.zeros_like(cropped)
    production = np.zeros((data.K, data.T), dtype=float)
    longest_idle = 0
    longest_internal = 0
    for lot, sequence in enumerate(by_lot):
        start = 0
        for event in sequence:
            gap = event["sow"] - start
            longest_idle = max(longest_idle, gap)
            if start:
                longest_internal = max(longest_internal, gap)
            cropped[lot, event["sow"]:event["harvest"]] = True
            production[event["crop"], event["harvest"]] += event["production_kg"]
            start = event["harvest"]
        longest_idle = max(longest_idle, data.T - start)
    if policy.max_idle_weeks is not None and longest_idle > policy.max_idle_weeks:
        raise ValueError(
            f"Crop calendar has {longest_idle} consecutive uncultivated weeks; "
            f"max_idle_weeks={policy.max_idle_weeks} includes initial and terminal gaps"
        )
    for row in care:
        lot, week = row["lot"], row["week"]
        if cropped[lot, week] or cared[lot, week]:
            raise ValueError("Maintenance overlaps cultivation or another pass")
        cared[lot, week] = True
    longest_untreated = 0
    for lot in range(data.L):
        streak = 0
        for week in range(data.T):
            streak = 0 if cropped[lot, week] or cared[lot, week] else streak + 1
            longest_untreated = max(longest_untreated, streak)
    if longest_untreated >= policy.interval_weeks:
        raise ValueError("Maintenance leaves too many untreated idle weeks")
    gross, risk = objectives_from_production(data, production)
    cost = sum(row["cost_cop"] for row in care) / float(data.monetary_scale)
    occupied_area_weeks = float((cropped * data.areas[:, None]).sum())
    total_area_weeks = float(data.areas.sum() * data.T)
    return {
        "gross_income_mcop": gross,
        "maintenance_cost_mcop": cost,
        "maintenance_mcop": cost,
        "net_income_mcop": gross - cost,
        "income_mcop": gross - cost,
        "income_minus_maintenance_mcop": gross - cost,
        "risk": risk,
        "maintenance_calls": len(care),
        "maintenance_ha_passes": sum(row["area_m2"] for row in care) / 10_000.0,
        "max_idle_weeks": int(longest_idle),
        "max_internal_gap": int(longest_internal),
        "max_untreated_weeks": int(longest_untreated),
        "occupancy": occupied_area_weeks / total_area_weeks,
        "idle_m2_weeks": total_area_weeks - occupied_area_weeks,
        "n_events": len(cultivation),
        "events": len(cultivation),
        "cultivation": cultivation,
        "maintenance": care,
    }


def evaluate(data: ProblemData, chromosome: Chromosome, policy: Policy) -> Chromosome:
    """Decode whole-plot genes and set minimization objectives [-net, risk].

    Family release times retain the complete earlier family history.  This
    matters when a different family is planted between two members of the
    same family.  Invalid hard-gap calendars are rejected, not repaired.
    """
    _check_scope(data)
    if (chromosome.crops.ndim != 2 or chromosome.crops.shape[0] != data.L
            or chromosome.crops.shape[1] < 1
            or chromosome.waits.shape != chromosome.crops.shape
            or chromosome.area_frac.shape != chromosome.crops.shape):
        raise ValueError("Chromosome crops, waits and area_frac must share an L by slots shape")
    if (not np.isfinite(chromosome.crops).all() or not np.isfinite(chromosome.waits).all()
            or np.any(chromosome.crops != np.floor(chromosome.crops))
            or np.any(chromosome.waits != np.floor(chromosome.waits))
            or np.any(chromosome.waits < 0)
            or np.any(chromosome.crops < -1) or np.any(chromosome.crops >= data.K)):
        raise ValueError("Chromosome crop indices and nonnegative waits must be integers")
    chromosome.area_frac[:] = 1.0
    events: list[dict] = []
    for lot in range(data.L):
        previous_harvest = -int(data.setup_weeks)
        family_release: dict[Any, int] = {}
        for slot in range(chromosome.crops.shape[1]):
            crop = int(chromosome.crops[lot, slot])
            if crop < 0 or (data.eligible is not None and not data.eligible[lot, crop]):
                continue
            family = data.botanical_family[crop]
            earliest = max(0, previous_harvest + int(data.setup_weeks),
                           family_release.get(family, 0))
            sow = data.next_allowed_sow(crop, earliest + int(chromosome.waits[lot, slot]))
            if sow is None:
                continue
            harvest = sow + int(data.maturity[crop])
            if harvest >= data.T or data.event_yield(lot, crop, sow) <= 0:
                continue
            events.append({"lot": lot, "crop": crop, "sow": sow, "harvest": harvest,
                           "area_m2": float(data.areas[lot])})
            previous_harvest = harvest
            family_release[family] = harvest + int(data.setup_weeks) + int(data.same_family_extra_rest)
    audited = audit_calendar(data, events, policy)
    chromosome.events = audited["cultivation"]
    chromosome.income_mcop = audited["net_income_mcop"]
    chromosome.risk = audited["risk"]
    chromosome.objectives = np.array([-chromosome.income_mcop, chromosome.risk], dtype=float)
    return chromosome


def encode_events(data: ProblemData, events: list[dict], policy: Policy) -> Chromosome:
    """Encode a dated, feasible whole-plot calendar without changing it."""
    audited = audit_calendar(data, events, policy)
    by_lot: list[list[dict]] = [[] for _ in range(data.L)]
    for event in audited["cultivation"]:
        by_lot[event["lot"]].append(event)
    slots = max(infer_max_slots(data), max(map(len, by_lot), default=0))
    chromosome = Chromosome(
        crops=np.full((data.L, slots), -1, dtype=int),
        waits=np.zeros((data.L, slots), dtype=int),
        area_frac=np.ones((data.L, slots), dtype=float),
    )
    for lot, sequence in enumerate(by_lot):
        previous_harvest = -int(data.setup_weeks)
        family_release: dict[Any, int] = {}
        for slot, event in enumerate(sequence):
            crop = event["crop"]
            family = data.botanical_family[crop]
            earliest = max(0, previous_harvest + int(data.setup_weeks),
                           family_release.get(family, 0))
            chromosome.crops[lot, slot] = crop
            chromosome.waits[lot, slot] = event["sow"] - earliest
            previous_harvest = event["harvest"]
            family_release[family] = previous_harvest + int(data.setup_weeks) + int(data.same_family_extra_rest)
    evaluate(data, chromosome, policy)
    expected = [(e["lot"], e["crop"], e["sow"], e["harvest"]) for e in audited["cultivation"]]
    actual = [(e["lot"], e["crop"], e["sow"], e["harvest"]) for e in chromosome.events or []]
    if actual != expected:
        raise ValueError("Dated events are not preserved by the maintenance decoder")
    return chromosome


def _candidates(data: ProblemData, lot: int) -> list[tuple[int, int, int, float]]:
    events = []
    for crop in range(data.K):
        if data.eligible is not None and not data.eligible[lot, crop]:
            continue
        for sow in np.flatnonzero(data.sow_allowed[crop]):
            harvest = int(sow + data.maturity[crop])
            yld = data.event_yield(lot, crop, int(sow))
            if harvest < data.T and np.isfinite(yld) and yld > 0:
                events.append((crop, int(sow), harvest, float(yld * data.prices[crop, harvest])))
    return sorted(events, key=lambda event: (event[2], event[1], event[0]))


def _lot_anchor(data: ProblemData, candidates: list[tuple[int, int, int, float]],
                policy: Policy) -> tuple[tuple[int, ...], float] | None:
    """Longest-path DP; a full family-state DP handles unusually long family rests."""
    n = len(candidates)
    crop = np.array([e[0] for e in candidates], dtype=int)
    sow = np.array([e[1] for e in candidates], dtype=int)
    harvest = np.array([e[2] for e in candidates], dtype=int)
    value = np.array([e[3] for e in candidates], dtype=float)
    cost = policy.cost_cop_ha_pass / 10_000.0
    interval = policy.interval_weeks
    gap_limit = policy.max_idle_weeks
    # A sufficiently long intervening crop makes every older family's release
    # expire before the next planting.  Otherwise retain all family releases.
    fast_dp = (not n or int(np.min(data.maturity[crop])) + data.setup_weeks
               >= data.same_family_extra_rest)
    if fast_dp:
        score = np.full(n, -np.inf)
        previous = np.full(n, -1, dtype=int)
        families = data.botanical_family[crop]
        for j in range(n):
            best = -cost * (sow[j] // interval) if gap_limit is None or sow[j] <= gap_limit else -np.inf
            rest = data.setup_weeks + data.same_family_extra_rest * (families[:j] == families[j])
            ids = np.flatnonzero(harvest[:j] + rest <= sow[j])
            if gap_limit is not None:
                ids = ids[sow[j] - harvest[ids] <= gap_limit]
            if len(ids):
                options = score[ids] - cost * ((sow[j] - harvest[ids]) // interval)
                selected = int(np.argmax(options))
                if options[selected] > best:
                    best = float(options[selected])
                    previous[j] = int(ids[selected])
            score[j] = best + value[j]
        best = -cost * (data.T // interval) if gap_limit is None or data.T <= gap_limit else -np.inf
        end = -1
        ids = np.arange(n)
        if gap_limit is not None:
            ids = ids[data.T - harvest <= gap_limit]
        if len(ids):
            options = score[ids] - cost * ((data.T - harvest[ids]) // interval)
            selected = int(np.argmax(options))
            if options[selected] > best:
                best = float(options[selected])
                end = int(ids[selected])
        if not np.isfinite(best):
            return None
        path = []
        while end >= 0:
            path.append(end)
            end = int(previous[end])
        return tuple(reversed(path)), best

    family_ids = {family: index for index, family in enumerate(dict.fromkeys(data.botanical_family))}
    event_families = [family_ids[data.botanical_family[e[0]]] for e in candidates]

    @lru_cache(maxsize=None)
    def visit(previous_harvest: int, release: tuple[int, ...]) -> tuple[float, tuple[int, ...]]:
        gap_start = max(0, previous_harvest)
        earliest = max(0, previous_harvest + data.setup_weeks)
        tail = data.T - gap_start
        best = -cost * (tail // interval) if gap_limit is None or tail <= gap_limit else -np.inf
        path: tuple[int, ...] = ()
        for j, (_, start, finish, receipt) in enumerate(candidates):
            family = event_families[j]
            if start < max(earliest, release[family]):
                continue
            idle = start - gap_start
            if gap_limit is not None and idle > gap_limit:
                continue
            updated = list(release)
            updated[family] = finish + data.setup_weeks + data.same_family_extra_rest
            # Expired release dates no longer distinguish states.
            next_earliest = finish + data.setup_weeks
            updated = [date if date > next_earliest else 0 for date in updated]
            remaining, suffix = visit(finish, tuple(updated))
            candidate = receipt - cost * (idle // interval) + remaining
            if candidate > best:
                best, path = float(candidate), (j,) + suffix
        return best, path

    best, path = visit(-int(data.setup_weeks), (0,) * len(family_ids))
    return None if not np.isfinite(best) else (path, best)


def economic_anchor(data: ProblemData, policy: Policy) -> Chromosome:
    """Find the exact deterministic maximum net-receipts calendar per plot.

    The income objective decomposes only in the absence of coupled demand.
    Risk is evaluated for the assembled endpoint but is not optimized here.
    Cache reuse requires identical eligibility and yield arrays, rather than
    merely sharing a municipality name.
    """
    _check_scope(data)
    cache: dict[tuple[bytes, bytes, bytes | None], tuple[list, tuple[tuple[int, ...], float] | None]] = {}
    events: list[dict] = []
    expected_net = 0.0
    for lot in range(data.L):
        eligible = np.ones(data.K, dtype=bool) if data.eligible is None else data.eligible[lot]
        seasonal = None if data.yield_by_sow_week is None else data.yield_by_sow_week[lot].tobytes()
        key = (eligible.tobytes(), data.yields[lot].tobytes(), seasonal)
        if key not in cache:
            candidates = _candidates(data, lot)
            cache[key] = (candidates, _lot_anchor(data, candidates, policy))
        candidates, solution = cache[key]
        if solution is None:
            raise ValueError(
                f"No feasible crop calendar for lot {lot} under "
                f"max_idle_weeks={policy.max_idle_weeks}; initial and terminal gaps are included"
            )
        path, net_per_m2 = solution
        expected_net += net_per_m2 * data.areas[lot] / data.monetary_scale
        for index in path:
            crop, sow, harvest, _ = candidates[index]
            events.append({"lot": lot, "crop": crop, "sow": sow, "harvest": harvest,
                           "area_m2": float(data.areas[lot])})
    result = encode_events(data, events, policy)
    if not np.isclose(result.income_mcop, expected_net, rtol=1e-10, atol=1e-7):
        raise RuntimeError("Economic DP and physical maintenance objectives disagree")
    return result


def build_exact_model(data: ProblemData, policy: Policy, ideal: np.ndarray,
                      span: np.ndarray, seed: int, memory_mb: float = 512,
                      mip_gap: float = 1e-4) -> dict:
    """Build the same whole-plot maintenance MIQCP used by the search evaluation.

    H-week windows require cultivation or maintenance.  Optional (G+1)-week
    windows require cultivation alone.  The returned floor constrains *net*
    receipts; the returned ``rv`` minimizes risk scaled by ``span[1]``.
    The solver may use extra care passes in a tied solution.  Re-evaluating its
    cultivation calendar with canonical care can only improve net receipts.
    """
    _check_scope(data)
    ideal = np.asarray(ideal, dtype=float)
    span = np.asarray(span, dtype=float)
    if ideal.shape != (2,) or span.shape != (2,) or not np.isfinite(ideal).all() or not np.isfinite(span).all() or np.any(span <= 0):
        raise ValueError("ideal and positive span must be finite vectors of length two")
    if not np.isfinite(memory_mb) or memory_mb <= 0:
        raise ValueError("memory_mb must be positive and finite")
    from ..exact_scip import build_scip_model
    from pyscipopt import quicksum

    model, events, y, z, exposure, gross, risk, _ = build_scip_model(
        data, objective="max_income", threads=1, seed=seed, mip_gap=mip_gap,
    )
    try:
        model.setRealParam("limits/memory", float(memory_mb))
        occupied: dict[tuple[int, int], list] = {}
        for event in events:
            model.addCons(z[event.idx] == float(data.areas[event.lot]) * y[event.idx],
                          name=f"whole_plot_{event.idx}")
            for week in range(event.sow, event.harvest):
                occupied.setdefault((event.lot, week), []).append(y[event.idx])
        busy, care = {}, {}
        for lot in range(data.L):
            for week in range(data.T):
                busy[lot, week] = quicksum(occupied.get((lot, week), []))
                care[lot, week] = model.addVar(vtype="B", name=f"maint_{lot}_{week}")
                model.addCons(care[lot, week] + busy[lot, week] <= 1,
                              name=f"maintenance_outside_crops_{lot}_{week}")
            for start in range(data.T - policy.interval_weeks + 1):
                model.addCons(
                    quicksum(care[lot, week] + busy[lot, week]
                             for week in range(start, start + policy.interval_weeks)) >= 1,
                    name=f"untreated_limit_{lot}_{start}",
                )
            if policy.max_idle_weeks is not None:
                width = policy.max_idle_weeks + 1
                for start in range(data.T - width + 1):
                    model.addCons(quicksum(busy[lot, week] for week in range(start, start + width)) >= 1,
                                  name=f"cultivation_gap_limit_{lot}_{start}")
        cost = quicksum(
            float(data.areas[lot] * policy.cost_cop_ha_pass / (10_000.0 * data.monetary_scale)) * care[lot, week]
            for lot in range(data.L) for week in range(data.T)
        )
        minimum_net = -float(data.areas.sum() * data.T * policy.cost_cop_ha_pass
                             / (10_000.0 * data.monetary_scale))
        net = model.addVar(lb=minimum_net, name="net_receipts")
        model.addCons(net == gross - cost, name="net_receipts_definition")
        floor = model.addCons(net >= minimum_net, name="epsilon_net_receipts_floor")
        rv = model.addVar(lb=0.0, name="scaled_risk")
        model.addCons(rv >= risk / float(span[1]), name="scaled_risk_epigraph")
        model.setObjective(rv, "minimize")
        return {"model": model, "events": events, "y": y, "z": z,
                "exposure": exposure, "care": care, "busy": busy,
                "net": net, "rv": rv, "floor": floor, "gross": gross, "risk": risk}
    except Exception:
        model.freeProb()
        raise

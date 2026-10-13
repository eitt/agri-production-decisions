"""Resumable economic-endpoint sensitivity and hard-idle feasibility checks.

These dynamic programmes maximise receipts minus maintenance cost. Their risk
is evaluated afterwards; they do not optimise the joint income-risk frontier.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_config
from ..paths import repo_root
from ..provenance import sha256_file
from .inputs import load_inputs
from .instances import CATALOG, make_instance
from .model import Policy, audit_calendar, economic_anchor
from .persistence import run_lock


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _atomic_json(path: Path, value) -> None:
    _atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str))


def _csv(path: Path, rows: list[dict]) -> None:
    buffer = io.StringIO()
    pd.DataFrame(rows).to_csv(buffer, index=False)
    _atomic_text(path, buffer.getvalue())


def _record_key(instance: str, H: int, cost: float, G: int | None, suite: str) -> str:
    return f"{instance}__{suite}__H{H}__c{cost:g}__G{'none' if G is None else G}"


def _record_hash(value: dict) -> str:
    payload = {key: item for key, item in value.items() if key != "sha256"}
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _protocol(config: dict, input_files: list[Path], instances: list[str], grid: dict) -> dict:
    clean = {key: value for key, value in config.items() if not key.startswith("_")}
    package = Path(__file__).parent
    # Include the decoder and all maintenance sources that determine instances,
    # feasible crop calendars, monetary units, and objective evaluation.
    sources = [*package.glob("*.py"), package.parent / "model.py", package.parent / "nsga2.py"]
    contract = {
        "schema": 1,
        "scope": "exact separable economic endpoint; risk evaluated, not optimised",
        "config": clean,
        "inputs": {str(path.resolve()): sha256_file(path) for path in input_files},
        "sources": {str(path.relative_to(package.parent)): sha256_file(path) for path in sorted(sources)},
        "instances": instances,
        "grid": grid,
    }
    encoded = json.dumps(contract, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    return {"fingerprint": hashlib.sha256(encoded).hexdigest(), "contract": contract}


def _publish(directory: Path, records: dict[str, dict], total: int, fingerprint: str) -> None:
    rows = [item["result"] for item in records.values()]
    sensitivity = [row for row in rows if row["suite"] == "maintenance_cost"]
    hard_idle = [row for row in rows if row["suite"] == "hard_idle"]
    _csv(directory / "results.csv", sensitivity)
    _csv(directory / "hard_idle_feasibility.csv", hard_idle)
    _atomic_json(directory / "plans.json", [item["plan"] for item in records.values() if item.get("plan")])
    _atomic_json(directory / "progress.json", {
        "fingerprint": fingerprint, "completed": len(rows), "expected": total,
        "remaining": total - len(rows), "complete": len(rows) == total,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "feasible": sum(bool(row["feasible"]) for row in rows),
        "infeasible": sum(not bool(row["feasible"]) for row in rows),
        "interpretation": "Economic endpoints only; partial outputs are provisional.",
    })


def _validate_saved(saved: dict, *, key: str, fingerprint: str, data, policy: Policy) -> bool:
    if saved.get("fingerprint") != fingerprint or saved.get("key") != key or saved.get("status") != "completed":
        return False
    if saved.get("sha256") != _record_hash(saved):
        return False
    row = saved.get("result", {})
    if not isinstance(row.get("feasible"), bool):
        return False
    if _record_key(row["instance"], int(row["H"]), float(row["cost_cop_ha_pass"]), row["G"], row["suite"]) != key:
        return False
    if not row["feasible"]:
        # A completed infeasibility diagnostic must carry the actual DP reason.
        return saved.get("plan") is None and str(row.get("reason", "")).startswith("No feasible crop calendar")
    plan = saved.get("plan")
    if not isinstance(plan, dict) or not isinstance(plan.get("cultivation"), list):
        return False
    audit = audit_calendar(data, plan["cultivation"], policy)
    for metric in ("gross_income_mcop", "maintenance_cost_mcop", "net_income_mcop", "risk", "maintenance_calls", "occupancy"):
        if metric not in row or not np.isclose(float(audit[metric]), float(row[metric]), rtol=1e-10, atol=1e-7):
            return False
    fixed = audit_calendar(data, plan["baseline_cultivation"], policy)
    if not np.isclose(float(fixed["net_income_mcop"]), float(row["fixed_calendar_net_mcop"]), rtol=1e-10, atol=1e-7):
        return False
    gain = float(audit["net_income_mcop"] - fixed["net_income_mcop"])
    if gain < -1e-6 or not np.isclose(gain, float(row["reoptimization_gain_mcop"]), rtol=1e-10, atol=1e-7):
        return False
    return (audit["maintenance"] == plan.get("maintenance") and
            bool(row["crop_calendar_changed"]) == (audit["cultivation"] != fixed["cultivation"]))


def run_scenarios(config_path: str | Path, run_id: str | None = None) -> Path:
    """Run or resume the sensitivity grid, saving each audited case atomically.

    The same run identifier requires the same configuration, input bytes and
    source hashes. A malformed or physically invalid case is recomputed. A
    changed experiment requires a new run identifier.
    """
    config = load_config(config_path)
    root = repo_root()
    requested = config.get("experiment", {}).get("instances", CATALOG["instance"].tolist())
    instances = [str(name) for name in requested]
    catalog = CATALOG.set_index("instance")
    if len(instances) != len(set(instances)) or not instances:
        raise ValueError("Scenario instances must be nonempty and unique")
    unknown = sorted(set(instances) - set(catalog.index))
    if unknown:
        raise ValueError(f"Unknown maintenance instances: {unknown}")
    legacy_settings = config.get("scenarios", {})
    settings = {
        "interval_weeks": legacy_settings.get("maintenance_intervals_weeks", [4, 8, 12]),
        "cost_cop_ha_pass": legacy_settings.get("maintenance_costs_cop_ha_pass", [0, 100000, 250000, 500000]),
        "max_idle_weeks": legacy_settings.get("max_idle_weeks", [None, 26]),
        "hard_idle_weeks": legacy_settings.get("idle_limits_weeks", [4, 8, 12, 26]),
        **config.get("maintenance", {}).get("scenarios", {}),
    }
    grid = {
        "interval_weeks": settings.get("interval_weeks", [4, 8, 12]),
        "cost_cop_ha_pass": settings.get("cost_cop_ha_pass", [0, 100000, 250000, 500000]),
        "max_idle_weeks": settings.get("max_idle_weeks", [None, 26]),
        "hard_idle_weeks": settings.get("hard_idle_weeks", [4, 8, 12, 26]),
        "hard_idle_interval_weeks": settings.get("hard_idle_interval_weeks", 8),
        "hard_idle_cost_cop_ha_pass": settings.get("hard_idle_cost_cop_ha_pass", 250000),
    }
    cases = [(int(H), float(cost), None if G is None else int(G), "maintenance_cost")
             for H, cost, G in product(grid["interval_weeks"], grid["cost_cop_ha_pass"], grid["max_idle_weeks"])]
    cases.extend((int(grid["hard_idle_interval_weeks"]), float(grid["hard_idle_cost_cop_ha_pass"]), int(G), "hard_idle")
                 for G in grid["hard_idle_weeks"])
    for H, cost, G, _ in cases:
        Policy(interval_weeks=H, cost_cop_ha_pass=cost, max_idle_weeks=G)
    if len(cases) != len(set(cases)) or not cases:
        raise ValueError("Scenario grid must be nonempty and must not duplicate cases")
    productivity, history, forecast, demand, files = load_inputs(config, root)
    protocol = _protocol(config, files, instances, grid)
    fingerprint = protocol["fingerprint"]
    rid = run_id or "maintenance_scenarios_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if Path(rid).name != rid or rid in {".", ".."}:
        raise ValueError("run_id must be a single directory name")
    directory = root / "outputs" / "maintenance" / rid / "scenarios"
    directory.mkdir(parents=True, exist_ok=True)
    with run_lock(directory):
        return _execute_scenarios(directory, protocol, instances, cases, grid, productivity, history, forecast, demand)


def _execute_scenarios(directory, protocol, instances, cases, grid, productivity, history, forecast, demand):
    fingerprint = protocol["fingerprint"]
    catalog = CATALOG.set_index("instance")
    protocol_path = directory / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Scenario configuration, inputs or source changed; use a new run_id")
    else:
        if (directory / "cases").exists() and any((directory / "cases").iterdir()):
            raise ValueError("Saved scenario cases have no protocol; use a new run_id")
        _atomic_json(protocol_path, protocol)
    records: dict[str, dict] = {}
    expected = len(instances) * len(cases)
    if not (directory / "progress.json").exists():
        _publish(directory, records, expected, fingerprint)
    for instance in instances:
        row = catalog.loc[instance].copy()
        row["instance"] = instance
        data, metadata = make_instance(row, productivity, forecast, history, demand=demand)
        anchors: dict[tuple[int, float, int | None], dict] = {}

        def anchor(H: int, cost: float, G: int | None) -> dict:
            index = (H, cost, G)
            if index not in anchors:
                policy = Policy(interval_weeks=H, cost_cop_ha_pass=cost, max_idle_weeks=G)
                chromosome = economic_anchor(data, policy)
                anchors[index] = audit_calendar(data, chromosome.events, policy)
            return anchors[index]

        for H, cost, G, suite in cases:
            policy = Policy(interval_weeks=H, cost_cop_ha_pass=cost, max_idle_weeks=G)
            key = _record_key(instance, H, cost, G, suite)
            target = directory / "cases" / f"{key}.json"
            if target.exists():
                try:
                    saved = json.loads(target.read_text(encoding="utf-8"))
                    if _validate_saved(saved, key=key, fingerprint=fingerprint, data=data, policy=policy):
                        records[key] = saved
                        _publish(directory, records, expected, fingerprint)
                        continue
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    pass
            base = {"instance": instance, "suite": suite, "H": H, "G": G, "cost_cop_ha_pass": cost,
                    "scope": "economic endpoint; risk evaluated only"}
            try:
                baseline = anchor(H, 0.0, G)
                result = anchor(H, cost, G)
                # Charge the actual canonical activity on the zero-cost optimum
                # using the requested price before comparing with reoptimisation.
                fixed = audit_calendar(data, baseline["cultivation"], policy)
                gain = float(result["net_income_mcop"] - fixed["net_income_mcop"])
                if gain < -1e-6:
                    raise RuntimeError("Maintenance reoptimisation is worse than its feasible baseline")
                metrics = {name: value for name, value in result.items() if name not in {"cultivation", "maintenance"}}
                output = {**base, "feasible": True, **metrics,
                          "fixed_calendar_net_mcop": fixed["net_income_mcop"],
                          "reoptimization_gain_mcop": gain,
                          "crop_calendar_changed": result["cultivation"] != baseline["cultivation"]}
                plan = {**base, "cultivation": result["cultivation"], "maintenance": result["maintenance"],
                        "baseline_cultivation": baseline["cultivation"], "instance_metadata": metadata}
            except ValueError as exc:
                if not str(exc).startswith("No feasible crop calendar"):
                    raise
                output = {**base, "feasible": False, "reason": str(exc)}
                plan = None
            saved = {"status": "completed", "key": key, "fingerprint": fingerprint, "result": output, "plan": plan}
            saved["sha256"] = _record_hash(saved)
            _atomic_json(target, saved)
            records[key] = saved
            _publish(directory, records, expected, fingerprint)
            print(f"scenario {key}: {'feasible' if output['feasible'] else 'infeasible'} ({len(records)}/{expected})", flush=True)
    _atomic_json(directory / "audit.json", {
        "fingerprint": fingerprint, "complete": len(records) == expected,
        "scenarios": len(records), "expected": expected,
        "maintenance_cost_cases": len(instances) * (len(cases) - len(grid["hard_idle_weeks"])),
        "hard_idle_cases": len(instances) * len(grid["hard_idle_weeks"]),
        "feasible": sum(bool(item["result"]["feasible"]) for item in records.values()),
        "infeasible": sum(not bool(item["result"]["feasible"]) for item in records.values()),
        "scope": "Deterministic separable economic endpoints, not stochastic Pareto comparisons.",
        "checks": ["physical calendar audit", "canonical maintenance coverage", "cost recomputation",
                   "reoptimisation no worse than same zero-cost calendar charged afterward",
                   "saved-case objective and activity validation before resume"],
    })
    return directory

"""Incremental, explicitly preliminary outputs and paired final comparisons."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from ..metrics import hypervolume_2d, normalize
from .persistence import atomic_json, atomic_text, read_json, utc_now


def _csv(folder: Path, name: str, rows: list[dict], columns: list[str] | None = None) -> None:
    frame = pd.DataFrame(rows)
    if frame.empty and columns:
        frame = pd.DataFrame(columns=columns)
    atomic_text(folder / name, frame.to_csv(index=False))


def _records(item) -> list[dict]:
    """Keep physical records intact even when an implementation uses positional events."""
    return [record if isinstance(record, dict) else {"record": json.dumps(record)} for record in (item or [])]


def _paired_inference(metrics: pd.DataFrame, results: dict, expected_seeds: list[int]) -> list[dict]:
    rows = []
    if metrics.empty or "surrogate_local" not in set(metrics.method):
        return rows
    generator = np.random.default_rng(8831)
    for instance in sorted(metrics.instance.unique()):
        block = metrics[metrics.instance == instance]
        for comparator in ("nsga2", "random_local", "proxy_local"):
            left = block[block.method == "surrogate_local"].set_index("seed")
            right = block[block.method == comparator].set_index("seed")
            seeds = sorted(set(left.index) & set(right.index))
            if not seeds:
                continue
            for seed in seeds:
                a, b = results[(instance, "surrogate_local", seed)], results[(instance, comparator, seed)]
                if a["budget_mode"] != b["budget_mode"] or not np.array_equal(a["ideal"], b["ideal"]) \
                        or not np.array_equal(a["span"], b["span"]):
                    raise ValueError("Paired fronts have incompatible budgets or normalization.")
            differences = left.loc[seeds, "hypervolume"].to_numpy() - right.loc[seeds, "hypervolume"].to_numpy()
            complete = set(seeds) == set(expected_seeds)
            timed_restart = any(results[(instance, method, seed)].get("budget_mode") == "wall_clock"
                                and bool(results[(instance, method, seed)].get("restarts", 0))
                                for method in ("surrogate_local", comparator) for seed in seeds)
            row = {"instance": instance, "method": "surrogate_local", "comparator": comparator,
                   "paired_seeds": len(seeds), "complete_seed_block": complete,
                   "uninterrupted_timed_block": not timed_restart,
                   "mean_hv_difference": float(differences.mean()),
                   "positive_pairs": int(np.sum(differences > 0)), "negative_pairs": int(np.sum(differences < 0)),
                   "bootstrap_ci_low": None, "bootstrap_ci_high": None,
                   "sign_flip_p": None, "holm_p": None,
                   "primary_contrast": comparator in {"nsga2", "random_local"},
                   "interpretation": "Descriptive; no confirmatory superiority claim."}
            if complete and not timed_restart and len(seeds) >= 3:
                samples = generator.choice(differences, size=(50000, len(seeds)), replace=True).mean(axis=1)
                lower, upper = np.quantile(samples, [0.025, 0.975])
                row.update(bootstrap_ci_low=float(lower), bootstrap_ci_high=float(upper))
                observed = abs(float(differences.mean()))
                if len(seeds) <= 16:
                    bit_patterns = np.arange(2 ** len(seeds), dtype=np.uint64)[:, None]
                    signs = ((bit_patterns >> np.arange(len(seeds), dtype=np.uint64)) & 1).astype(float) * 2 - 1
                    null = np.abs(signs @ differences / len(seeds))
                    p_value = float(np.mean(null >= observed - 1e-14))
                    row["test"] = "two-sided exact paired sign-flip"
                else:
                    signs = generator.choice([-1, 1], size=(200000, len(seeds)))
                    null = np.abs(signs @ differences / len(seeds))
                    p_value = float((1 + np.sum(null >= observed - 1e-14)) / (1 + len(null)))
                    row["test"] = "two-sided paired sign-flip Monte Carlo, 200000 samples"
                row["sign_flip_p"] = p_value
                row["interpretation"] = "Paired test assumes exchangeable signs under the null; assess effect size and CI."
            elif timed_restart:
                row["interpretation"] = "Descriptive only: a timed search was restarted; lost in-flight time cannot be reconstructed."
            rows.append(row)
    primary = [row for row in rows if row["primary_contrast"]]
    ordered = sorted(primary, key=lambda row: row["sign_flip_p"] if row["sign_flip_p"] is not None else 1.0)
    previous = 0.0
    for index, row in enumerate(ordered):
        row["holm_family_size"] = len(ordered)
        if row["sign_flip_p"] is None:
            continue
        adjusted = max(previous, min(1.0, row["sign_flip_p"] * (len(ordered) - index)))
        row["holm_p"] = adjusted
        previous = adjusted
    return rows


def build_report(run: str | Path) -> dict:
    """Refresh available results; create final outputs only when every task validates."""
    from .runner import resolve_run, status
    folder = resolve_run(run)
    manifest = read_json(folder / "manifest.json")
    current = status(folder)
    rows, front_rows, history_rows, decision_rows = [], [], [], []
    cultivation_rows, care_rows, solver_rows = [], [], []
    results = {}
    for state in current["tasks"]:
        task_folder = folder / "tasks" / state["key"]
        completed = state["status"] == "completed"
        path = task_folder / ("result.json" if completed else "partial.json")
        if not path.exists():
            continue
        try:
            result = read_json(path)
        except (OSError, ValueError):
            continue
        base = {"instance": state["instance"], "method": state["method"], "seed": state["seed"],
                "task_status": state["status"], "preliminary": not completed,
                "budget_mode": result.get("budget_mode", state.get("settings", {}).get("budget_mode"))}
        front = np.asarray(result.get("front", []), dtype=float)
        if front.ndim != 2 or front.shape[1] != 2 or not len(front) or "ideal" not in result or "span" not in result:
            continue
        if not np.isfinite(front).all():
            continue
        scaled = normalize(front, np.asarray(result["ideal"]), np.asarray(result["span"]))
        hv = hypervolume_2d(scaled, reference_point=(1.1, 1.1))
        rows.append(dict(base, hypervolume=hv, points=len(front), elapsed_s=result.get("elapsed_s"),
                         evaluations=result.get("evaluations"), generation=result.get("generation"),
                         restarts=result.get("restarts"), checkpoint_io_s=result.get("checkpoint_io_s"),
                         progress_io_s=result.get("progress_io_s")))
        results[(base["instance"], base["method"], base["seed"])] = result
        for point, normalized in zip(front, scaled):
            front_rows.append(dict(base, negative_net_income_scaled=float(point[0]), risk_scaled=float(point[1]),
                                   normalized_income_objective=float(normalized[0]), normalized_risk=float(normalized[1])))
        for item in result.get("history", []):
            trace_front = np.asarray(item.get("front", []), dtype=float)
            if trace_front.ndim == 2 and trace_front.shape[1] == 2 and len(trace_front):
                value = hypervolume_2d(normalize(trace_front, np.asarray(result["ideal"]),
                                                np.asarray(result["span"])), reference_point=(1.1, 1.1))
            else:
                value = item.get("hypervolume")
            history_rows.append(dict(base, elapsed_s=item.get("elapsed_s"), generation=item.get("generation"),
                                     hypervolume=value))
        for index, decision in enumerate(result.get("decisions", [])):
            identity = dict(base, decision=index)
            physical = {name: decision.get(name) for name in ("gross_income_cop", "maintenance_cost_cop",
                        "net_income_cop", "risk", "occupied_area_week_share", "max_idle_weeks", "max_untreated_weeks")}
            objectives = decision.get("objectives", [None, None])
            decision_rows.append(dict(identity, **physical, negative_net_income_scaled=objectives[0],
                                      risk_scaled=objectives[1]))
            cultivation_rows.extend(dict(identity, **event) for event in
                                    _records(decision.get("cultivation", decision.get("events", []))))
            care_rows.extend(dict(identity, **event) for event in _records(decision.get("maintenance", [])))
        for log in result.get("logs", []):
            if isinstance(log, dict):
                flat = {key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in log.items()}
                solver_rows.append(dict(base, **flat))
    metrics = pd.DataFrame(rows)
    summaries = []
    if not metrics.empty:
        for (instance, method), block in metrics.groupby(["instance", "method"], sort=True):
            done = block[~block.preliminary]
            summaries.append({"instance": instance, "method": method, "completed_seeds": len(done),
                              "partial_seeds": int(block.preliminary.sum()),
                              "expected_seeds": len(manifest["protocol"]["config"]["experiment"]["seeds"]),
                              "mean_completed_hypervolume": float(done.hypervolume.mean()) if len(done) else None,
                              "sd_completed_hypervolume": float(done.hypervolume.std(ddof=1)) if len(done) > 1 else None,
                              "mean_available_hypervolume": float(block.hypervolume.mean()),
                              "budget_mode": block.budget_mode.iloc[0]})
    destination = folder / "preliminary"
    destination.mkdir(parents=True, exist_ok=True)
    outputs = [("metrics.csv", rows, ["instance", "method", "seed", "task_status", "hypervolume"]),
               ("summary.csv", summaries, ["instance", "method", "completed_seeds", "partial_seeds"]),
               ("fronts.csv", front_rows, ["instance", "method", "seed", "negative_net_income_scaled", "risk_scaled"]),
               ("hypervolume_history.csv", history_rows, ["instance", "method", "seed", "elapsed_s", "hypervolume"]),
               ("decisions.csv", decision_rows, ["instance", "method", "seed", "decision", "net_income_cop"]),
               ("cultivation.csv", cultivation_rows, ["instance", "method", "seed", "decision", "lot", "crop", "start", "end"]),
               ("maintenance.csv", care_rows, ["instance", "method", "seed", "decision", "lot", "week", "cost_cop"]),
               ("solver_logs.csv", solver_rows, ["instance", "method", "seed", "status", "gap"])]
    for name, values, columns in outputs:
        _csv(destination, name, values, columns)
    report = {"updated_at": utc_now(), "complete": current["all_completed"], "total_tasks": current["total"],
              "counts": current["counts"], "reported_results": len(rows), "reference_point": [1.1, 1.1],
              "normalization": "Shared initial population anchors for each instance/seed, fixed across methods.",
              "warning": "Available partial fronts are preliminary. Mean rankings are descriptive.",
              "maintenance": manifest["protocol"]["config"]["maintenance"],
              "units": "Net receipts = wholesale receipts minus maintenance; production costs are not included.",
              "risk": "Sum of weekly contemporaneous correlated monetary exposure; not annual profit variance."}
    atomic_json(destination / "report.json", report)
    if current["all_completed"]:
        final = folder / "final"
        final.mkdir(parents=True, exist_ok=True)
        for name, values, columns in outputs:
            _csv(final, name, values, columns)
        contrasts = _paired_inference(metrics, results, manifest["protocol"]["config"]["experiment"]["seeds"])
        _csv(final, "paired_comparisons.csv", contrasts,
             ["instance", "method", "comparator", "paired_seeds", "mean_hv_difference"])
        from .figures import export_figures
        export_figures(final / "figures", metrics, results)
        # Commit the publication marker only after all final exports succeed.
        atomic_json(final / "report.json", report)
    elif (folder / "final" / "report.json").exists():
        invalidated = folder / "invalidated"
        invalidated.mkdir(parents=True, exist_ok=True)
        old_final = invalidated / f"final-{time.time_ns()}"
        (folder / "final").replace(old_final)
        atomic_json(old_final / "invalidation.json", dict(report, warning="Final outputs invalidated: task validation failed. "
                                                         "Use preliminary outputs until the run is repaired."))
    return report

"""Resumable task grid for the explicit land-maintenance model."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from copy import deepcopy
from contextlib import contextmanager
import importlib.metadata
import hashlib
import multiprocessing as mp
import os
from pathlib import Path
import platform
import re
import socket
import sys
import threading
import time
import traceback

import numpy as np
import pandas as pd

from ..config import load_config
from .persistence import _process_alive, atomic_bytes, atomic_json, atomic_text, canonical_json, file_sha256, read_json, run_lock, utc_now

METHODS = ("exact", "nsga2", "random_search", "random_local", "proxy_local", "surrogate_local")
REPORT_INTERVAL_S = 30


def task_key(instance: str, method: str, seed: int) -> str:
    return f"{instance}__{method}__{int(seed)}"


def _clean_config(config: dict) -> dict:
    return {key: value for key, value in deepcopy(config).items() if not key.startswith("_")}


def _sources() -> dict:
    package = Path(__file__).resolve().parent
    source_root = package.parent.parent
    paths = list(package.parent.rglob("*.py"))
    hashes = {str(path.relative_to(source_root)): file_sha256(path) for path in sorted(paths)}
    entry = source_root.parent / "main.py"
    if entry.is_file():
        hashes["main.py"] = file_sha256(entry)
    return hashes


def _input_hashes(paths, root: Path) -> dict:
    result = {}
    for raw in paths:
        path = Path(raw).resolve()
        try:
            label = str(path.relative_to(root))
        except ValueError:
            label = str(path)
        result[label] = file_sha256(path)
    return result


def _environment() -> dict:
    versions = {"python": sys.version.split()[0], "system": platform.system(), "machine": platform.machine()}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "pyscipopt"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def resolve_run(run: str | Path) -> Path:
    """Accept either a full run folder or the short ID printed by the CLI."""
    path = Path(run)
    if path.is_absolute() or path.exists() or len(path.parts) > 1:
        return path.resolve()
    from ..paths import repo_root
    return repo_root() / "outputs" / "maintenance" / path


def _settings(config: dict, instance: str) -> dict:
    experiment = config["experiment"]
    budgets = experiment.get("budget_seconds_by_instance", {})
    return {
        "budget_mode": experiment.get("budget_mode", "generations"),
        "budget_seconds": float(budgets.get(instance, experiment.get("budget_seconds", 60))),
        "population": int(experiment.get("population", 40)),
        "generations": int(experiment.get("generations", 150)),
        "safety_limit_s": experiment.get("safety_limit_s"),
        "search": deepcopy(experiment.get("search", {})),
        "exact": deepcopy(experiment.get("exact", {})),
    }


def _task_grid(config: dict) -> list[dict]:
    from .instances import CATALOG
    experiment = config.get("experiment", {})
    instances = experiment.get("instances", CATALOG.instance.tolist())
    methods = experiment.get("methods", METHODS)
    seeds = experiment.get("seeds", [7200])
    unknown_instances = set(instances) - set(CATALOG.instance)
    unknown_methods = set(methods) - set(METHODS)
    if unknown_instances or unknown_methods:
        raise ValueError(f"Unknown instances {sorted(unknown_instances)} or methods {sorted(unknown_methods)}")
    if not instances or not methods or not seeds:
        raise ValueError("The experiment must contain instances, methods and seeds.")
    grid = [{"key": task_key(instance, method, seed), "instance": instance, "method": method,
             "seed": int(seed), "settings": _settings(config, instance)}
            for instance in instances for method in methods for seed in seeds]
    if len({task["key"] for task in grid}) != len(grid):
        raise ValueError("Repeated instances, methods or seeds create duplicate tasks.")
    if any(task["settings"]["budget_mode"] not in {"wall_clock", "generations"} for task in grid):
        raise ValueError("budget_mode must be 'wall_clock' or 'generations'.")
    return grid


def _validate_result(result: dict, task: dict) -> None:
    for name in ("instance", "method", "seed"):
        if result.get(name) != task[name]:
            raise ValueError(f"Result identity mismatch: {name}")
    if result.get("status") != "completed":
        raise ValueError("Result is not completed.")
    if not result.get("audit", {}).get("passed"):
        raise ValueError("Completed result has no successful physical audit.")
    for name in ("front", "decisions", "ideal", "span", "history", "logs"):
        if name not in result:
            raise ValueError(f"Result lacks {name}.")
    front = np.asarray(result["front"], dtype=float)
    ideal, span = np.asarray(result["ideal"], dtype=float), np.asarray(result["span"], dtype=float)
    if front.ndim != 2 or front.shape[1] != 2 or not len(front) or not np.isfinite(front).all():
        raise ValueError("Completed result must contain a finite nonempty two-objective front.")
    if ideal.shape != (2,) or span.shape != (2,) or not np.isfinite(ideal).all() \
            or not np.isfinite(span).all() or not np.all(span > 0):
        raise ValueError("Invalid normalization.")
    saved = {tuple(item["objectives"]) for item in result["decisions"] if "objectives" in item}
    if not set(map(tuple, front.tolist())).issubset(saved):
        raise ValueError("A front point has no saved physical decision.")


def _protocol_sha256(protocol: dict) -> str:
    return hashlib.sha256(canonical_json(protocol).encode("utf-8")).hexdigest()


def _completion(run: Path, task: dict, protocol_sha256: str | None = None) -> dict | None:
    """An independently committed receipt, not progress-file state, proves completion."""
    folder = run / "tasks" / task["key"]
    path = folder / "result.json"
    try:
        receipt = read_json(folder / "completion.json")
        expected_protocol = protocol_sha256 or _protocol_sha256(read_json(run / "manifest.json")["protocol"])
        if receipt.get("status") != "completed" or receipt.get("task_key") != task["key"] \
                or receipt.get("protocol_sha256") != expected_protocol:
            return None
        if file_sha256(path) != receipt.get("result_sha256"):
            return None
        result = read_json(path)
        if result.get("protocol_sha256") != expected_protocol:
            return None
        _validate_result(result, task)
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return receipt


def _completed(run: Path, task: dict, state: dict, protocol_sha256: str | None = None) -> bool:
    return _completion(run, task, protocol_sha256) is not None


def _state_view(state: dict, receipt: dict) -> dict:
    summary = receipt.get("summary", {})
    return dict(state, **summary, status="completed", result_sha256=receipt["result_sha256"],
                updated_at=receipt["completed_at"], error=None)


def status(run: str | Path) -> dict:
    folder = resolve_run(run)
    if not (folder / "manifest.json").exists():
        launches = sorted((folder / "launches").glob("*.json"))
        if launches:
            launch = read_json(launches[-1])
            return {"run": str(folder), "total": 0, "counts": {}, "all_completed": False,
                    "tasks": [], "preflight_status": launch["status"], "error": launch.get("error"),
                    "launch_evidence": str(launches[-1])}
    manifest = read_json(folder / "manifest.json")
    protocol_sha = _protocol_sha256(manifest["protocol"])
    counts = {name: 0 for name in ("pending", "running", "completed", "failed", "interrupted")}
    tasks = []
    for task in manifest["tasks"]:
        state_path = folder / "tasks" / task["key"] / "status.json"
        state = read_json(state_path) if state_path.exists() else dict(task, status="pending")
        receipt = _completion(folder, task, protocol_sha)
        if receipt is not None:
            state = _state_view(state, receipt)
        elif state.get("status") == "completed":
            state = dict(state, status="failed", error="Completed result is missing, corrupt, or incompatible.")
        counts[state.get("status", "failed")] += 1
        tasks.append(state)
    return {"run": str(folder), "total": len(tasks), "counts": counts, "all_completed":
            counts["completed"] == len(tasks), "tasks": tasks}


def _write_index(run: Path) -> dict:
    current = status(run)
    rows = [{key: item.get(key) for key in ("instance", "method", "seed", "status", "generation",
             "evaluations", "elapsed_s", "attempts", "updated_at", "error", "result_sha256")}
            for item in current["tasks"]]
    atomic_text(run / "run_index.csv", pd.DataFrame(rows).to_csv(index=False))
    atomic_json(run / "progress.json", {key: value for key, value in current.items() if key != "tasks"})
    return current


@contextmanager
def _report_heartbeat(run: Path):
    """Refresh committed partials even while the single worker is inside SCIP.

    This thread reads atomic snapshots; it never touches algorithm state or
    checkpoints. The launcher joins it before publishing the final report.
    """
    from .reporting import build_report
    stop = threading.Event()

    def refresh():
        while not stop.wait(REPORT_INTERVAL_S):
            try:
                _write_index(run)
                build_report(run)
            except Exception as error:
                atomic_json(run / "heartbeat_error.json", {"at": utc_now(), "error": str(error)})
                print(f"Report refresh failed: {error}", file=sys.stderr, flush=True)

    thread = threading.Thread(target=refresh, name="maintenance-report", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


def _worker(payload: dict) -> dict:
    from threadpoolctl import threadpool_limits
    from .inputs import load_inputs
    from .instances import CATALOG, make_instance
    from .model import Policy, audit_calendar, evaluate
    from ..nsga2 import Chromosome
    from .search import run_search
    run, root = Path(payload["run"]), Path(payload["root"])
    task, config = payload["task"], payload["config"]
    folder = run / "tasks" / task["key"]
    state = read_json(folder / "status.json")
    state.update(status="running", attempts=int(state.get("attempts", 0)) + 1,
                 pid=os.getpid(), host=socket.gethostname(), updated_at=utc_now(), error=None)
    atomic_json(folder / "status.json", state)

    def progress(partial: dict) -> None:
        partial = dict(partial, instance=task["instance"], method=task["method"], seed=task["seed"],
                       budget_mode=task["settings"]["budget_mode"], status="running")
        atomic_json(folder / "partial.json", partial)
        state.update({name: partial[name] for name in ("generation", "evaluations", "elapsed_s")
                      if name in partial})
        state["updated_at"] = utc_now()
        atomic_json(folder / "status.json", state)

    try:
        with threadpool_limits(limits=1):
            productivity, history, forecast, demand, _ = load_inputs(config, root)
            row = CATALOG.set_index("instance").loc[task["instance"]].copy()
            row["instance"] = task["instance"]
            data, metadata = make_instance(row, productivity, forecast, history, demand=demand)
            metadata = dict(metadata, horizon_weeks=data.T, lots=data.L, total_area_ha=float(data.areas.sum()) / 10000,
                            monetary_scale=float(data.monetary_scale), products=data.crop_names)
            policy = Policy(**{name: config["maintenance"][name] for name in
                                ("interval_weeks", "cost_cop_ha_pass", "max_idle_weeks")})
            result = run_search(data, policy, task["method"], task["seed"], task["settings"],
                                folder / "checkpoint.pkl", progress=progress)
            if result.get("status") != "completed":
                progress(result)
                state.update(status="interrupted", updated_at=utc_now(),
                             error=f"Search stopped at a checkpoint: {result.get('stop_reason', result.get('status'))}")
                atomic_json(folder / "status.json", state)
                return {"key": task["key"], "status": "interrupted", "error": state["error"]}
            for decision in result["decisions"] + result.get("epsilon_incumbents", []):
                physical = audit_calendar(data, decision.get("events", []), policy)
                expected = [-physical["net_income_mcop"], physical["risk"]]
                np.testing.assert_allclose(decision["objectives"], expected, rtol=1e-8, atol=1e-7)
                chromosome = Chromosome(np.asarray(decision["crops"], dtype=int),
                                        np.asarray(decision["waits"], dtype=int),
                                        np.asarray(decision["area_frac"], dtype=float))
                evaluate(data, chromosome, policy)
                np.testing.assert_allclose(chromosome.objectives, expected, rtol=1e-8, atol=1e-7)
                np.testing.assert_allclose(decision["maintenance_cost_cop"],
                                           physical["maintenance_cost_mcop"] * data.monetary_scale,
                                           rtol=1e-8, atol=1e-5)
                if canonical_json(decision.get("maintenance", [])) != canonical_json(physical["maintenance"]):
                    raise ValueError("Saved maintenance activity differs from the physical calendar audit.")
            result["audit"] = {"passed": True,
                                "decisions_checked": len(result["decisions"]) + len(result.get("epsilon_incumbents", [])),
                                "scope": "calendar feasibility, care coverage, costs, objectives and chromosome reconstruction"}
            result.update(instance=task["instance"], method=task["method"], seed=task["seed"],
                          status="completed", budget_mode=task["settings"]["budget_mode"],
                          settings=task["settings"], instance_metadata=metadata, maintenance=config["maintenance"],
                          protocol_sha256=payload["protocol_sha256"])
            _validate_result(result, task)
            atomic_json(folder / "result.json", result)
            result_sha = file_sha256(folder / "result.json")
            receipt = {"schema_version": 1, "task_key": task["key"], "status": "completed",
                       "protocol_sha256": payload["protocol_sha256"], "result_sha256": result_sha,
                       "completed_at": utc_now(), "summary": {"elapsed_s": result.get("elapsed_s"),
                           "generation": result.get("generation"), "evaluations": result.get("evaluations"),
                           "attempts": state["attempts"]}}
            atomic_json(folder / "completion.json", receipt)
            state.update(status="completed", result_sha256=result_sha,
                         updated_at=utc_now(), elapsed_s=result.get("elapsed_s"),
                         generation=result.get("generation"), evaluations=result.get("evaluations"))
            atomic_json(folder / "status.json", state)
            return {"key": task["key"], "status": "completed"}
    except BaseException as error:
        state.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                     updated_at=utc_now(), error=f"{type(error).__name__}: {error}")
        atomic_text(folder / "error.txt", traceback.format_exc())
        atomic_json(folder / "status.json", state)
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        return {"key": task["key"], "status": "failed", "error": state["error"]}


def run_experiment(config_path: str | Path, run_id: str | None = None, workers: int | None = None) -> Path:
    """Run or resume one immutable experiment; a reused ID never changes its protocol."""
    from .inputs import load_inputs
    from .reporting import build_report
    config = load_config(config_path)
    root = Path(config["_root"])
    experiment = config.get("experiment", {})
    selected_id = run_id or config.get("project", {}).get("run_id", "maintenance")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", selected_id):
        raise ValueError("run-id must contain only letters, digits, underscores, periods and hyphens.")
    count = int(workers if workers is not None else experiment.get("workers", 1))
    if count < 1:
        raise ValueError("workers must be at least 1.")
    _, _, _, _, paths = load_inputs(config, root)
    grid = _task_grid(config)
    protocol = {"schema_version": 2, "config": _clean_config(config), "inputs_sha256": _input_hashes(paths, root),
                "source_sha256": _sources(), "environment": _environment(), "actual_workers": count, "tasks": grid}
    protocol_sha = _protocol_sha256(protocol)
    run = root / "outputs" / "maintenance" / selected_id
    with run_lock(run):
        manifest_path = run / "manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if canonical_json(manifest["protocol"]) != canonical_json(protocol):
                raise RuntimeError("Cannot resume: configuration, input files, source code, or software environment changed. "
                                   "Use a new --run-id to preserve the previous experiment.")
        else:
            atomic_json(manifest_path, {"created_at": utc_now(), "protocol": protocol, "tasks": grid,
                                        "environment": _environment(), "config_path": str(config_path)})
        # A killed launcher can leave spawned workers alive. Reclaiming its
        # lock does not authorize duplicate writers for those same tasks.
        for task in grid:
            old_path = run / "tasks" / task["key"] / "status.json"
            if not old_path.exists():
                continue
            previous = read_json(old_path)
            if previous.get("status") == "running" and not _completed(run, task, previous, protocol_sha):
                host = previous.get("host", socket.gethostname())
                if host != socket.gethostname() or _process_alive(int(previous.get("pid", -1))):
                    raise RuntimeError(f"Previous worker {previous.get('pid')} on {host} is still active for "
                                       f"{task['key']}. Wait for it to finish or stop the previous local launcher.")
        pending = []
        for task in grid:
            folder = run / "tasks" / task["key"]
            state_path = folder / "status.json"
            state = read_json(state_path) if state_path.exists() else dict(task, status="pending", attempts=0)
            receipt = _completion(run, task, protocol_sha)
            if receipt is not None:
                # Repair the cosmetic progress record without rerunning search.
                atomic_json(state_path, _state_view(state, receipt))
                continue
            # Keep every former result and receipt before repair overwrites it.
            if (folder / "result.json").exists() or (folder / "completion.json").exists():
                previous = run / "invalidated" / task["key"] / str(time.time_ns())
                previous.mkdir(parents=True)
                for name in ("result.json", "completion.json", "status.json", "partial.json", "error.txt"):
                    if (folder / name).is_file():
                        atomic_bytes(previous / name, (folder / name).read_bytes())
            if state.get("status") == "completed":
                state["error"] = "Previous result failed validation; rerunning from its checkpoint."
            if state.get("status") == "running":
                state["status"] = "interrupted"
                state["error"] = "Previous worker stopped; resuming the last atomic checkpoint. In-flight work may be lost."
                interruptions = list(state.get("interruptions", []))
                interruptions.append({"detected_at": utc_now(), "previous_pid": state.get("pid"),
                                      "last_checkpoint_update": state.get("updated_at")})
                state["interruptions"] = interruptions
            else:
                state["status"] = "pending"
            state["updated_at"] = utc_now()
            atomic_json(state_path, state)
            pending.append(task)
        _write_index(run)
        build_report(run)
        order = np.random.default_rng(int(config.get("project", {}).get("schedule_seed", 715)))
        order.shuffle(pending)
        for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[variable] = "1"
        payloads = [{"run": str(run), "root": str(root), "config": config, "task": task,
                     "protocol_sha256": protocol_sha} for task in pending]
        try:
            if count == 1:
                with _report_heartbeat(run):
                    for payload in payloads:
                        outcome = _worker(payload)
                        _write_index(run)
                        # The heartbeat is the only concurrent report writer.
                        print(f"{outcome['key']}: {outcome['status']}", flush=True)
            else:
                options = {"max_workers": count, "mp_context": mp.get_context("spawn")}
                if sys.version_info >= (3, 11):
                    options["max_tasks_per_child"] = int(experiment.get("max_tasks_per_child", 1))
                with ProcessPoolExecutor(**options) as pool:
                    futures = {pool.submit(_worker, payload): payload for payload in payloads}
                    waiting = set(futures)
                    try:
                        while waiting:
                            done, waiting = wait(waiting, timeout=REPORT_INTERVAL_S, return_when=FIRST_COMPLETED)
                            for future in done:
                                outcome = future.result()
                                print(f"{outcome['key']}: {outcome['status']}", flush=True)
                            _write_index(run)
                            build_report(run)
                    except BaseException:
                        for future in waiting:
                            future.cancel()
                        # Python <3.14 has no public terminate_workers(). Every
                        # worker already persisted its last complete boundary.
                        processes = list((getattr(pool, "_processes", None) or {}).values())
                        for process in processes:
                            if process.is_alive():
                                process.terminate()
                        for process in processes:
                            process.join(timeout=5)
                        for task in pending:
                            state_path = run / "tasks" / task["key"] / "status.json"
                            state = read_json(state_path)
                            if state.get("status") == "running":
                                state.update(status="interrupted", updated_at=utc_now(),
                                             error="Launcher interrupted; resume from the last complete checkpoint.")
                                atomic_json(state_path, state)
                        raise
        except KeyboardInterrupt:
            _write_index(run)
            build_report(run)
            raise
        except Exception:
            _write_index(run)
            build_report(run)
            raise
        current = _write_index(run)
        build_report(run)
        if not current["all_completed"]:
            raise RuntimeError(f"{current['counts']['failed']} tasks failed; {current['counts']['interrupted']} interrupted. "
                               "See run_index.csv and task error.txt; "
                               "repeat the same command to resume.")
    return run

"""Build a publication-safe snapshot from completed run metadata.

This scanner deliberately ignores run_index.csv: it can run while the experiment is
still active and includes only runs whose metadata and expected front are both complete.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
INSTANCES = [
    "I01_micro", "I02_small", "I03_diverse", "I04_two_farms",
    "I05_coop_small", "I06_coop_medium", "I07_network", "I08_network_large",
]
METHODS = ("exact", "nsga2", "nsga2_qp", "hnsga2")
SEEDS = tuple(range(2026, 2036))
FRONTS = {
    "exact": "exact_front_scip.csv",
    "nsga2": "nsga2_front.csv",
    "nsga2_qp": "nsga2_qp_front.csv",
    "hnsga2": "hnsga2_front.csv",
}
EVENTS = {
    "exact": "exact_events_scip.csv",
    "nsga2": "nsga2_events.csv",
    "nsga2_qp": "nsga2_qp_events.csv",
    "hnsga2": "hnsga2_events.csv",
}


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _expected(instance: str) -> dict[str, set[int]]:
    required = {m: set(SEEDS) for m in METHODS if m != "exact"}
    if instance in INSTANCES[:4]:
        required["exact"] = {2026}
    return required


def scan(
    run: str = "publication_slsqp_test",
    baseline_run: str | None = None,
) -> tuple[dict, list[dict]]:
    method_run_sources = {method: run for method in METHODS}
    if baseline_run:
        method_run_sources["nsga2"] = baseline_run
    config_path = ROOT / "configs" / "publication_slsqp.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    fx = cfg["reporting"]["fx"]
    valid: list[dict] = []
    completed: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))

    for method, source_run in method_run_sources.items():
        run_dir = ROOT / "results" / source_run
        for metadata_path in sorted(run_dir.rglob("metadata.json")):
            try:
                meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue  # file may be in flight
            instance = meta.get("instance")
            if instance not in INSTANCES or meta.get("method") != method or meta.get("status") != "ok":
                continue
            try:
                seed = int(meta["seed"])
            except (KeyError, TypeError, ValueError):
                continue
            front_path = metadata_path.parent / FRONTS[method]
            if not front_path.is_file() or front_path.stat().st_size == 0:
                continue
            completed[instance][method].add(seed)
            events_path = metadata_path.parent / EVENTS[method]
            valid.append({
                "instance": instance,
                "method": method,
                "seed": seed,
                "source_run": source_run,
                "status": "ok",
                "runtime_s": float(meta.get("runtime_s", "nan")),
                "metadata_path": metadata_path.relative_to(ROOT).as_posix(),
                "front_path": front_path.relative_to(ROOT).as_posix(),
                "events_path": events_path.relative_to(ROOT).as_posix() if events_path.is_file() else "",
            })

    included, excluded = [], []
    details = {}
    for instance in INSTANCES:
        required = _expected(instance)
        missing = {
            method: sorted(seeds - completed[instance].get(method, set()))
            for method, seeds in required.items()
            if seeds - completed[instance].get(method, set())
        }
        counts = {m: len(completed[instance].get(m, set())) for m in METHODS}
        details[instance] = {"complete": not missing, "completed_runs": counts, "missing": missing}
        (included if not missing else excluded).append(instance)

    included_set = set(included)
    safe_rows = sorted(
        (r for r in valid if r["instance"] in included_set),
        key=lambda r: (INSTANCES.index(r["instance"]), METHODS.index(r["method"]), r["seed"]),
    )
    snapshot = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run": run,
        "method_run_sources": method_run_sources,
        "configuration_path": config_path.relative_to(ROOT).as_posix(),
        "git_commit": _git_commit(),
        "fx": fx,
        "included_complete_instances": included,
        "excluded_incomplete_instances": excluded,
        "instances": details,
        "completed_runs_by_method": {
            m: sum(details[i]["completed_runs"][m] for i in INSTANCES) for m in METHODS
        },
        "included_runs_by_method": {
            m: sum(details[i]["completed_runs"][m] for i in included) for m in METHODS
        },
    }
    return snapshot, safe_rows


def write_snapshot(
    run: str = "publication_slsqp_test",
    baseline_run: str | None = None,
    snapshot_id: str | None = None,
) -> tuple[Path, Path]:
    snapshot, rows = scan(run, baseline_run)
    out = ROOT / "artifacts" / (snapshot_id or run)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "paper_snapshot.json"
    csv_path = out / "paper_snapshot_runs.csv"
    json_path.write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")
    fields = ["instance", "method", "seed", "source_run", "status", "runtime_s", "metadata_path", "front_path", "events_path"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="publication_slsqp_test")
    parser.add_argument("--baseline-run")
    parser.add_argument("--snapshot-id")
    args = parser.parse_args()
    json_path, csv_path = write_snapshot(args.run, args.baseline_run, args.snapshot_id)
    snapshot = json.loads(json_path.read_text(encoding="utf-8"))
    print(f"Included: {', '.join(snapshot['included_complete_instances']) or 'none'}")
    print(f"Excluded: {', '.join(snapshot['excluded_incomplete_instances']) or 'none'}")
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()

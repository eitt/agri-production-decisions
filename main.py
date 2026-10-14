"""Simple Python 3.12 entry point for the existing resumable maintenance runner."""
from __future__ import annotations

import argparse
import multiprocessing
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/maintenance_local.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)
    if sys.version_info[:2] != (3, 12):
        parser.error("Use Python 3.12: .\\.venv\\Scripts\\python.exe main.py ...")
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be at least 1")
    # Anchor relative config and data paths even when launched from another folder.
    os.chdir(ROOT)
    from agriopt.config import load_config
    from agriopt.maintenance.persistence import atomic_json, utc_now
    from agriopt.maintenance.runner import run_experiment

    try:
        config = load_config(args.config)
    except (OSError, ValueError) as error:
        print(f"Cannot load configuration: {error}", file=sys.stderr)
        return 1
    run_id = args.run_id or config.get("project", {}).get("run_id", "maintenance")
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        parser.error("Invalid --run-id; use letters, digits, periods, underscores or hyphens")
    launch_path = ROOT / "outputs" / "maintenance" / run_id / "launches" / f"{time.time_ns()}-{os.getpid()}.json"
    record = {"started_at": utc_now(), "pid": os.getpid(), "python": sys.executable,
              "config": args.config, "run_id": run_id, "workers": args.workers,
              "status": "starting", "global_session_time_limit": None}
    atomic_json(launch_path, record)
    try:
        output = run_experiment(args.config, run_id=run_id, workers=args.workers)
    except KeyboardInterrupt:
        record.update(status="interrupted", finished_at=utc_now(), exit_code=130,
                      error="Interrupted by the operator; repeat the same command to resume.")
        atomic_json(launch_path, record)
        print(record["error"], file=sys.stderr)
        return 130
    except Exception as error:
        record.update(status="blocked" if isinstance(error, FileNotFoundError) else "failed",
                      finished_at=utc_now(), exit_code=1, error=f"{type(error).__name__}: {error}")
        atomic_json(launch_path, record)
        print(record["error"], file=sys.stderr)
        print(f"Launch evidence: {launch_path}", file=sys.stderr)
        return 1
    record.update(status="completed", finished_at=utc_now(), exit_code=0)
    atomic_json(launch_path, record)
    print(output, flush=True)
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())

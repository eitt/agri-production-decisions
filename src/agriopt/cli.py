from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

from .config import load_config
from .data_pipeline import update_data
from .experiment import run_experiment
from .forecast_pipeline import build_forecasts
from .paths import repo_root
from .reporting import build_report


def _yes(flag: bool) -> str:
    return "OK" if flag else "MISSING"


def doctor() -> int:
    root = repo_root()
    checks = [
        ("Python >= 3.10", sys.version_info >= (3, 10), sys.version.split()[0]),
        ("PySCIPOpt", importlib.util.find_spec("pyscipopt") is not None, "exact MIQCP + QP refinement"),
        ("zeep", importlib.util.find_spec("zeep") is not None, "DANE SIPSA SOAP download"),
        ("pdflatex", shutil.which("pdflatex") is not None, "optional manuscript compilation"),
        ("git", shutil.which("git") is not None, "provenance commit capture"),
    ]
    print(f"Repository: {root}")
    width = max(len(x[0]) for x in checks)
    for name, ok, note in checks:
        print(f"{name:<{width}}  {_yes(ok):<7}  {note}")
    seed_files = [
        root / "data/seed/productividad_actualizada.csv",
        root / "data/seed/precios_historicos_2013_2017.csv",
        root / "data/seed/demo_price_forecast_104w.csv",
    ]
    for p in seed_files:
        print(f"{'seed:'+p.name:<{width}}  {_yes(p.exists()):<7}  {p}")
    return 0


def _add_config_arg(p: argparse.ArgumentParser, default: str = "configs/quick.yaml"):
    p.add_argument("--config", default=default, help="YAML experiment configuration")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agriopt",
        description="Reproducible agricultural production decision experiments under price risk.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="Check optional solvers, data connectors, and local tooling")

    data = sub.add_parser("data", help="Download and prepare DANE SIPSA data")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    data_update = data_sub.add_parser("update", help="Download/normalize price and supply data")
    _add_config_arg(data_update, "configs/publication.yaml")

    fc = sub.add_parser("forecast", help="Backtest and build price forecasts")
    fc_sub = fc.add_subparsers(dest="forecast_command", required=True)
    fc_build = fc_sub.add_parser("build", help="Select ARIMA/GP/ARIMA+GP per crop and forecast")
    _add_config_arg(fc_build, "configs/publication.yaml")

    solve = sub.add_parser("solve", help="Run one method on one benchmark instance")
    solve_sub = solve.add_subparsers(dest="solve_method", required=True)
    for method in ["exact", "nsga2", "nsga2_qp", "hnsga2"]:
        q = solve_sub.add_parser(method, help=f"Run {method}")
        _add_config_arg(q, "configs/publication.yaml")
        q.add_argument("--instance", default="I01_micro")
        q.add_argument("--seed", type=int, default=2026)
        q.add_argument("--run-id", default=None)
        q.add_argument("--strict", action="store_true", help="Stop immediately on solver/model errors")

    exp = sub.add_parser("experiment", help="Run configured benchmark experiment")
    _add_config_arg(exp, "configs/publication.yaml")
    exp.add_argument("--run-id", default=None)
    exp.add_argument("--strict", action="store_true")

    rep = sub.add_parser("report", help="Generate publication figures and tables from a run")
    rep.add_argument("--run", default="latest", help="Run id, results/<id>, absolute path, or 'latest'")
    rep.add_argument("--dpi", type=int, default=300)
    rep.add_argument("--no-pdf", action="store_true")
    rep.add_argument("--no-latex", action="store_true")

    pipe = sub.add_parser("pipeline", help="Data -> forecast -> experiment -> publication artifacts")
    _add_config_arg(pipe, "configs/quick.yaml")
    pipe.add_argument("--run-id", default=None)
    pipe.add_argument("--strict", action="store_true")

    maintenance = sub.add_parser("maintenance", help="Whole-plot model with idle-land maintenance")
    maintenance_sub = maintenance.add_subparsers(dest="maintenance_command", required=True)
    for command in ["run", "scenarios"]:
        q = maintenance_sub.add_parser(command, help="Run resumable benchmark" if command == "run" else "Run maintenance sensitivity grid")
        _add_config_arg(q, "configs/maintenance_smoke.yaml")
        q.add_argument("--run-id", default=None)
        if command == "run":
            q.add_argument("--workers", type=int, default=None)
    for command in ["status", "report"]:
        q = maintenance_sub.add_parser(command, help="Read saved progress" if command == "status" else "Refresh preliminary/final reports")
        q.add_argument("--run", required=True, help="Maintenance run id or absolute run directory")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "maintenance":
        if args.maintenance_command == "run":
            from .maintenance.runner import run_experiment as run_maintenance
            out = run_maintenance(args.config, run_id=args.run_id, workers=args.workers)
        elif args.maintenance_command == "scenarios":
            from .maintenance.scenarios import run_scenarios
            out = run_scenarios(args.config, run_id=args.run_id)
        elif args.maintenance_command == "status":
            from .maintenance.runner import status
            print(json.dumps(status(args.run), indent=2, ensure_ascii=False))
            return 0
        else:
            from .maintenance.reporting import build_report as report_maintenance
            out = report_maintenance(args.run)
        print(out)
        return 0
    if args.command == "doctor":
        return doctor()
    if args.command == "data" and args.data_command == "update":
        result = update_data(args.config)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "forecast" and args.forecast_command == "build":
        result = build_forecasts(args.config)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "solve":
        method = str(args.solve_method)
        run = run_experiment(
            args.config,
            run_id=args.run_id,
            continue_on_error=not args.strict,
            instances_override=[args.instance],
            methods_override=[method],
            seeds_override=[args.seed],
            force_exact_instances=(method == "exact"),
        )
        print(run)
        return 0
    if args.command == "experiment":
        run = run_experiment(args.config, run_id=args.run_id, continue_on_error=not args.strict)
        print(run)
        return 0
    if args.command == "report":
        out = build_report(args.run, dpi=args.dpi, export_pdf=not args.no_pdf, export_latex=not args.no_latex)
        print(out)
        return 0
    if args.command == "pipeline":
        cfg = load_config(args.config)
        if cfg.get("data", {}).get("download", {}).get("enabled", False):
            print("[1/4] Updating DANE SIPSA inputs...")
            update_data(args.config)
        else:
            print("[1/4] Data download skipped by configuration.")
        if cfg.get("forecast", {}).get("enabled", False):
            print("[2/4] Building price forecasts...")
            build_forecasts(args.config)
        else:
            print("[2/4] Forecast build skipped by configuration.")
        print("[3/4] Running optimization experiment...")
        run = run_experiment(args.config, run_id=args.run_id, continue_on_error=not args.strict)
        rcfg = cfg.get("report", {})
        print("[4/4] Building figures and tables...")
        out = build_report(
            run,
            dpi=int(rcfg.get("dpi", 300)),
            export_pdf=bool(rcfg.get("export_pdf", True)),
            export_latex=bool(rcfg.get("export_latex", True)),
        )
        print(f"Run: {run}\nArtifacts: {out}")
        return 0
    raise RuntimeError("Unhandled command")


if __name__ == "__main__":
    raise SystemExit(main())


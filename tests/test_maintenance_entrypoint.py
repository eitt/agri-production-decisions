"""The simple launcher delegates and records preflight failures without results."""
import importlib.util
from pathlib import Path

import pytest

from agriopt.maintenance import runner
from agriopt.maintenance.inputs import load_inputs
from agriopt.maintenance.persistence import read_json


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("maintenance_main", Path(__file__).parents[1] / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module.sys, "version_info", (3, 12, 3))
    monkeypatch.setattr("agriopt.config.load_config", lambda _: {"project": {"run_id": "default"}})
    monkeypatch.chdir(tmp_path)
    return module, tmp_path


def test_launcher_delegates_config_id_workers_and_records_completion(launcher, monkeypatch):
    module, root = launcher
    calls = []

    def existing_runner(config_path, run_id=None, workers=None):
        calls.append((config_path, run_id, workers))
        return root / "outputs" / "maintenance" / run_id

    monkeypatch.setattr(runner, "run_experiment", existing_runner)
    assert module.main(["--config", "configs/maintenance_local.yaml", "--run-id", "mantenimiento_02",
                        "--workers", "2"]) == 0
    assert calls == [("configs/maintenance_local.yaml", "mantenimiento_02", 2)]
    evidence = next(root.glob("outputs/maintenance/mantenimiento_02/launches/*.json"))
    assert read_json(evidence)["status"] == "completed"
    assert read_json(evidence)["global_session_time_limit"] is None


def test_launcher_records_missing_bundle_and_status_is_queryable(launcher, monkeypatch):
    module, root = launcher

    def missing(*args, **kwargs):
        raise FileNotFoundError("Datos_mantenimiento_calibrados.zip is missing")

    monkeypatch.setattr(runner, "run_experiment", missing)
    assert module.main(["--run-id", "missing_data"]) == 1
    folder = root / "outputs" / "maintenance" / "missing_data"
    viewed = runner.status(folder)
    assert viewed["preflight_status"] == "blocked"
    assert "Datos_mantenimiento_calibrados.zip" in viewed["error"]
    assert not (folder / "final").exists()
    assert not (folder / "manifest.json").exists()


def test_launcher_records_operator_interruptions(launcher, monkeypatch):
    module, root = launcher

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(runner, "run_experiment", interrupted)
    assert module.main([]) == 130
    evidence = next(root.glob("outputs/maintenance/default/launches/*.json"))
    assert read_json(evidence)["status"] == "interrupted"


def test_missing_input_error_lists_all_four_files(tmp_path):
    config = {"data": {"productivity": "data/maintenance/productividad_actualizada.csv",
                       "price_history": "data/maintenance/prices_weekly_model_prefix_observed.csv",
                       "price_forecast": "data/maintenance/price_forecast_104w.csv", "demand": None}}
    with pytest.raises(FileNotFoundError) as error:
        load_inputs(config, tmp_path)
    for name in ("productividad_actualizada.csv", "prices_weekly_model_prefix_observed.csv",
                 "price_forecast_104w.csv", "forecast_provenance.json", "Datos_mantenimiento_calibrados.zip"):
        assert name in str(error.value)


def test_heartbeat_refreshes_while_worker_has_not_committed(tmp_path, monkeypatch):
    import threading

    refreshed = threading.Event()
    observations = []
    monkeypatch.setattr(runner, "REPORT_INTERVAL_S", .01)
    monkeypatch.setattr(runner, "_write_index", lambda run: observations.append(run))

    def report(run):
        observations.append(run)
        refreshed.set()

    monkeypatch.setattr("agriopt.maintenance.reporting.build_report", report)
    with runner._report_heartbeat(tmp_path):
        assert refreshed.wait(2), "report never refreshed while the search was busy"
    assert observations[:2] == [tmp_path, tmp_path]


def test_source_identity_covers_entrypoint_and_transitive_modules():
    sources = {name.replace("\\", "/") for name in runner._sources()}
    assert {"main.py", "agriopt/maintenance/search.py", "agriopt/maintenance/inputs.py",
            "agriopt/maintenance/instances.py", "agriopt/model.py", "agriopt/__init__.py"} <= sources

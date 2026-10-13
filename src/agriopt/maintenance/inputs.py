"""Frozen, dated inputs for the maintenance experiments."""
from pathlib import Path

from ..paths import resolve_path
from .io_utils import read_matrix_csv, validate_forecast_provenance
from .productivity import load_productivity


def load_inputs(config: dict, root: Path):
    """Load verified weekly prices and productivity; reject coupled demand.

    The common economic anchor decomposes by plot. Market caps would invalidate
    that decomposition and its normalisation, so they require another protocol.
    """
    data = config.get("data", {})
    if data.get("demand") is not None:
        raise ValueError("Maintenance experiments currently require demand: null")
    names = ("productivity", "price_history", "price_forecast")
    paths = [resolve_path(data.get(name), root) for name in names]
    for name, path in zip(names, paths):
        if path is None or not path.is_file():
            raise FileNotFoundError(f"Missing maintenance input {name}: {path}")
    productivity_path, history_path, forecast_path = paths
    validate_forecast_provenance(history_path, forecast_path)
    productivity = load_productivity(productivity_path)
    history = read_matrix_csv(history_path)
    forecast = read_matrix_csv(forecast_path)
    provenance = forecast_path.parent / "forecast_provenance.json"
    return productivity, history, forecast, None, [*paths, provenance]

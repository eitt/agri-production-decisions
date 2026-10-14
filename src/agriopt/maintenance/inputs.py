"""Frozen, dated inputs for the maintenance experiments."""
from pathlib import Path
import tempfile
import zipfile
import numpy as np

from ..paths import resolve_path
from .io_utils import read_matrix_csv, validate_forecast_provenance, validate_weekly_matrix
from .productivity import load_productivity
from .persistence import atomic_bytes, atomic_json, file_sha256, utc_now

BUNDLE_NAME = "Datos_mantenimiento_calibrados.zip"
INPUT_NAMES = ("productividad_actualizada.csv", "prices_weekly_model_prefix_observed.csv",
               "price_forecast_104w.csv", "forecast_provenance.json")


def import_calibrated_bundle(archive: str | Path, root: Path) -> dict:
    """Validate an entire bundle in staging before installing any input bytes.

    Only the four named files are imported; archive paths are never executed or
    extracted. Ambiguous copies and changes to existing inputs are rejected.
    """
    archive = Path(archive)
    target = root / "data" / "maintenance"
    with zipfile.ZipFile(archive) as zipped:
        members = {}
        for name in INPUT_NAMES:
            candidates = [entry for entry in zipped.infolist()
                          if not entry.is_dir() and Path(entry.filename.replace("\\", "/")).name == name]
            if len(candidates) != 1:
                raise ValueError(f"{archive.name}: expected exactly one {name}; found {len(candidates)}")
            members[name] = zipped.read(candidates[0])  # also validates the ZIP CRC
    for name, content in members.items():
        existing = target / name
        if existing.exists() and existing.read_bytes() != content:
            raise ValueError(f"Refusing to replace existing calibrated input: {existing}. Preserve it separately first.")
    with tempfile.TemporaryDirectory(prefix="agriopt-calibrated-") as temporary:
        staging = Path(temporary)
        for name, content in members.items():
            atomic_bytes(staging / "data" / "maintenance" / name, content)
        stage_config = {"data": dict(zip(("productivity", "price_history", "price_forecast"),
                                        [f"data/maintenance/{name}" for name in INPUT_NAMES[:3]]))}
        load_inputs(stage_config, staging)
        receipt = {"validated_at": utc_now(), "archive": str(archive.resolve()),
                   "archive_sha256": file_sha256(archive),
                   "inputs_sha256": {name: file_sha256(staging / "data" / "maintenance" / name)
                                     for name in INPUT_NAMES}, "forecast_provenance_checked": True}
        for name, content in members.items():
            if not (target / name).exists():
                atomic_bytes(target / name, content)
        atomic_json(target / "bundle_import.json", receipt)
    return receipt


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
    provenance = paths[2].parent / "forecast_provenance.json" if paths[2] is not None else None
    required = [*zip(names, paths), ("forecast_provenance", provenance)]
    missing = [(name, path) for name, path in required if path is None or not path.is_file()]
    standard = [root / "data" / "maintenance" / name for name in INPUT_NAMES]
    if missing and paths + [provenance] == standard and (root / BUNDLE_NAME).is_file():
        import_calibrated_bundle(root / BUNDLE_NAME, root)
        return load_inputs(config, root)
    if missing:
        raise FileNotFoundError(
            "Missing calibrated maintenance inputs:\n" +
            "\n".join(f"  {name}: {path}" for name, path in missing) +
            "\nPlace Datos_mantenimiento_calibrados.zip in the project root and repeat the command; "
            "it will validate and import the four files into data/maintenance/. No forecast is generated or substituted."
        )
    productivity_path, history_path, forecast_path = paths
    validate_forecast_provenance(history_path, forecast_path)
    productivity = load_productivity(productivity_path)
    if productivity.empty:
        raise ValueError("The calibrated productivity file contains no usable empirical observations")
    history = read_matrix_csv(history_path)
    forecast = read_matrix_csv(forecast_path)
    validate_weekly_matrix(history, name="observed price history", min_rows=27)
    validate_weekly_matrix(forecast, name="frozen 104-week forecast", min_rows=104)
    if not np.isfinite(forecast.to_numpy(float)).all() or (forecast.to_numpy(float) <= 0).any():
        raise ValueError("Frozen forecast prices must be finite and positive; no filling is permitted")
    return productivity, history, forecast, None, [*paths, provenance]

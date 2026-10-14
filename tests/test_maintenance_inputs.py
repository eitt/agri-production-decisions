"""Synthetic test fixtures verify import gates; they are never experiment data."""
import hashlib
import json
import zipfile

import pandas as pd
import pytest

from agriopt.maintenance.inputs import BUNDLE_NAME, INPUT_NAMES, import_calibrated_bundle, load_inputs


def bundle_members():
    history = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=30, freq="7D"),
                            "Test crop": [1000.] * 30}).to_csv(index=False).encode()
    forecast = pd.DataFrame({"date": pd.date_range("2025-01-06", periods=104, freq="7D"),
                             "Test crop": [1200.] * 104}).to_csv(index=False).encode()
    provenance = json.dumps({"inputs": {"history": {"sha256": hashlib.sha256(history).hexdigest()}},
                             "output_forecast_sha256": hashlib.sha256(forecast).hexdigest()}).encode()
    productivity = b"Municipio,Producto,Rendimiento,Mes de siembra\nTest,Test crop,1,Enero\n"
    return dict(zip(INPUT_NAMES, (productivity, history, forecast, provenance)))


def write_bundle(path, members):
    with zipfile.ZipFile(path, "w") as zipped:
        for name, content in members.items():
            zipped.writestr(f"data/maintenance/{name}", content)


def test_validated_import_preserves_every_byte_and_checks_both_hashes(tmp_path):
    members = bundle_members()
    archive = tmp_path / BUNDLE_NAME
    write_bundle(archive, members)
    cfg = {"data": dict(zip(("productivity", "price_history", "price_forecast"),
                            [f"data/maintenance/{name}" for name in INPUT_NAMES[:3]]))}
    _, history, forecast, _, paths = load_inputs(cfg, tmp_path)
    assert len(history) == 30 and len(forecast) == 104
    assert len(paths) == 4
    for name, content in members.items():
        assert (tmp_path / "data" / "maintenance" / name).read_bytes() == content
    receipt = import_calibrated_bundle(archive, tmp_path)
    assert receipt["forecast_provenance_checked"]
    assert receipt["inputs_sha256"][INPUT_NAMES[2]] == hashlib.sha256(members[INPUT_NAMES[2]]).hexdigest()


@pytest.mark.parametrize("tampered", [INPUT_NAMES[1], INPUT_NAMES[2]])
def test_invalid_hash_never_installs_any_input(tmp_path, tampered):
    members = bundle_members()
    members[tampered] += b"\n"
    archive = tmp_path / BUNDLE_NAME
    write_bundle(archive, members)
    with pytest.raises(ValueError, match="SHA256"):
        import_calibrated_bundle(archive, tmp_path)
    assert not (tmp_path / "data" / "maintenance").exists()


def test_import_rejects_ambiguous_files_and_keeps_existing_forecast(tmp_path):
    archive = tmp_path / BUNDLE_NAME
    members = bundle_members()
    write_bundle(archive, members)
    existing = tmp_path / "data" / "maintenance" / INPUT_NAMES[2]
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"previous frozen forecast")
    with pytest.raises(ValueError, match="Refusing to replace"):
        import_calibrated_bundle(archive, tmp_path)
    assert existing.read_bytes() == b"previous frozen forecast"
    with zipfile.ZipFile(archive, "a") as zipped:
        zipped.writestr(INPUT_NAMES[2], members[INPUT_NAMES[2]])
    with pytest.raises(ValueError, match="expected exactly one"):
        import_calibrated_bundle(archive, tmp_path)


def test_extra_archive_paths_are_never_extracted(tmp_path):
    archive = tmp_path / BUNDLE_NAME
    write_bundle(archive, bundle_members())
    with zipfile.ZipFile(archive, "a") as zipped:
        zipped.writestr("../unexpected.py", b"not an input")
    import_calibrated_bundle(archive, tmp_path)
    assert not (tmp_path / "unexpected.py").exists()
    assert not (tmp_path.parent / "unexpected.py").exists()

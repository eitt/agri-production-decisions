"""Regression coverage for Windows reader/atomic-replacement conflicts."""
import os
import threading

import pytest

from agriopt.maintenance import persistence


@pytest.mark.skipif(os.name != "nt", reason="Windows sharing semantics")
def test_atomic_writer_retries_until_a_reader_releases_the_old_file(tmp_path):
    target = tmp_path / "partial.json"
    persistence.atomic_json(target, {"generation": 1})
    reader = target.open("rb")
    release = threading.Timer(.1, reader.close)
    release.start()
    try:
        persistence.atomic_json(target, {"generation": 2})
    finally:
        release.join()
        reader.close()
    assert persistence.read_json(target) == {"generation": 2}
    assert not list(tmp_path.glob("*.tmp"))


def test_json_parser_runs_after_the_read_handle_is_closed(tmp_path, monkeypatch):
    target = tmp_path / "partial.json"
    persistence.atomic_json(target, {"generation": 1})
    original = persistence.json.loads

    def parse_while_replacing(contents):
        # This rename would fail on Windows if json.load kept target open.
        persistence.atomic_json(target, {"generation": 2})
        return original(contents)

    monkeypatch.setattr(persistence.json, "loads", parse_while_replacing)
    assert persistence.read_json(target) == {"generation": 1}
    assert original(target.read_bytes()) == {"generation": 2}


@pytest.mark.skipif(os.name != "nt", reason="Windows sharing semantics")
def test_persistent_access_error_preserves_previous_snapshot(tmp_path, monkeypatch):
    target = tmp_path / "partial.json"
    persistence.atomic_json(target, {"old": 1})
    clock = [0.]

    def denied(*args):
        error = PermissionError("persistent Windows access denial")
        error.winerror = 5
        raise error

    monkeypatch.setattr(persistence.os, "replace", denied)
    monkeypatch.setattr(persistence.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(persistence.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    with pytest.raises(PermissionError):
        persistence.atomic_json(target, {"new": 2})
    assert persistence.read_json(target) == {"old": 1}
    assert not list(tmp_path.glob("*.tmp"))

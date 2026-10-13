"""Durable local files for maintenance experiments (including Windows)."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import tempfile
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_value(value: Any) -> Any:
    """Convert NumPy values and non-finite solver diagnostics to portable JSON."""
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(json_value(value), sort_keys=True, ensure_ascii=False, allow_nan=False)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: str | Path, value: str) -> None:
    """Replace only after a complete write; a previous checkpoint survives interruption."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        if os.name != "nt":
            # Persist the rename itself as well as the file contents when the
            # filesystem supports directory fsync (Windows does not).
            try:
                directory = os.open(target.parent, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            except OSError:
                pass
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: str | Path, value: Any) -> None:
    atomic_text(path, json.dumps(json_value(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def read_json(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        # os.kill(pid, 0) is not the POSIX liveness probe on Windows.
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5  # access denied can mean a live protected process
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


@contextmanager
def run_lock(run: str | Path):
    """Prevent concurrent writers. Reclaim only a dead PID on this same computer."""
    folder = Path(run)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / ".run.lock"
    owner = {"pid": os.getpid(), "host": socket.gethostname(), "started_at": utc_now()}
    token = canonical_json(owner)
    for attempt in range(2):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            try:
                previous_text = path.read_text(encoding="utf-8")
                previous = json.loads(previous_text)
            except (OSError, ValueError):
                raise RuntimeError(f"Unreadable run lock: {path}. Check that no experiment is running before removing it.")
            if previous.get("host") != owner["host"] or _process_alive(int(previous.get("pid", -1))):
                raise RuntimeError(f"This run is locked by PID {previous.get('pid')} on {previous.get('host')}: {path}")
            # A changed lock belongs to another launcher. Never delete that lock.
            if attempt or path.read_text(encoding="utf-8") != previous_text:
                raise RuntimeError(f"Run lock changed while starting: {path}")
            path.unlink()
            continue
        else:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(token)
                stream.flush()
                os.fsync(stream.fileno())
            break
    try:
        yield
    finally:
        try:
            if path.read_text(encoding="utf-8") == token:
                path.unlink()
        except FileNotFoundError:
            pass

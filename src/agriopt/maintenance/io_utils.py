from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import hashlib
import json


def validate_forecast_provenance(history_path: str | Path, forecast_path: str | Path) -> None:
    """Refuse forecasts generated from a different historical price file."""
    history_path, forecast_path = Path(history_path), Path(forecast_path)
    provenance_path=forecast_path.parent/"forecast_provenance.json"
    if not provenance_path.exists():
        raise ValueError(f"Forecast provenance is required for rebuilt instances: {provenance_path}")
    metadata=json.loads(provenance_path.read_text(encoding="utf-8"))
    history_hash=hashlib.sha256(history_path.read_bytes()).hexdigest()
    recorded={item.get("sha256") for item in metadata.get("inputs",{}).values()}
    if history_hash not in recorded:
        raise ValueError("Stale forecast: historical price SHA256 differs from forecast provenance")
    expected=metadata.get("output_forecast_sha256")
    if not expected or expected != hashlib.sha256(forecast_path.read_bytes()).hexdigest():
        raise ValueError("Stale or modified forecast: output SHA256 differs from forecast provenance")


def validate_weekly_matrix(df: pd.DataFrame, *, name: str = "price matrix", min_rows: int = 2) -> None:
    """Reject matrices whose dates cannot support a weekly interpretation.

    A missing quote is represented by NaN on its real weekly date, never by
    dropping a date or expanding a monthly quote into invented weekly records.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{name}: a dated weekly index is required")
    if len(df) < min_rows:
        raise ValueError(f"{name}: requires at least {min_rows} weekly rows")
    if df.index.hasnans or df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError(f"{name}: dates must be unique, valid and chronologically increasing")
    days = np.diff(df.index.asi8) / (86400 * 1e9)
    if not np.all(days == 7):
        observed = sorted(set(days.tolist()))
        raise ValueError(f"{name}: expected exactly 7 days between weekly observations; found {observed}")


def causal_training_fill(df: pd.DataFrame, *, max_gap: int = 4) -> pd.DataFrame:
    """Fill short missing quotes using past observations only.

    Leading unavailable periods are removed jointly. No backfill or two-sided
    interpolation is allowed. Call this on training data only before validation.
    """
    if max_gap < 0:
        raise ValueError("max_gap must be nonnegative")
    out = df.copy()
    valid_start = []
    for c in out.columns:
        observed = out[c].dropna()
        if observed.empty:
            raise ValueError(f"No observed training prices for {c}")
        valid_start.append(observed.index[0])
    out = out.loc[max(valid_start):]
    if max_gap:
        out = out.ffill(limit=max_gap)
    if out.isna().any().any():
        bad = out.columns[out.isna().any()].tolist()
        raise ValueError(f"Training gaps exceed causal fill limit {max_gap}: {bad}")
    if not np.isfinite(out.to_numpy(float)).all() or (out.to_numpy(float) <= 0).any():
        raise ValueError("Observed and filled training prices must be finite and positive")
    return out


def read_matrix_csv(path: str | Path) -> pd.DataFrame:
    """Read a wide matrix CSV and remove common date/index columns from the data matrix."""
    p = Path(path)
    df = pd.read_csv(p)
    if df.empty:
        raise ValueError(f"Empty CSV: {p}")
    first = str(df.columns[0]).strip().lower()
    if first in {"week", "date", "fecha", "timestamp", "unnamed: 0"}:
        idx = df.iloc[:, 0]
        df = df.iloc[:, 1:].copy()
        try:
            df.index = pd.to_datetime(idx)
        except Exception:
            df.index = idx
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df.isna().all().any():
        bad = df.columns[df.isna().all()].tolist()
        raise ValueError(f"Columns with no numeric observations in {p}: {bad}")
    df.attrs["source_path"]=str(p.resolve())
    return df


def read_demand_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_csv(p)
    if df.empty:
        raise ValueError(f"Empty demand CSV: {p}")
    first = str(df.columns[0]).strip().lower()
    if first in {"week", "date", "fecha", "timestamp", "unnamed: 0"}:
        idx = df.iloc[:, 0]
        df = df.iloc[:, 1:].copy()
        try:
            df.index = pd.to_datetime(idx)
        except Exception:
            df.index = idx
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

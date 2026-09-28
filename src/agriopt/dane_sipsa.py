from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import json
import re
import time
import unicodedata

import numpy as np
import pandas as pd
import requests
import yaml

DANE_SERIES_URL = "https://www.dane.gov.co/files/operaciones/SIPSA/anex-SIPSA-SerieHistoricaMayorista-{year}.xlsx"
DANE_PRICE_PAGE = "https://www.dane.gov.co/index.php/estadisticas-por-tema/agropecuario/sistema-de-informacion-de-precios-sipsa/mayoristas-boletin-mensual-1"
DANE_WSDL = "http://appweb.dane.gov.co/sipsaWS/SrvSipsaUpraBeanService?WSDL"
DANE_WSDL_ALT = "http://appweb.dane.gov.co:9001/sipsaWS/SrvSipsaUpraBeanService?WSDL"

SOAP_OPERATIONS = {
    "prices_weekly": "promediosSipsaSemanaMadr",
    "prices_monthly": "promediosSipsaMesMadr",
    "supply_monthly": "promedioAbasSipsaMesMadr",
}


def _norm(s: Any) -> str:
    x = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().strip()
    return re.sub(r"[^a-z0-9]+", " ", x).strip()


def _find_col(cols: Iterable[Any], candidates: Iterable[str]) -> Any | None:
    pairs = [(_norm(c), c) for c in cols]
    wanted = [_norm(x) for x in candidates]
    for key, original in pairs:
        if key in wanted:
            return original
    for key, original in pairs:
        if any(w in key for w in wanted):
            return original
    return None


def _http_get(url: str, timeout: int = 90, retries: int = 3) -> requests.Response:
    headers = {
        "User-Agent": "agri-production-decisions/0.3 (+academic reproducibility; DANE SIPSA)",
        "Accept": "*/*",
    }
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout, headers=headers)
            r.raise_for_status()
            return r
        except Exception as exc:  # pragma: no cover - network dependent
            last = exc
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Download failed after {retries} attempts: {url}") from last


def download_year(
    year: int,
    out_dir: str | Path = "data/raw/sipsa_prices",
    timeout: int = 90,
    overwrite: bool = False,
) -> Path:
    """Download DANE SIPSA annual wholesale historical series.

    The filename convention is the one currently published by DANE. Files are cached locally so
    repeated publication runs do not repeatedly request the same official resource.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"SIPSA_wholesale_{int(year)}.xlsx"
    if target.exists() and target.stat().st_size > 0 and not overwrite:
        return target
    url = DANE_SERIES_URL.format(year=int(year))
    r = _http_get(url, timeout=timeout)
    content_type = r.headers.get("content-type", "").lower()
    if len(r.content) < 1000 or ("html" in content_type and not r.content.startswith(b"PK")):
        raise RuntimeError(
            f"DANE returned an unexpected response for {year}. Check the official SIPSA page: {DANE_PRICE_PAGE}"
        )
    target.write_bytes(r.content)
    return target


def download_years(
    years: Iterable[int],
    out_dir: str | Path = "data/raw/sipsa_prices",
    overwrite: bool = False,
) -> list[Path]:
    return [download_year(int(y), out_dir=out_dir, overwrite=overwrite) for y in years]


def normalize_file(
    path: str | Path,
    source_contains: str | None = "Bucaramanga",
    market_contains: str | None = "Centroabastos",
) -> pd.DataFrame:
    """Normalize a DANE historical-series workbook to date/product/source/price.

    Column names are detected rather than hard-coded because DANE workbooks vary across years.
    Filtering is applied to the source/market text only when a recognizable source column exists.
    """
    sheets = pd.read_excel(path, sheet_name=None)
    output: list[pd.DataFrame] = []
    for sheet, df in sheets.items():
        if df.empty:
            continue
        cols = list(df.columns)
        date = _find_col(cols, ["fecha programada", "fecha", "fecha captura", "fecha encuesta", "fecha inicio"])
        product = _find_col(cols, ["nombre articulo", "articulo", "producto", "alimento"])
        source = _find_col(cols, ["nombre de la fuente", "fuen nombre", "fuente", "mercado", "plaza"])
        city = _find_col(cols, ["ciudad", "municipio mercado", "municipio fuente"])
        price = _find_col(cols, ["precio promedio diario", "precio promedio", "promedio kg", "precio", "promedio"])
        if date is None or product is None or price is None:
            continue
        use = [date, product] + ([source] if source is not None else []) + ([city] if city is not None else []) + [price]
        z = df[use].copy()
        rename = {date: "date", product: "product", price: "price"}
        if source is not None:
            rename[source] = "source"
        if city is not None:
            rename[city] = "city"
        z = z.rename(columns=rename)
        z["date"] = pd.to_datetime(z["date"], errors="coerce", dayfirst=True)
        z["price"] = pd.to_numeric(z["price"], errors="coerce")
        z = z.dropna(subset=["date", "product", "price"])
        if source_contains:
            searchable = z.get("source", pd.Series("", index=z.index)).astype(str) + " " + z.get(
                "city", pd.Series("", index=z.index)
            ).astype(str)
            z = z[searchable.str.contains(source_contains, case=False, na=False)]
        if market_contains:
            searchable = z.get("source", pd.Series("", index=z.index)).astype(str) + " " + z.get(
                "city", pd.Series("", index=z.index)
            ).astype(str)
            z = z[searchable.str.contains(market_contains, case=False, na=False)]
        z["sheet"] = str(sheet)
        z["source_file"] = Path(path).name
        output.append(z)
    if not output:
        raise ValueError(f"No recognizable SIPSA wholesale table was found in {path}")
    return pd.concat(output, ignore_index=True).sort_values(["product", "date"])


def weekly_prices(df: pd.DataFrame) -> pd.DataFrame:
    z = df.copy()
    z["week"] = pd.to_datetime(z["date"]).dt.to_period("W-SUN").dt.start_time
    return z.groupby(["week", "product"], observed=True)["price"].mean().unstack("product").sort_index()


def _serialize_soap(result: Any) -> Any:
    try:
        from zeep.helpers import serialize_object

        return serialize_object(result, target_cls=dict)
    except Exception:
        if isinstance(result, (str, int, float, bool, type(None))):
            return result
        if isinstance(result, dict):
            return {k: _serialize_soap(v) for k, v in result.items()}
        if isinstance(result, (list, tuple)):
            return [_serialize_soap(v) for v in result]
        if hasattr(result, "__dict__"):
            return {k: _serialize_soap(v) for k, v in vars(result).items() if not k.startswith("_")}
        return str(result)


def _flatten_records(obj: Any) -> list[dict]:
    if obj is None:
        return []
    if isinstance(obj, list):
        rows: list[dict] = []
        for item in obj:
            rows.extend(_flatten_records(item))
        return rows
    if isinstance(obj, dict):
        scalar = {k: v for k, v in obj.items() if not isinstance(v, (dict, list, tuple))}
        nested = [v for v in obj.values() if isinstance(v, (dict, list, tuple))]
        if scalar and not nested:
            return [scalar]
        rows: list[dict] = []
        if scalar:
            rows.append(scalar)
        for value in nested:
            rows.extend(_flatten_records(value))
        return rows
    return []


def soap_download(
    kind: str,
    out_dir: str | Path = "data/raw/sipsa_soap",
    wsdl: str = DANE_WSDL,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Download a current SIPSA SOAP payload and save both JSON and flattened CSV.

    Requires the optional dependency ``zeep``. The official service currently exposes weekly/monthly
    wholesale prices and monthly food-supply information. Raw payloads are always retained for audit.
    """
    if kind not in SOAP_OPERATIONS:
        raise ValueError(f"Unknown SOAP kind {kind!r}; choose from {sorted(SOAP_OPERATIONS)}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{kind}.json"
    csv_path = out / f"{kind}.csv"
    if json_path.exists() and csv_path.exists() and not overwrite:
        return json_path, csv_path
    try:
        from zeep import Client
        from zeep.transports import Transport
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError('SOAP download requires zeep. Install with: pip install -e ".[data]"') from exc
    session = requests.Session()
    session.headers.update({"User-Agent": "agri-production-decisions/0.3"})
    transport = Transport(session=session, timeout=90, operation_timeout=180)
    last_exc: Exception | None = None
    for candidate in [wsdl, DANE_WSDL_ALT]:
        try:
            client = Client(wsdl=candidate, transport=transport)
            operation = getattr(client.service, SOAP_OPERATIONS[kind])
            result = operation()
            payload = _serialize_soap(result)
            json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            rows = _flatten_records(payload)
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            return json_path, csv_path
        except Exception as exc:  # pragma: no cover - network dependent
            last_exc = exc
    raise RuntimeError(f"Could not query DANE SIPSA SOAP service for {kind}") from last_exc


def load_product_aliases(path: str | Path) -> dict[str, list[str]]:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return {str(k): [str(x) for x in (v or [])] for k, v in raw.items()}


def canonicalize_product(name: Any, aliases: dict[str, list[str]]) -> str | None:
    key = _norm(name)
    if not key:
        return None
    # Exact normalized aliases first.
    for canonical, values in aliases.items():
        candidates = [canonical, *values]
        if key in {_norm(v) for v in candidates}:
            return canonical
    # Conservative substring match for longer names, avoiding very short ambiguous tokens.
    matches: list[tuple[int, str]] = []
    for canonical, values in aliases.items():
        for value in [canonical, *values]:
            v = _norm(value)
            if len(v) >= 5 and (v in key or key in v):
                matches.append((len(v), canonical))
    if not matches:
        return None
    matches.sort(reverse=True)
    return matches[0][1]


def canonical_price_matrix(
    raw: pd.DataFrame,
    aliases: dict[str, list[str]],
    *,
    max_interp_gap: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    z = raw.copy()
    z["canonical_product"] = z["product"].map(lambda x: canonicalize_product(x, aliases))
    coverage = (
        z.assign(mapped=z["canonical_product"].notna())
        .groupby("product", observed=True)
        .agg(n=("price", "size"), mapped=("mapped", "max"), first=("date", "min"), last=("date", "max"))
        .reset_index()
    )
    z = z.dropna(subset=["canonical_product"])
    z["week"] = pd.to_datetime(z["date"]).dt.to_period("W-SUN").dt.start_time
    matrix = z.groupby(["week", "canonical_product"], observed=True)["price"].mean().unstack().sort_index()
    matrix = matrix.replace([np.inf, -np.inf], np.nan).interpolate(limit=max_interp_gap, limit_direction="both")
    return matrix, coverage


def normalize_soap_supply(path: str | Path, city: str = "Bucaramanga", market: str = "Centroabastos") -> pd.DataFrame:
    """Normalize a flattened SOAP supply CSV using schema discovery.

    The function intentionally emits a clear diagnostic when DANE changes field names, while the raw
    payload remains available in data/raw for manual inspection and parser adjustment.
    """
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty SOAP supply payload: {path}")
    cols = list(df.columns)
    date = _find_col(cols, ["fecha ini", "fecha inicio", "fecha", "periodo", "mes"])
    product = _find_col(cols, ["arti nombre", "articulo", "alimento", "producto"])
    source = _find_col(cols, ["fuen nombre", "fuente", "mercado", "central"])
    group = _find_col(cols, ["grupo", "grupo nombre"])
    quantity = _find_col(cols, ["cantidad kg", "cant kg", "promedio kg", "cantidad", "promedio"])
    if quantity is None:
        raise ValueError(f"Could not identify a supply quantity column. Available fields: {cols}")
    use = [quantity] + [c for c in [date, product, source, group] if c is not None]
    z = df[use].copy()
    rename = {quantity: "quantity_kg"}
    for c, name in [(date, "date"), (product, "product"), (source, "source"), (group, "group")]:
        if c is not None:
            rename[c] = name
    z = z.rename(columns=rename)
    z["quantity_kg"] = pd.to_numeric(z["quantity_kg"], errors="coerce")
    z = z.dropna(subset=["quantity_kg"])
    if "source" in z:
        txt = z["source"].astype(str)
        mask = txt.str.contains(city, case=False, na=False)
        if market:
            mask &= txt.str.contains(market, case=False, na=False)
        if mask.any():
            z = z[mask]
    if "date" in z:
        z["date"] = pd.to_datetime(z["date"], errors="coerce", dayfirst=True)
    return z


def update_recent(years=(2025, 2026), out_dir="data/raw/sipsa_prices"):
    frames = []
    for y in years:
        p = download_year(y, out_dir=out_dir)
        frames.append(normalize_file(p))
    raw = pd.concat(frames, ignore_index=True)
    keys = [c for c in ["date", "product", "source"] if c in raw.columns]
    raw = raw.drop_duplicates(keys)
    weekly = weekly_prices(raw)
    out = Path(out_dir)
    raw.to_csv(out / "sipsa_bucaramanga_raw.csv", index=False)
    weekly.to_csv(out / "sipsa_bucaramanga_weekly.csv")
    return raw, weekly

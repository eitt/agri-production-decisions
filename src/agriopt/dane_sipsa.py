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

DANE_SERIES_URLS = {
    2018: "https://www.dane.gov.co/files/investigaciones/agropecuario/sipsa/series-historicas/series-historicas-precios-mayoristas-2018.xlsx",
    2019: "https://www.dane.gov.co/files/investigaciones/agropecuario/sipsa/series-historicas/series-historicas-precios-mayoristas-2019.xlsx",
    2020: "https://www.dane.gov.co/files/investigaciones/agropecuario/sipsa/series-historicas/series-historicas-precios-mayoristas-2020.xlsx",
    2021: "https://www.dane.gov.co/files/investigaciones/agropecuario/sipsa/series-historicas/series-historicas-precios-mayoristas-2021.xlsx",
    2022: "https://www.dane.gov.co/files/investigaciones/agropecuario/sipsa/series-historicas/series-historicas-precios-mayoristas-2022.xlsx",
    2023: "https://www.dane.gov.co/files/operaciones/SIPSA/anex-SIPSA-SerieHistoricaMayorista-Dic2023.xlsx",
    2024: "https://www.dane.gov.co/files/operaciones/SIPSA/anex-SIPSA-SerieHistoricaMayorista-2024.xlsx",
    2025: "https://www.dane.gov.co/files/operaciones/SIPSA/anex-SIPSA-SerieHistoricaMayorista-2025.xlsx",
    2026: "https://www.dane.gov.co/files/operaciones/SIPSA/anex-SIPSA-SerieHistoricaMayorista-2026.xlsx",
}
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
    year = int(year)
    if year not in DANE_SERIES_URLS:
        raise ValueError(
            f"No official SIPSA wholesale-series URL is configured for year {year}. "
            f"Check {DANE_PRICE_PAGE}"
        )

    url = DANE_SERIES_URLS[year]
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

def _parse_sipsa_dates(s: pd.Series) -> pd.Series:
    """Parse SIPSA dates stored either as Excel serials or text dates."""
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    numeric = pd.to_numeric(s, errors="coerce")
    is_excel_serial = numeric.notna() & numeric.between(20000, 80000)

    if is_excel_serial.any():
        out.loc[is_excel_serial] = pd.to_datetime(
            numeric.loc[is_excel_serial],
            unit="D",
            origin="1899-12-30",
            errors="coerce",
        )

    remaining = ~is_excel_serial
    if remaining.any():
        out.loc[remaining] = pd.to_datetime(
            s.loc[remaining],
            errors="coerce",
            dayfirst=True,
        )

    return out


def _parse_sipsa_numbers(s: pd.Series) -> pd.Series:
    """Parse numeric SIPSA values while tolerating Colombian number formats."""
    numeric = pd.to_numeric(s, errors="coerce")

    unresolved = numeric.isna() & s.notna()

    if unresolved.any():
        txt = (
            s.loc[unresolved]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", "", regex=True)
        )

        # Colombian format: 1.234,56 -> 1234.56
        both = txt.str.contains(r"\.", regex=True) & txt.str.contains(",", regex=False)
        txt.loc[both] = (
            txt.loc[both]
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )

        # Decimal comma only: 1234,56 -> 1234.56
        comma_only = ~both & txt.str.contains(",", regex=False)
        txt.loc[comma_only] = txt.loc[comma_only].str.replace(",", ".", regex=False)

        numeric.loc[unresolved] = pd.to_numeric(txt, errors="coerce")

    return numeric

def normalize_file(
    path: str | Path,
    source_contains: str | None = "Bucaramanga",
    market_contains: str | None = "Centroabastos",
) -> pd.DataFrame:
    """Normalize DANE SIPSA historical wholesale-price workbooks.

    SIPSA historical workbooks are not ordinary rectangular datasets.
    Depending on the year, they contain metadata sheets and monthly or
    annual data sheets whose first retained row acts as an embedded header.

    The normalized schema is:
        date, group, product, source, price, sheet, source_file
    """

    path = Path(path)

    # Critical: read without assuming that row 1 contains column names.
    sheets = pd.read_excel(
        path,
        sheet_name=None,
        header=None,
        engine="openpyxl",
    )

    output: list[pd.DataFrame] = []

    for sheet_name, raw in sheets.items():
        if raw is None or raw.empty:
            continue

        df = raw.copy()

        # Remove rows and columns that are mostly empty.
        row_keep = df.notna().mean(axis=1) >= 0.50
        df = df.loc[row_keep]

        if df.empty:
            continue

        col_keep = df.notna().mean(axis=0) >= 0.50
        df = df.loc[:, col_keep]

        df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

        if df.empty or df.shape[1] < 5:
            continue

        # The official SIPSA files use five analytical fields.
        # Additional columns are usually formatting/metadata artefacts.
        df = df.iloc[:, :5].copy()

        # The first retained row is an embedded header/title row in
        # the historical files and is not an observation.
        if len(df) > 1:
            df = df.iloc[1:].copy()

        df.columns = [
            "date",
            "group",
            "product",
            "source",
            "price",
        ]

        df["date"] = _parse_sipsa_dates(df["date"])
        df["price"] = _parse_sipsa_numbers(df["price"])

        df["group"] = df["group"].astype("string").str.strip()
        df["product"] = df["product"].astype("string").str.strip()
        df["source"] = df["source"].astype("string").str.strip()

        # Keep only actual observations.
        df = df.dropna(subset=["date", "product", "price"])

        # Remove rows that may still contain repeated headers.
        header_like = (
            df["product"]
            .astype(str)
            .map(_norm)
            .isin(["alimento", "producto", "articulo"])
        )
        df = df.loc[~header_like]

        if df.empty:
            continue

        # Market information is normally contained in one string such as
        # "Bucaramanga, Centroabastos".
        searchable = df["source"].fillna("").astype(str)

        if source_contains:
            df = df[
                searchable.str.contains(
                    source_contains,
                    case=False,
                    na=False,
                    regex=False,
                )
            ]
            searchable = df["source"].fillna("").astype(str)

        if market_contains:
            df = df[
                searchable.str.contains(
                    market_contains,
                    case=False,
                    na=False,
                    regex=False,
                )
            ]

        if df.empty:
            continue

        df["sheet"] = str(sheet_name)
        df["source_file"] = path.name

        output.append(df)

    if not output:
        # Produce a much more useful diagnostic than the previous error.
        sheet_shapes = {
            str(name): tuple(frame.shape)
            for name, frame in sheets.items()
        }

        raise ValueError(
            f"No SIPSA observations for "
            f"{source_contains!r}/{market_contains!r} were found in {path}. "
            f"Workbook sheets and raw shapes: {sheet_shapes}"
        )

    result = pd.concat(output, ignore_index=True)

    # Monthly sheets sometimes repeat observations already present
    # in previous sheets; retain each actual market observation once.
    result = result.drop_duplicates(
        subset=["date", "group", "product", "source", "price"]
    )

    return result.sort_values(
        ["product", "date", "source"]
    ).reset_index(drop=True)

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

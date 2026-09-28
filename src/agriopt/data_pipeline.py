from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd

from .config import load_config
from .dane_sipsa import (
    download_years,
    normalize_file,
    load_product_aliases,
    canonical_price_matrix,
    soap_download,
    normalize_soap_supply,
    canonicalize_product,
)
from .parameters import CROP_ORDER, SALES_GROUP_NAME
from .paths import repo_root, resolve_path
from .provenance import environment_metadata, write_json


DANE_GROUP_TO_MODEL = {
    "verduras y hortalizas": "VH",
    "frutas": "F",
    "tuberculos raices y platanos": "TRP",
    "tuberculos raices platanos": "TRP",
    "granos y cereales": "OG",
}


def _norm_text(x: Any) -> str:
    import re, unicodedata

    s = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def _weekly_demand_from_supply(
    supply: pd.DataFrame,
    aliases: dict[str, list[str]],
    *,
    start: pd.Timestamp | None = None,
    horizon_weeks: int = 104,
) -> pd.DataFrame:
    """Create a future weekly market-absorption proxy from SIPSA-A monthly supply.

    SIPSA-A measures food inflows to wholesale markets rather than final consumer demand, so the
    resulting cap is explicitly a market-throughput proxy. A month-of-year median profile is estimated
    from observed monthly inflows and repeated over the planning horizon; each monthly total is then
    distributed across the Monday-started weeks touching that month.
    """
    z = supply.copy()
    if "date" not in z.columns or z["date"].isna().all():
        raise ValueError(
            "The SIPSA-A payload did not expose a parseable date. Inspect data/raw/sipsa_soap/supply_monthly.csv."
        )
    z = z.dropna(subset=["date", "quantity_kg"])
    z["model_group"] = None
    if "product" in z.columns:
        z["canonical_product"] = z["product"].map(lambda x: canonicalize_product(x, aliases))
        z.loc[z["canonical_product"].notna(), "model_group"] = z.loc[
            z["canonical_product"].notna(), "canonical_product"
        ].map(SALES_GROUP_NAME)
    if "group" in z.columns:
        broad = z["group"].map(lambda x: DANE_GROUP_TO_MODEL.get(_norm_text(x)))
        z["model_group"] = z["model_group"].fillna(broad)
    z = z.dropna(subset=["model_group"])
    if z.empty:
        raise ValueError("No SIPSA-A rows could be mapped to the four model sales groups.")
    z["month"] = pd.to_datetime(z["date"]).dt.to_period("M")
    monthly = z.groupby(["month", "model_group"], observed=True)["quantity_kg"].sum().unstack(fill_value=0.0)
    for g in ["VH", "F", "TRP", "OG"]:
        if g not in monthly:
            monthly[g] = 0.0
    monthly = monthly[["VH", "F", "TRP", "OG"]].sort_index()

    # Seasonal profile; missing calendar months fall back to the all-month median.
    tmp = monthly.copy()
    tmp["month_number"] = [p.month for p in tmp.index]
    seasonal = tmp.groupby("month_number")[["VH", "F", "TRP", "OG"]].median()
    overall = monthly.median(axis=0)
    for month_number in range(1, 13):
        if month_number not in seasonal.index:
            seasonal.loc[month_number] = overall
    seasonal = seasonal.sort_index()

    if start is None:
        last_observed = monthly.index.max().end_time
        start = (last_observed + pd.Timedelta(days=1)).to_period("W-SUN").start_time
    start = pd.Timestamp(start).to_period("W-SUN").start_time
    weeks = pd.date_range(start=start, periods=int(horizon_weeks), freq="W-MON")
    out = pd.DataFrame(index=weeks, columns=["VH", "F", "TRP", "OG"], dtype=float)
    for w in weeks:
        month = w.to_period("M")
        month_total = seasonal.loc[month.month].to_numpy(float)
        month_start = month.start_time
        month_end = month.end_time
        first_week = month_start.to_period("W-SUN").start_time
        touching = pd.date_range(start=first_week, end=month_end, freq="W-MON")
        out.loc[w] = month_total / max(1, len(touching))
    out.index.name = "week"
    return out


def update_data(config_path: str | Path) -> dict[str, str]:
    cfg = load_config(config_path)
    root = repo_root()
    data_cfg = cfg.get("data", {})
    dl = data_cfg.get("download", {})
    if not dl.get("enabled", False):
        return {"status": "skipped", "reason": "data.download.enabled=false"}

    raw_price_dir = root / "data" / "raw" / "sipsa_prices"
    raw_soap_dir = root / "data" / "raw" / "sipsa_soap"
    interim_dir = root / "data" / "interim"
    processed_dir = root / "data" / "processed"
    for d in [raw_price_dir, raw_soap_dir, interim_dir, processed_dir]:
        d.mkdir(parents=True, exist_ok=True)

    aliases = load_product_aliases(root / "configs" / "products.yaml")
    city = str(data_cfg.get("city", "Bucaramanga"))
    market = str(data_cfg.get("market", "Centroabastos"))
    overwrite = bool(dl.get("overwrite", False))

    years = [int(x) for x in dl.get("price_years", [])]
    if not years:
        raise ValueError("No data.download.price_years were configured")
    files = download_years(years, out_dir=raw_price_dir, overwrite=overwrite)
    price_frames = []
    for p in files:
        try:
            price_frames.append(normalize_file(p, source_contains=city, market_contains=market))
        except ValueError as exc:
            warnings.warn(str(exc))
    if not price_frames:
        raise RuntimeError("No DANE SIPSA wholesale files could be normalized for the requested market")
    raw_prices = pd.concat(price_frames, ignore_index=True)
    raw_prices.to_csv(interim_dir / "prices_market_raw_normalized.csv", index=False)
    prices, coverage = canonical_price_matrix(raw_prices, aliases)
    coverage.to_csv(interim_dir / "price_mapping_coverage.csv", index=False)

    missing = [p for p in CROP_ORDER if p not in prices.columns]
    if missing:
        warnings.warn(
            "Current SIPSA extraction is missing canonical products: " + ", ".join(missing) +
            ". Review configs/products.yaml and data/interim/price_mapping_coverage.csv."
        )
    keep = [p for p in CROP_ORDER if p in prices.columns]
    prices = prices[keep]
    prices.to_csv(processed_dir / "prices_weekly.csv", index_label="week")

    outputs = {
        "prices_weekly": str(processed_dir / "prices_weekly.csv"),
        "price_coverage": str(interim_dir / "price_mapping_coverage.csv"),
    }

    supply_source = str(dl.get("supply_source", "soap"))
    if supply_source == "soap":
        try:
            _, supply_csv = soap_download("supply_monthly", out_dir=raw_soap_dir, overwrite=overwrite)
            supply = normalize_soap_supply(supply_csv, city=city, market=market)
            supply.to_csv(interim_dir / "supply_market_normalized.csv", index=False)
            horizon = int(cfg.get("forecast", {}).get("horizon", 104))
            demand = _weekly_demand_from_supply(supply, aliases, horizon_weeks=horizon)
            demand.to_csv(processed_dir / "demand_weekly.csv", index_label="week")
            outputs["demand_weekly"] = str(processed_dir / "demand_weekly.csv")
        except Exception as exc:
            warnings.warn(
                f"SIPSA-A supply download/preparation was not completed: {exc}. "
                "The optimization can run without this cap, but the publication experiment should document the choice."
            )

    provenance_inputs = [*files]
    write_json(
        environment_metadata(root, provenance_inputs)
        | {
            "official_price_page": "https://www.dane.gov.co/index.php/estadisticas-por-tema/agropecuario/sistema-de-informacion-de-precios-sipsa/mayoristas-boletin-mensual-1",
            "official_soap_wsdl": "http://appweb.dane.gov.co/sipsaWS/SrvSipsaUpraBeanService?WSDL",
            "city": city,
            "market": market,
            "years": years,
        },
        processed_dir / "data_provenance.json",
    )
    return outputs

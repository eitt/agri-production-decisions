from __future__ import annotations
from dataclasses import dataclass,asdict
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from ..model import ProblemData
from .parameters import PRODUCT_SETS,MATURITY_WEEKS,BOTANICAL,SALES_GROUP_NAME,SALES_GROUP_ID
from .productivity import estimate_yield,weekly_sow_calendar,weekly_yield_tensor
from .io_utils import validate_weekly_matrix, validate_forecast_provenance

CATALOG=pd.DataFrame([
 ['I01_micro',1,2,0.6,4,52,'low',101],
 ['I02_small',1,4,1.8,7,52,'medium',102],
 ['I03_diverse',1,8,4.5,11,104,'high',103],
 ['I04_two_farms',2,8,6.0,7,52,'medium',104],
 ['I05_coop_small',5,20,15.0,11,104,'medium',105],
 ['I06_coop_medium',10,40,30.0,11,104,'high',106],
 ['I07_network',20,80,60.0,19,104,'high',107],
 ['I08_network_large',30,150,100.0,19,104,'high',108],
],columns=['instance','farms','lots','total_area_ha','products','horizon_weeks','heterogeneity','seed'])


def weekly_log_return_covariance(history: pd.DataFrame, *, min_observations: int = 26):
    """Shrink covariance of contemporaneous, observed adjacent-week returns.

    The resulting matrix captures cross-crop covariance within one weekly
    return. It does not estimate lagged or intertemporal covariance.
    """
    validate_weekly_matrix(history, name="covariance price history", min_rows=min_observations+1)
    hist=history.to_numpy(float)
    if (hist[np.isfinite(hist)]<=0).any():
        raise ValueError("Historical observed prices must be positive")
    ret=np.diff(np.log(hist),axis=0)
    ret=ret[np.isfinite(ret).all(axis=1)]
    if len(ret)<min_observations:
        raise ValueError(f"Weekly covariance requires at least {min_observations} complete adjacent observed return rows; found {len(ret)}")
    return LedoitWolf().fit(ret).covariance_, len(ret)


def admissible_municipalities(productivity: pd.DataFrame, products: list[str], period_dates: pd.DatetimeIndex,
                              *, min_crop_support: int = 3):
    """Condition synthetic farm sampling on a real nontrivial production option.

    A municipality must have the stated empirical crop support and at least one
    allowed sowing week with positive modelled yield and harvest in the horizon.
    No empirical eligibility or crop maturity is altered to force feasibility.
    """
    T=len(period_dates)
    counts=productivity.groupby(['Municipio','Producto']).size()
    mun_counts=productivity['Municipio'].value_counts()
    municipalities=[str(m) for m in mun_counts.index]
    sow=weekly_sow_calendar(productivity,products,T,expand_sparse=True,period_dates=period_dates)
    tensor=weekly_yield_tensor(productivity,municipalities,products,T,period_dates=period_dates)
    viable={}
    for i,m in enumerate(municipalities):
        viable_crops=[]
        for k,p in enumerate(products):
            available=T-int(MATURITY_WEEKS[p])
            n=int(counts.get((m,p),0))
            if n<min_crop_support or available<=0:
                continue
            active=sow[k,:available] & np.isfinite(tensor[i,k,:available]) & (tensor[i,k,:available]>0)
            if active.any():
                viable_crops.append(p)
        if viable_crops:
            viable[m]=viable_crops
    if not viable:
        raise ValueError("No empirically supported municipality has a positive-yield crop that can be harvested within this horizon")
    allowed=np.asarray([m for m in municipalities if m in viable],dtype=object)
    weights=mun_counts.loc[allowed].to_numpy(float)
    weights/=weights.sum()
    return allowed,weights,viable,sow

def _lot_allocation(farms,lots,total_area_ha,heterogeneity,rng):
    # Allocate total area across farms then lots. Every individual farm remains <=5 ha.
    avg=total_area_ha/farms
    sigma={'low':0.15,'medium':0.4,'high':0.75}[heterogeneity]
    raw=rng.lognormal(np.log(max(avg,0.2)),sigma,size=farms)
    raw=np.minimum(raw,5.0)
    farm_area=raw/raw.sum()*total_area_ha
    # if rescaling exceeds 5, iteratively cap and redistribute
    for _ in range(10):
        over=farm_area>5
        if not over.any():break
        excess=(farm_area[over]-5).sum();farm_area[over]=5
        under=~over
        if under.any():farm_area[under]+=excess*farm_area[under]/farm_area[under].sum()
    counts=np.ones(farms,dtype=int)
    for _ in range(lots-farms): counts[np.argmin(counts/farm_area)]+=1
    areas=[];ids=[]
    for f,(a,n) in enumerate(zip(farm_area,counts)):
        alpha=np.full(n,4.0 if heterogeneity=='low' else (2.0 if heterogeneity=='medium' else 0.8))
        shares=rng.dirichlet(alpha)
        areas.extend((a*10000*shares).tolist());ids.extend([f]*n)
    return np.asarray(areas),np.asarray(ids)

def make_instance(row, productivity, price_forecast:pd.DataFrame, price_history:pd.DataFrame, demand:pd.DataFrame|None=None):
    if not isinstance(row,pd.Series): row=pd.Series(row)
    rng=np.random.default_rng(int(row.seed))
    products=PRODUCT_SETS[int(row.products)]
    T=int(row.horizon_weeks)
    validate_weekly_matrix(price_history, name="instance price history", min_rows=27)
    validate_weekly_matrix(price_forecast, name="instance price forecast", min_rows=T)
    if price_history.attrs.get("source_path") and price_forecast.attrs.get("source_path"):
        validate_forecast_provenance(price_history.attrs["source_path"],price_forecast.attrs["source_path"])
    missing=[p for p in products if p not in price_history.columns or p not in price_forecast.columns]
    if missing:
        raise ValueError(f"Instance {row.instance}: missing supported crop prices: {missing}")
    if price_forecast.index[0] != price_history.index[-1] + pd.Timedelta(days=7):
        raise ValueError("Forecast must begin exactly one week after the last historical observation")
    future_prices=price_forecast[products].iloc[:T].to_numpy(float)
    if not np.isfinite(future_prices).all() or (future_prices<=0).any():
        raise ValueError("Forecast prices must be finite and positive")
    areas,farm_ids=_lot_allocation(int(row.farms),int(row.lots),float(row.total_area_ha),str(row.heterogeneity),rng)
    muns,probs,viable_crops,sow=admissible_municipalities(productivity,products,price_forecast.index[:T])
    farm_mun=rng.choice(muns,size=int(row.farms),replace=True,p=probs)
    municipalities=[str(farm_mun[f]) for f in farm_ids]
    Y=np.zeros((len(areas),len(products)))
    support=np.zeros_like(Y,dtype=int)
    for l,m in enumerate(municipalities):
        for k,p in enumerate(products):
            Y[l,k],support[l,k],_=estimate_yield(productivity,m,p)
    Yweek=weekly_yield_tensor(productivity,municipalities,products,T,period_dates=price_forecast.index[:T])
    P=future_prices.T
    # Risk estimation uses real adjacent quotes only. Missing prices are not
    # carried forward into artificial zero returns for the covariance.
    cov,n_return_rows=weekly_log_return_covariance(price_history[products])
    maturity=np.array([MATURITY_WEEKS[p] for p in products],dtype=int)
    botan=np.array([BOTANICAL[p] for p in products],dtype=object)
    groups=np.array([SALES_GROUP_ID[SALES_GROUP_NAME[p]] for p in products],dtype=int)
    eligible=support>=3
    if not eligible.any(axis=1).all():
        raise ValueError("Selected municipality has no empirically eligible crop; forced eligibility is prohibited")
    demand_arr=None
    if demand is not None:
        # Demand/capacity columns follow the model group labels VH, F, TRP, OG.
        labels=['VH','F','TRP','OG']
        d=demand.copy()
        missing=[g for g in labels if g not in d.columns]
        if missing:
            raise ValueError(f'demand is missing required group columns: {missing}')
        if len(d)<T:
            raise ValueError(f'demand has {len(d)} periods but instance {row.instance} requires {T}')
        demand_arr=d[labels].iloc[:T].to_numpy(float).T
    data=ProblemData(products,areas,farm_ids,municipalities,Y,P,cov,maturity,botan,groups,sow,eligible=eligible,yield_by_sow_week=Yweek,demand=demand_arr)
    metadata={'instance':row.instance,'yield_support_min':int(support.min()),'yield_support_median':float(np.median(support)),'yield_support_zero_share':float((support==0).mean()),'eligible_share':float(eligible.mean()),
              'products':products,'product_count':len(products),'history_frequency':'weekly',
              'covariance_return_unit':'adjacent-week log returns','covariance_observed_rows':int(n_return_rows),
              'covariance_missing_policy':'complete adjacent observed returns only; no return imputation',
              'history_start':str(price_history.index[0].date()),'history_end':str(price_history.index[-1].date()),
              'forecast_start':str(price_forecast.index[0].date()),'forecast_point_interpretation':'conditional median under Gaussian log errors; gross revenue point projection, not profit'}
    metadata.update({'municipality_sampling_policy':'empirical frequency weights conditioned on >=3 local crop observations and at least one positive-yield harvest-feasible allowed sowing date',
                     'admissible_municipalities':muns.tolist(),'admissible_municipality_probabilities':dict(zip(muns.tolist(),probs.tolist())),
                     'admissible_crops_by_municipality':viable_crops,'sampled_farm_municipalities':farm_mun.tolist(),
                     'forced_crop_eligibility':False})
    return data,metadata

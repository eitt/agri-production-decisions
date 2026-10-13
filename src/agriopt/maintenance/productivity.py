from __future__ import annotations
import numpy as np
import pandas as pd

MONTHS=['Enero','Febrero','Marzo','Abril','Mayo','Junio','Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre']
MONTH_TO_I={m:i for i,m in enumerate(MONTHS)}

def load_productivity(path):
    df=pd.read_csv(path)
    df['Rendimiento']=pd.to_numeric(df['Rendimiento'],errors='coerce')
    return df.dropna(subset=['Municipio','Producto','Rendimiento'])

def quality_table(df:pd.DataFrame):
    g=df.groupby('Producto').agg(n=('Rendimiento','size'),municipalities=('Municipio','nunique'),months=('Mes de siembra','nunique'),mean=('Rendimiento','mean'),median=('Rendimiento','median'),sd=('Rendimiento','std'))
    g['cv']=g['sd']/g['mean'].replace(0,np.nan)
    def tier(r):
        if r['n']>=50 and r['municipalities']>=3 and (pd.isna(r['cv']) or r['cv']<=0.5): return 'A'
        if r['n']>=20 and r['municipalities']>=3: return 'B'
        return 'C'
    g['tier']=g.apply(tier,axis=1)
    return g.sort_values(['tier','n'],ascending=[True,False])

def robust_product_center(vals:np.ndarray):
    vals=np.asarray(vals,float)
    if len(vals)==0:return np.nan
    # 10% trimmed mean for n>=20; median for small samples.
    if len(vals)<20:return float(np.median(vals))
    lo,hi=np.quantile(vals,[0.10,0.90])
    z=vals[(vals>=lo)&(vals<=hi)]
    return float(np.mean(z)) if len(z) else float(np.median(vals))

def estimate_yield(df:pd.DataFrame, municipality:str, product:str, tau:float=8.0):
    p=df[df['Producto'].eq(product)]['Rendimiento'].to_numpy(float)
    if len(p)==0:return np.nan,0,'missing'
    center=robust_product_center(p)
    mp=df[df['Producto'].eq(product)&df['Municipio'].eq(municipality)]['Rendimiento'].to_numpy(float)
    if len(mp)==0:return center,0,'product'
    local=robust_product_center(mp)
    w=len(mp)/(len(mp)+tau)
    return float(w*local+(1-w)*center),len(mp),'shrinkage'

def empirical_sow_months(df:pd.DataFrame, product:str):
    x=df[df['Producto'].eq(product)]['Mes de siembra'].dropna().astype(str)
    return [m for m in MONTHS if m in set(x)]

def _period_month_indices(T:int, period_dates=None):
    """Map weekly periods to actual months, with an explicit archival fallback.

    ``None`` preserves the historical week-zero-January approximation for old
    undated fixtures. Empirical runs must pass the actual forecast dates.
    """
    if T < 1:
        raise ValueError("T must be positive")
    if period_dates is None:
        return np.asarray([int((t * 12) // 52) % 12 for t in range(T)], dtype=int)
    if not isinstance(period_dates, pd.DatetimeIndex):
        raise ValueError("period_dates must be a dated weekly DatetimeIndex")
    if (len(period_dates) != T or period_dates.hasnans or period_dates.has_duplicates
            or not period_dates.is_monotonic_increasing):
        raise ValueError("period_dates must contain exactly T unique increasing dates")
    # Date-localized indices preserve calendar months across DST while measuring
    # weekly spacing in calendar days rather than elapsed UTC hours.
    local_dates = period_dates.tz_localize(None) if period_dates.tz is not None else period_dates
    if len(local_dates) > 1 and not np.all(np.diff(local_dates.to_numpy()) == np.timedelta64(7, 'D')):
        raise ValueError("period_dates must be spaced exactly seven calendar days apart")
    return period_dates.month.to_numpy(dtype=int) - 1

def weekly_sow_calendar(df:pd.DataFrame, products:list[str], T:int, expand_sparse:bool=True, period_dates=None):
    # Sparse products (<10 obs) get adjacent months as a sensitivity safeguard.
    period_months=_period_month_indices(T,period_dates)
    out=np.zeros((len(products),T),dtype=bool)
    for i,p in enumerate(products):
        sub=df[df['Producto'].eq(p)]
        months={MONTH_TO_I[m] for m in sub['Mes de siembra'].dropna() if m in MONTH_TO_I}
        if expand_sparse and len(sub)<10:
            months=months | {((m-1)%12) for m in months} | {((m+1)%12) for m in months}
        if not months:
            months=set(range(12))
        for t in range(T):
            month=int(period_months[t])
            if month in months: out[i,t]=True
    return out

def estimate_yield_month(df:pd.DataFrame, municipality:str, product:str, month:str, tau_local:float=8.0, tau_month:float=5.0):
    base,n_local,_=estimate_yield(df,municipality,product,tau=tau_local)
    sub=df[(df['Producto'].eq(product)) & (df['Municipio'].eq(municipality)) & (df['Mes de siembra'].eq(month))]['Rendimiento'].to_numpy(float)
    if len(sub)==0:
        # product-month information still adds seasonal signal if local cell is empty
        pm=df[(df['Producto'].eq(product)) & (df['Mes de siembra'].eq(month))]['Rendimiento'].to_numpy(float)
        if len(pm)==0: return base,0
        c=robust_product_center(pm); w=len(pm)/(len(pm)+tau_month)
        return float(w*c+(1-w)*base),0
    c=robust_product_center(sub); w=len(sub)/(len(sub)+tau_month)
    return float(w*c+(1-w)*base),len(sub)

def weekly_yield_tensor(df:pd.DataFrame, municipalities:list[str], products:list[str], T:int, tau_local:float=8.0, tau_month:float=5.0, period_dates=None):
    # Cached hierarchical shrinkage: product -> municipality-product -> product-month -> municipality-product-month.
    period_months=_period_month_indices(T,period_dates)
    d=df[df['Producto'].isin(products)].copy()
    pg=d.groupby('Producto')['Rendimiento'].agg(['median','count'])
    mg=d.groupby(['Municipio','Producto'])['Rendimiento'].agg(['median','count'])
    pm=d.groupby(['Producto','Mes de siembra'])['Rendimiento'].agg(['median','count'])
    mpm=d.groupby(['Municipio','Producto','Mes de siembra'])['Rendimiento'].agg(['median','count'])
    Y=np.zeros((len(municipalities),len(products),T),float)
    week_month=[MONTHS[i] for i in period_months]
    for l,m in enumerate(municipalities):
        for k,p in enumerate(products):
            base=float(pg.loc[p,'median']) if p in pg.index else np.nan
            if (m,p) in mg.index:
                n=float(mg.loc[(m,p),'count']); loc=float(mg.loc[(m,p),'median']); w=n/(n+tau_local); local=w*loc+(1-w)*base
            else:
                local=base
            month_vals={}
            for mo in set(week_month):
                prior=local
                if (p,mo) in pm.index:
                    n=float(pm.loc[(p,mo),'count']); val=float(pm.loc[(p,mo),'median']); w=n/(n+tau_month); prior=w*val+(1-w)*prior
                if (m,p,mo) in mpm.index:
                    n=float(mpm.loc[(m,p,mo),'count']); val=float(mpm.loc[(m,p,mo),'median']); w=n/(n+tau_month); prior=w*val+(1-w)*prior
                month_vals[mo]=prior
            Y[l,k,:]=[month_vals[mo] for mo in week_month]
    return Y

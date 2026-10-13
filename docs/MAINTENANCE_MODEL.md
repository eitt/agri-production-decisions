# Crop planning with explicit maintenance of idle land

The `agriopt.maintenance` package implements the revised whole-plot model. The
original continuous-area implementation and its configurations remain available
for reproducing the earlier study. Results from those formulations should not be
pooled: they represent different decision spaces and economic objectives.

## Decisions, costs and time conventions

An admissible cultivation event is a plot, crop and sowing week `(l, k, s)`. Its
harvest week is `h = s + maturity[k]`, with `h < T`. The event must have positive
empirical yield, an eligible plot-crop combination, and an allowed sowing week.
The binary decision `y[l,k,s]` uses the entire plot area `A[l]`, measured in m².
Partial cultivation areas are not decision variables in this revised model.

A crop occupies weeks `s <= w < h`. Define weekly crop occupancy as

\[
z_{lw}=\sum_{(k,s):\,s\le w<s+n_k}y_{lks},\qquad z_{lw}\le1.
\]

Crop calendars retain the empirical sowing restrictions, maturity, local crop
eligibility, common preparation interval, and additional rest for the existing
crop-group codes. These historical `BOTANICAL` labels (H/F/T) are broad crop
groups rather than taxonomic botanical families; this implementation preserves
them for comparability. Successive events must satisfy these preparation and
family-release rules. Maintenance does not release a plot for sowing early and
does not reset an agricultural rotation rule.

Let `m[l,w]` be a binary maintenance activity. It takes place on the entire plot
within its indexed week. Cultivation and maintenance cannot coincide:

\[
m_{lw}+z_{lw}\le1.
\]

Every complete window of `H` consecutive weeks must contain cultivation or a
maintenance activity:

\[
\sum_{w=u}^{u+H-1}(z_{lw}+m_{lw})\ge1,
\quad u=0,\ldots,T-H.
\]

Plots start clean at the beginning of the horizon. Cultivation also resets the
untreated-idle counter. Therefore `H = 8` permits at most **seven consecutive
untreated idle weeks**. For a crop-free gap starting at week `b` and lasting `g`
weeks, the minimum-activity canonical schedule maintains at `b+H-1`, `b+2H-1`,
and so forth while inside the gap. It contains `floor(g/H)` activities. Preparation
and rest weeks count as idle and may receive maintenance.

The main assumed cost is **COP 250,000 per ha per maintenance activity**, equal
to **COP 25 per m² per activity**. This is a scenario assumption, not an estimated
local service tariff. Total maintenance cost is

\[
C_M=\sum_{l,w}c\frac{A_l}{10\,000}m_{lw}.
\]

No additional charge per idle week is included: charging both without separate
cost definitions would count the same maintenance concern twice. Activity costs
are charged even on plots with no cultivation during the horizon.

## Optional maximum cultivation gap

An optional limit `G` restricts the number of consecutive weeks without a crop,
regardless of maintenance. It applies to initial, internal and terminal idle
periods, including preparation and rest:

\[
\sum_{w=u}^{u+G}z_{lw}\ge1,
\quad u=0,\ldots,T-G-1.
\]

Maintenance controls untreated land; it does not satisfy this cultivation-gap
constraint. A small `G` can be infeasible because sowing calendars, crop maturity
and rotation remain fixed. The sensitivity suite reports that infeasibility
instead of altering crop eligibility or inventing sowing dates. The default
comparison uses `G = null`: long idle periods are allowed if maintained.

## Economic and risk objectives

Let `q[k,t]` denote production harvested in week `t`, calculated from the selected
whole-plot events and sowing-week yields. At forecast prices `p[k,t]`, gross
receipts are `R = sum(k,t) p[k,t] q[k,t]`. The first objective maximises

\[
R-C_M.
\]

This quantity is **receipts minus maintenance cost**, not farm profit: planting,
labour, irrigation, harvesting, transport and other production costs are absent.
Prices are the frozen wholesale forecast supplied with the experiment.

The second objective minimises contemporaneous covariance exposure:

\[
\sum_t e_t^\top\Sigma e_t,
\qquad e_{kt}=\frac{p_{kt}q_{kt}}{10^6}.
\]

`Sigma` is the shrinkage covariance of complete adjacent-week observed log price
returns. Missing quotes are not converted to artificial zero returns. This
measure sums weekly cross-crop monetary exposure; it is not a forecast of annual
profit variance and does not model covariance across different weeks. Stored
minimisation objectives are `[-net_income_mcop, risk]`. The monetary scale is
normally one million COP; maintenance cost does not enter the price covariance.

## Experiments and interpretation

The fixed catalogue has eight instances, from two plots and 0.6 ha to 150 plots
and 100 ha, with 52- or 104-week horizons. Synthetic farms retain a 5 ha area
limit and are sampled only from empirically supported municipalities. Updated
product sets exclude raw grain crops whose former prices represented processed
products. The frozen 19-product price series must pass input and forecast
provenance checks before an instance is built.

The comparison includes exact reference computation, NSGA-II, random search,
random local refinement, a simple proxy selector, and the learned surrogate
selector. Methods share their instance, initial population, economic anchor,
seeds and objective normalisation. A finite epsilon-constraint calculation is an
exact-solver reference. Each subproblem's status, gap and resource limit must be
reported; these runs do not automatically certify the complete Pareto frontier.
Reports label incomplete tasks as provisional and distinguish solver optima from
time- or memory-limited solutions.

`python -m agriopt maintenance scenarios` runs deterministic, separable dynamic programmes:

| Suite | Parameters | Full-catalogue cases |
|---|---|---:|
| Maintenance sensitivity | `H ∈ {4,8,12}`, `c ∈ {0,100000,250000,500000}` COP/ha/activity, `G ∈ {null,26}` | 192 |
| Hard cultivation-gap diagnostic | `G ∈ {4,8,12,26}`, `H = 8`, `c = 250000` COP/ha/activity | 32 |

For each feasible sensitivity case, the suite compares the reoptimised economic
calendar with the zero-cost economic optimum charged for its required
maintenance afterwards. The former cannot have a lower economic objective,
within numerical tolerance. Risk is evaluated for these calendars, **not
optimised jointly**. These are economic endpoints, not 224 independent stochastic
Pareto trials, and do not require significance tests over seeds.

Every completed case is written atomically under
`outputs/maintenance/<run-id>/scenarios/cases/`. `results.csv`,
`hard_idle_feasibility.csv`, `plans.json` and `progress.json` are refreshed after
each case. A repeated run identifier resumes only if the configuration, input
bytes and model source hashes match. Saved feasible calendars are physically
audited before reuse; malformed cases are recomputed. `audit.json` is written
after the full requested grid has finished. The `experiment.instances` setting
can restrict the suite for a local smoke run; scenario grids can be overridden
under `maintenance.scenarios` without changing the defaults. Earlier top-level
`scenarios` aliases are also supported for configuration compatibility:
`maintenance_intervals_weeks`, `maintenance_costs_cop_ha_pass`, and
`idle_limits_weeks`.

## Scope that requires a further model extension

Maintenance is an instantaneous weekly activity. The model currently has no
labour or equipment capacity for performing it, no measured weed-growth dynamics,
and no cover-crop option. Coupled sales or demand limits are rejected in the
maintenance protocol because they invalidate the separable economic anchor.
Other important extensions include production costs, measured maintenance
tariffs, intertemporal price dependence and carry-over crops. They should be
introduced with new configurations and separate experiment identifiers.

# Literature update and positioning record

The literature revision organizes thirty-one verified and manuscript-used references into five streams covering production planning, non-continuous resource use, agricultural planning, matheuristics, and surrogate-guided allocation. Every bibliography entry supports a nearby claim, while `LITERATURE_AUDIT.csv` records its DOI, verification source, stream assignment, and precise inclusion rationale.

## Positioning decisions

The revised Introduction follows six argumentative stages addressing planning significance, mandatory versus endogenous inactivity, agricultural specificity, computational alternatives, research questions, and bounded contributions. The Related Research section contains three subsections and introduces verified production-planning evidence showing that zero production or inserted idle time can be rational under particular costs, constraints, or temporal incentives.

The central positioning is comparative and integrative because the study evaluates weekly agricultural scheduling, covariance-based price risk, exact-solver evidence, exhaustive continuous refinement, and surrogate-gated refinement under shared controls. The manuscript no longer asserts that no previous study combines every component; instead, it identifies limited controlled evidence about allocating refinement effort within a common evolutionary backbone. The surrogate allocates genuine mathematical-refinement calls and never substitutes reported objective values or determines archive membership through predicted values.

## Verified non-continuous-production evidence

Wagner and Whitin (1958) establish the economic logic of zero-production periods within dynamic lot sizing, while Kanet and Sridharan (2000) provide a verified taxonomy of deliberately inserted idle time. Jodlbauer (2006) connects lot sizing with production scheduling, Trevino-Martinez et al. (2022) show that planned idle time can improve energy-carbon scheduling outcomes, and An et al. (2023) coordinate inserted inactivity with maintenance and time-of-use tariffs.

These sources support only a structural cross-industry claim that inactivity may be purposeful under specific model conditions. They do not provide empirical evidence that manufacturing effect sizes, energy incentives, or preferred schedules transfer directly to agricultural production, whose biological calendars and empirical yield structure remain application-specific.

## Scope controls

The mathematical objectives remain expected income maximization and covariance-based monetary price-risk minimization, while utilization is neither an objective nor a constraint. Space-time utilization, lot temporal occupancy, and mean idle weeks appear only as post-hoc operational diagnostics, and every interpretation distinguishes mandatory agronomic gaps from endogenous waiting, unused slots, or reduced cultivated area.

The fairness revision changes only the standard NSGA-II outputs because its prior implementation received an additional hand-crafted high-production chromosome that the hybrid variants did not receive. The corrected run retains the common zero-production boundary point and uses the same random initialization routine, population, generations, operators, and ten matched seeds as the hybrid methods. Exact, full-refinement, and surrogate-guided outputs remain unchanged, while the primary quality reference becomes the pooled empirical nondominated union and the SCIP-only reference is retained as a sensitivity analysis.

## Audit summary

The bibliography contains thirty-one references spanning 1958 through 2026, including five newly verified sources devoted to non-continuous production or inserted idle time. All retained sources resolve through DOI or publisher records, appear in the manuscript text, and have a single primary stream assignment for audit counting.

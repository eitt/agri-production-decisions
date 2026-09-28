# Publication visualization design

The figure generator applies a message-first workflow inspired by *storytelling with data* practice while preserving scientific auditability.

## Rules used in `agriopt.reporting`

1. One analytical question per figure; no dashboard-style multi-panel composites in the main paper.
2. Titles identify the decision comparison rather than repeating axis labels.
3. Axes use decision units directly, such as million COP, risk, seconds, lots, and exact-refinement calls.
4. Top and right borders are removed, grid lines are limited, and decorative elements are avoided.
5. Method identity is reinforced through marker and line style so interpretation does not depend only on color.
6. The Pareto figure states the preferred movement explicitly because one objective is maximized and the other minimized.
7. Exact, baseline evolutionary, full-refinement, and surrogate-gated methods retain consistent naming across figures and tables.
8. PNG files support review and slides; PDF files preserve vector quality for the manuscript.
9. Seed-level data remain available behind every summarized figure, preventing narrative graphics from hiding stochastic dispersion.

## Recommended main-paper figures

1. **Decision/data architecture** — a manually designed process figure explaining how productivity, price forecasts, risk, and market capacity enter the decision model.
2. **Representative Pareto front** — exact versus NSGA-II versus NSGA-II+QP versus H-NSGA-II on an instance where the exact reference is informative.
3. **Scalability** — runtime against number of lots with logarithmic runtime scaling.
4. **Hybrid ablation** — hypervolume against exact-refinement calls, showing whether the surrogate saves mathematical effort.

The repository automatically generates items 2–4. Figure 1 is best drawn as a conceptual vector diagram after the final data architecture is frozen.

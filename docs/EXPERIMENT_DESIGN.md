# Experiment design

## Controlled scale factors

The benchmark catalog varies the number of farms, lots, products, planning weeks, total area, and within-network heterogeneity. Individual synthetic farms remain within the small-farm scale used by the study, while cooperative/network instances increase the joint planning dimension.

The catalog is versioned at `data/seed/benchmark_instance_catalog.csv` and mirrored by `agriopt.instances.CATALOG`.

## Solver configurations

### Exact multiobjective optimization

SCIP solves a compact mixed-integer quadratically constrained formulation. The income maximum defines the epsilon range, each point minimizes price risk under an income floor, and an optional lexicographic tie-break maximizes income at the attained risk level.

### NSGA-II

The baseline evolutionary method searches crop sequence, timing, and area decisions using nondominated sorting and crowding distance, with an agronomic decoder that prevents infeasible overlaps.

### NSGA-II + QP

The full-refinement matheuristic applies a conditional continuous area-refinement problem to every selected candidate, isolating the benefit of mathematical intensification.

### Surrogate-assisted H-NSGA-II

The hybrid learns realized refinement gain with ExtraTrees, directs most QP calls toward candidates with high predicted improvement, and preserves an exploration share. The surrogate never replaces the final objective evaluation.

## Fair comparisons

Report both solution quality and computational budget. At minimum retain:

- wall-clock runtime;
- normalized hypervolume;
- IGD+;
- additive epsilon;
- number of QP refinement calls;
- QP hit rate;
- exact SCIP gap and status where available.

Use matched stochastic seeds across NSGA-II, NSGA-II+QP, and H-NSGA-II. Exact solutions are deterministic references except for solver randomization settings captured in the run metadata.

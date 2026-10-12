# IJPR submission checklist

## Submission identity

- Confirm final author list, affiliations, ORCID identifiers, and corresponding-author email.
- Confirm special issue selection in the Taylor & Francis submission system.
- Confirm title: **Agricultural Production Planning under Correlated Price Risk: Exact Reference Computation and Surrogate-Guided Continuous Refinement**.
- Remove every visible `PLACEHOLDER` before submission.

## Data layer

- Download and harmonize the final DANE-SIPSA wholesale price history through 2026.
- Build the final SIPSA-A demand series and document aggregation to sales groups.
- Freeze the municipality-product-month productivity dataset and its quality audit.
- Resolve or explicitly exclude anomalous yield distributions identified during quality control.
- Archive the exact instance files used by the computational experiment.
- Describe farms and lots as data-informed synthetic benchmark constructions while preserving the empirical status of productivity and price inputs.

## Forecasting

- Confirm that the manuscript describes the implemented terminal 52-week holdout and never calls it rolling-origin validation.
- Confirm that ARIMA, Gaussian-process, and ARIMA-plus-Gaussian-process candidates match the executable forecasting pipeline.
- Report product-level selected models and holdout errors, together with a concise aggregate table in the manuscript.
- Store the 104-week forecast paths used by the optimization instances and retain their preprocessing provenance.

## Exact SCIP experiment

- Install PySCIPOpt and record SCIP/PySCIPOpt versions.
- Solve I01 first, then I02-I04 while the exact method remains informative.
- Record status, relative gap, nodes, wall-clock time, and all epsilon points.
- Distinguish certified points from time-limited incumbents in the manuscript.
- Preserve SCIP logs in supplementary files.

## Evolutionary and hybrid experiment

- Freeze NSGA-II parameters before final testing.
- Run standard NSGA-II from the shared matched-initialization routine without an additional privileged high-production chromosome.
- Preserve full conditional SLSQP refinement and surrogate-gated H-NSGA-II outputs without rerunning those methods.
- Use ten predeclared matched seeds for every stochastic method in each complete instance.
- Save seed-level archives, histories, refinement logs, and surrogate logs.

## Required metrics

- Hypervolume, IGD+, and additive epsilon against the pooled empirical nondominated reference.
- Sensitivity metrics against the status-aware SCIP incumbent reference.
- Wall-clock time.
- SLSQP refinement calls and cumulative refinement time.
- Useful-refinement hit rate.
- Surrogate prediction error and calibration evidence.
- Exact gap and node count where applicable.
- Paired two-sided Wilcoxon tests with Holm correction, rank-biserial effects, and paired bootstrap intervals.
- Area-adjusted utilization, lot temporal occupancy, and mean idle weeks only as post-hoc diagnostics rather than objectives.

## Main-text figures

1. End-to-end data, decision, exact, evolutionary, and surrogate-assisted architecture.
2. Representative income-risk Pareto fronts with exact and approximate methods under the corrected baseline.
3. Computational scalability and pooled-reference quality-versus-time behaviour.
4. Surrogate ablation showing Pareto quality against SLSQP refinement time.

## Main-text tables

1. Data architecture and quality evidence.
2. Benchmark instances.
3. Compact formulation choices and computational implications.
4. Implemented terminal-holdout forecasting validation and selected models.
5. Exact SCIP performance.
6. NSGA-II / full-refinement / H-NSGA-II comparison under the pooled empirical reference.
7. Surrogate-gating ablation.
8. Reference-front sensitivity and paired seed-level inference.

## Supplementary material

- Product and municipality support tables.
- Full instance definitions and random seeds.
- Complete exact fronts and solver logs.
- All seed-level approximate fronts.
- Full forecasting diagnostics.
- Sensitivity analyses for support thresholds, covariance regularization, demand aggregation, and planting calendars.
- Reproducibility instructions and environment file.

## Final manuscript audit

- Verify every scientific claim against a table, figure, citation, or reproducible output.
- Confirm that the manuscript reads as a standalone study rather than as a correction or extension narrative.
- Check that narrative sentences inside paragraphs do not use short, fragmented prose.
- Recompile twice and visually inspect the complete PDF after final table and figure insertion.
- Confirm disclosure, data availability, funding, and generative-AI statements against current Taylor & Francis requirements.
- Confirm that `generated/method_run_sources.json` and `generated/submission_snapshot.json` identify every physical run source.

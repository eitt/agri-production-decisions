# IJPR submission checklist

## Submission identity

- Confirm final author list, affiliations, ORCID identifiers, and corresponding-author email.
- Confirm special issue selection in the Taylor & Francis submission system.
- Confirm title: **Agricultural Production Decisions under Price Risk: Exact Multiobjective Optimization and Surrogate-Assisted NSGA-II**.
- Remove every visible `PLACEHOLDER` before submission.

## Data layer

- Download and harmonize the final DANE-SIPSA wholesale price history through 2026.
- Build the final SIPSA-A demand series and document aggregation to sales groups.
- Freeze the municipality-product-month productivity dataset and its quality audit.
- Resolve or explicitly exclude anomalous yield distributions identified during quality control.
- Archive the exact instance files used by the computational experiment.

## Forecasting

- Run rolling-origin validation for ARIMA/SARIMA, GP, and ARIMA+GP by product.
- Evaluate GARCH-family volatility only when residual diagnostics justify conditional heteroskedasticity.
- Report the selected model and error metrics for each product.
- Store all forecast paths used by the optimization instances.

## Exact SCIP experiment

- Install PySCIPOpt and record SCIP/PySCIPOpt versions.
- Solve I01 first, then I02-I04 while the exact method remains informative.
- Record status, relative gap, nodes, wall-clock time, and all epsilon points.
- Distinguish certified points from time-limited incumbents in the manuscript.
- Preserve SCIP logs in supplementary files.

## Evolutionary and hybrid experiment

- Freeze NSGA-II parameters before final testing.
- Run standard NSGA-II with matched seeds.
- Run NSGA-II+QP with full conditional SCIP refinement.
- Run surrogate-gated H-NSGA-II with the same refinement model.
- Use 30 seeds when feasible, with the predeclared minimum fallback stated in the manuscript.
- Save seed-level archives, histories, refinement logs, and surrogate logs.

## Required metrics

- Hypervolume.
- IGD+.
- Additive epsilon indicator.
- Wall-clock time.
- QP calls and cumulative QP time.
- Useful-refinement hit rate.
- Surrogate prediction error and calibration evidence.
- Exact gap and node count where applicable.

## Main-text figures

1. End-to-end data, decision, exact, evolutionary, and surrogate-assisted architecture.
2. Representative income-risk Pareto fronts with exact and approximate methods.
3. Computational scalability and quality-versus-time behaviour.
4. Surrogate ablation showing Pareto quality against QP calls and time.

## Main-text tables

1. Data architecture and quality evidence.
2. Benchmark instances.
3. Compact formulation choices and computational implications.
4. Forecasting validation and selected models.
5. Exact SCIP performance.
6. NSGA-II / NSGA-II+QP / H-NSGA-II comparison.
7. Surrogate-gating ablation.

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

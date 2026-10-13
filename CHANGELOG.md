# Changelog

## 0.4.0 — 2026-10-13

- Added the revised whole-plot model with explicit idle-land maintenance, an assumed COP 250,000/ha/pass cost and optional hard cultivation-gap bounds.
- Added verified loading of the local frozen 19-product forecast and historical inputs; numerical inputs remain outside Git.
- Added six controlled comparison methods, generation and wall-clock presets, atomic checkpoints, resume validation and incremental reports.
- Added maintenance-cost sensitivity and hard-gap feasibility experiments, with physical calendar audits.
- Added Windows/VS Code instructions, a PowerShell launcher and vector/raster result exports.
- Shortened labels in the earlier methodology artwork to fit both Arial and the available fallback font.

## 0.3.0 — 2026-09-28

- unified data, forecasting, optimization, and reporting under the `agriopt` command;
- added cached DANE SIPSA wholesale downloads and SOAP connector support;
- added exact SCIP, NSGA-II, full-QP, and surrogate-gated experiment orchestration;
- added resumable run folders with configuration snapshots, hashes, and failure traces;
- added common Pareto metrics and publication table generation;
- added story-first, single-message publication figures with PNG/PDF export;
- added GitHub Actions CI, manual quick reproduction, citation metadata, and repository documentation.


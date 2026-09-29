# Methodology figure provenance and editorial notes

The three methodology figures are generated exclusively by `scripts/generate_methodology_figures.py`, which reads the publication configuration, benchmark catalog, English product-name mapping, and frozen publication snapshot without modifying experimental outputs.

## Figure A — chromosome encoding and decoding

The figure explains the aligned `crops`, `waits`, and `area_frac` matrices implemented by `Individual` in `src/agriopt/nsga2.py`, together with the deterministic event construction performed by `decode_individual`.

The decoder panel represents unused-slot filtering, lot–crop eligibility, previous-harvest availability, setup and same-family recovery, requested waiting, feasible sow-week selection, maturity-based harvest timing, horizon filtering, positive-yield filtering, and event creation.

The illustrative schedule uses English crop labels from `configs/product_names_en.yaml`, although its particular genes and dates are explanatory examples rather than optimization outputs from any experimental seed.

Optional demand limiting remains visually muted because the reported publication configuration does not activate demand caps, while the underlying decoder nevertheless supports that additional constraint.

Suggested manuscript placement is immediately after the opening explanation of the Solution Methods section, because readers need the representation and decoder semantics before comparing algorithmic variants.

## Figure B — shared evolutionary backbone and refinement variants

The shared backbone follows binary tournament selection, crossover, mutation, decoding, fast nondominated sorting, crowding distance, and environmental selection implemented across `src/agriopt/nsga2.py` and `src/agriopt/hybrid_nsga2.py`.

The baseline branch performs no continuous mathematical refinement, whereas the full branch sends every offspring through the conditional area problem implemented in `src/agriopt/refinement.py` using the configured SLSQP backend.

The surrogate-guided branch represents the configured 25 percent refinement budget, five-generation warm-up, thirty-observation training threshold, twenty-percent random exploration share, and two-generation retraining interval.

The Extra Trees model predicts realized refinement gain from seventeen grouped candidate features, so the diagram explicitly identifies it as an access gate rather than a replacement for objective evaluation.

Accepted refinements preserve income within tolerance, prevent risk deterioration, and update true objectives, decoded events, and inherited area-fraction genes before shared Pareto survival.

Detailed mutation probabilities, individual feature names, two hundred trees, the minimum leaf size of two, numerical optimizer tolerances, and initialization edge cases are omitted because including them would obscure the controlled architectural difference among variants.

Suggested manuscript placement follows the surrogate-assisted H-NSGA-II subsection, where the shared backbone and refinement-allocation contrast can consolidate the preceding algorithm descriptions.

## Figure C — computational and experimental workflow

The workflow begins with empirical productivity, DANE SIPSA wholesale-price, and agronomic inputs assembled by `src/agriopt/instances.py` into lot areas, municipalities, eligibility, yield tensors, forecast prices, and covariance structures.

The eight-instance benchmark ladder comes from the implementation catalog, while publication-complete status is read from `artifacts/publication_slsqp_test/paper_snapshot.json` rather than inferred from partially available result directories.

Exact epsilon-constraint SCIP applies to I01–I04 with eleven income thresholds, while the three stochastic methods use matched seeds 2026–2035, population eighty, and one hundred fifty generations.

The evaluation stage summarizes normalized hypervolume, IGD+, additive epsilon, nondominated-point counts, runtime, refinement calls, cumulative refinement time, useful-refinement hit rate, solver status, and solver gap.

The publication-completeness filter includes an instance only after every required method and stochastic seed output is complete, thereby preserving the manuscript’s existing status-aware reporting rule.

Suggested manuscript placement is at the beginning of the Computational Experiment section, although this workflow is the strongest candidate for supplementary relocation if journal page limits become restrictive.

## Export and reproduction

Running `python scripts/generate_methodology_figures.py` regenerates vector PDF and SVG masters plus 300-dpi PNG previews under `artifacts/publication_slsqp_test/figures/methodology/`, while manuscript-ready PDFs are copied under `paper/generated/figures/`.

The generator deliberately reads existing configuration and snapshot evidence without rerunning optimization, rewriting fronts, updating metadata, or invoking the snapshot-generation script.

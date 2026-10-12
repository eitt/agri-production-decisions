# Methodological fairness revision audit

## Version and immutable inputs

- Pre-revision commit: `5ba271a9ebb4d043d53b70d6e22c5af9c295af11`.
- Revision branch: the existing working branch was retained because `paper/main.tex` already contained uncommitted author work before this revision began.
- Original experiment run: `results/publication_slsqp_test/`, treated as immutable throughout the revision.
- Matched-baseline run: `results/publication_nsga2_matched_init/`, containing only newly computed standard NSGA-II outputs.
- Configuration: `configs/publication_slsqp.yaml`.
- Matched stochastic seeds: 2026–2035.
- Complete instances: I01_micro, I02_small, I03_diverse, and I04_two_farms.
- Generation date: 29 September 2026 in the America/Bogota time zone.
- Reporting conversion: 3,306.86 COP/USD, frozen to the manuscript reference date of 28 September 2026.

## Fairness correction and source mapping

The standard NSGA-II implementation formerly replaced a second random individual with an aggressive full-area schedule, whereas both hybrid variants retained that random individual. The correction removes only this baseline-specific schedule and routes all evolutionary variants through one shared initializer that creates the same random population, consumes the same random-number sequence, and injects the same zero-production boundary schedule at position zero.

The submission snapshot resolves method outputs without copying or relabelling them. Exact SCIP, full SLSQP refinement, and surrogate-guided H-NSGA-II are read from the original experiment, while standard NSGA-II is read from the matched-baseline run. The machine-readable mapping is regenerated at `paper/generated/method_run_sources.json`.

## Quality reference and inference

For every complete instance, the primary reference is the nondominated union of all finite, duplicate-free points from the SCIP incumbent archive and every seed-level front from the three stochastic methods. Its ideal point and componentwise spans define normalization, and hypervolume uses the normalized reference point `(1.10, 1.10)`. A secondary sensitivity analysis repeats the indicators against the nondominated SCIP incumbent archive without describing that archive as a certified or true Pareto front.

Seed-level stochastic inference compares NSGA-II, full refinement, and H-NSGA-II through two-sided paired Wilcoxon signed-rank tests. Pairing uses identical seeds, Holm correction is applied to the three method contrasts within each instance and metric, matched-pairs rank-biserial correlation reports effect direction and magnitude, and 95% paired bootstrap intervals describe uncertainty around median paired differences.

## Verification record

Regression tests cover shared initialization semantics, removal of the baseline-only aggressive seed, finite and duplicate front cleaning, empty and singleton fronts, zero-span normalization, pooled-reference construction, primary-versus-secondary reference behaviour, seed pairing, and deterministic bootstrap output. Final compilation, PDF page count, visual inspection findings, and the post-revision working-tree SHA are recorded after artifact regeneration in the completion section below.

## Completion

- Post-revision commit: not created automatically; the reviewed working tree remains based on `5ba271a9ebb4d043d53b70d6e22c5af9c295af11` and is available for author review before committing.
- Final PDF: `paper/main.pdf`, compiled twice after final artifact regeneration; 23 pages.
- Automated verification: 12 tests passed; all 40 matched-baseline runs have status `ok`, with seeds 2026--2035 present for each included instance.
- Visual quality control: all 23 pages were rendered at 160 dpi and inspected, including the title page, equations, Figures 1--7, Tables 1--5, discussion, conclusion, declarations, and references.
- Compilation warnings: no overfull boxes, undefined references, or undefined citations remain. The Taylor & Francis class emits benign Computer Modern size-substitution warnings and several underfull bibliography lines.
- Submission confirmations still required from the authors: funding and generative-AI declarations, author metadata, and replacement of the request-based data statement when a public archival record becomes available.

# Methodology artwork revision — 30 September 2026

Scope: chromosome encoding, NSGA-II variants and experimental workflow only.
No experimental runs, numerical result plots, tables or manuscript prose are changed.

## Visual thesis and figure budget

Retain three independent explanatory figures: representation and decoding;
refinement allocation within the shared evolutionary backbone; and experimental
inputs, completeness filtering, methods and evaluation. No empirical figure is
deleted, combined or moved to the supplement.

## Claim–source–visual mapping

| Figure | Manuscript section | Source and visual purpose |
|---|---|---|
| chromosome_encoding | Solution representation and decoding | Original illustrative matrices and schedule; aligned genes, six numbered decoder steps and three plot timelines |
| metaheuristic_variants | NSGA-II and area refinement; Surrogate-assisted refinement | Publication configuration and hybrid_nsga2 implementation; compare zero, all-offspring and budgeted SLSQP calls, showing true-objective survival |
| experimental_workflow | Experimental design and evaluation | paper_snapshot.json and its configuration_path; distinguish designed and included instances, then computational methods and assessment |

These are method diagrams, not statistical graphics; uncertainty intervals are
inapplicable. The illustrative schedule is explicitly labelled as non-empirical.
The matrices and schedule are schematic examples, not a verified decoding of
one common chromosome. All original illustrative values and dates are retained.

## Publication geometry and reduced density

- Width: 34 TeX picas = approximately 143.40 mm, matching local interact.cls.
- Typeface: Arial, with DejaVu Sans fallback; body text 9 pt, panel headings 10 pt.
- Physical coordinates in millimetres avoid accidental downscaling from a generic
  175-mm canvas into the narrower journal template.
- Box gutters: at least 4 mm vertically and 6 mm horizontally in the repeated layouts.
- Light fills and dark text preserve contrast; labels and position carry meaning
  independently of colour. No gradients, shadows or decorative titles.
- The outer vertical canvas is cropped to the actual artists with a one-point
  safety inset, excluding the invisible axes rectangle; full template width is retained.
- Consolidated decoder rejection criteria replace six subsidiary boxes.
- Horizontal method comparisons replace uneven three-column towers of boxes.
- Benchmark dimensions remain in the manuscript table instead of being repeated
  eight times in the workflow; operational diagnostics remain explicitly post-hoc.
- Optional inactive market-capacity notes remain in manuscript prose rather than
  occupying separate diagram branches.

## Reproduction

Run from the repository root:

```powershell
.venv\Scripts\python.exe scripts\generate_methodology_figures.py --run publication_slsqp_test
.venv\Scripts\python.exe -m pytest tests\test_methodology_figures.py -q
```

`--output-root` supports safe staging without overwriting project artwork.
Only source configuration and publication snapshot metadata are read;
the generator never executes an optimisation or experimental pipeline.

## Exports

PDF masters used by LaTeX retain their original filenames in
`paper/generated/figures` and are copied to `paper/overleaf_ready/generated/figures`.
Editable SVG and raster copies are in
`artifacts/publication_slsqp_test/figures/methodology`.

| Stem | Review PNG, 300 dpi | Publication PNG, 600 dpi |
|---|---|---|
| chromosome_encoding | 1693 × 1955 px | 3387 × 3910 px |
| metaheuristic_variants | 1693 × 1766 px | 3387 × 3532 px |
| experimental_workflow | 1693 × 1659 px | 3387 × 3319 px |

Vector masters: `<stem>.pdf` and `<stem>.svg`.
Rasters: `<stem>.png` and `<stem>_600dpi.png`.
Minor one-pixel rounding differences are expected between dpi profiles.

## Verification

Six regression tests cover all three final-size layouts, minimum font size,
box overlap rejection, text overflow rejection, editable vector export,
physical raster width and removal of exterior padding.
The three 300-dpi previews were inspected visually; final diagram labels fit,
connectors occupy gutters and the original illustrative calendar remains intact.
Existing experimental artwork is protected by before/after SHA-256 comparisons
during installation. Previous generator and explanatory artwork are backed up
under `C:/Users/LEONA/Documents/Playground/figure_revision`.

No Overleaf source or remote assets are changed by this local revision.

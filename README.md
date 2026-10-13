# Agricultural production decisions

Whole-plot crop planning now includes explicit maintenance of uncultivated land,
with an assumed cost of COP 250,000 per hectare and intervention. The default
policy requires maintenance in every eighth consecutive idle week. An optional
hard bound limits consecutive weeks without cultivation.

The new benchmark compares SCIP exact-reference computation, NSGA-II, random
search, random local selection, a simple occupancy proxy and a learned surrogate.
It saves intermediate fronts, complete generation checkpoints, task states and
incremental reports. Reuse the same configuration and run identifier to resume.

```bash
python -m pip install -e ".[scip,dev]"
python -m agriopt maintenance run --config configs/maintenance_smoke.yaml --run-id maintenance_smoke --workers 1
python -m agriopt maintenance status --run maintenance_smoke
python -m agriopt maintenance report --run maintenance_smoke
```

- [Windows and VS Code instructions](docs/VS_CODE_WINDOWS.md)
- [Mathematical model and assumptions](docs/MAINTENANCE_MODEL.md)
- `configs/maintenance_local.yaml`: fixed-generation local benchmark.
- `configs/maintenance_publication.yaml`: 576-task timed benchmark.
- `maintenance scenarios`: resumable maintenance and idle-bound sensitivity.

Load the supplied `Datos_mantenimiento_calibrados.zip` locally before running.
The CSV inputs and forecast manifest belong in `data/maintenance/` and are excluded
from Git. Forecasts are retrospective projections starting in
January 2025, not current market forecasts. Input hashes are checked before use.
The economic objective is gross receipts minus maintenance, not farm profit.

The original `experiment`, `pipeline` and `report` commands remain available for
the earlier continuous-area model. Use the `maintenance` commands for this model.



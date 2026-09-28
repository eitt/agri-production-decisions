# Contributing

Keep changes reproducible and auditable: add or update tests for model/data-schema changes, avoid committing downloaded raw files or generated experiment folders, preserve canonical crop names, and record any change to benchmark logic in `docs/EXPERIMENT_DESIGN.md`.

Before opening a pull request, run:

```bash
python -m compileall -q src
pytest
agriopt pipeline --config configs/quick.yaml
```

Changes that alter objective definitions, feasibility rules, productivity shrinkage, market-data mapping, or Pareto metrics should be described explicitly because they affect scientific comparability across releases.

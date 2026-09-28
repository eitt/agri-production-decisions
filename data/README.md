# Data directory

- `seed/`: versioned inputs that make the repository runnable offline.
- `raw/`: untouched DANE downloads and SOAP payloads; ignored by Git.
- `interim/`: normalized long tables and mapping/forecast audits; ignored by Git.
- `processed/`: model-ready weekly prices, market-capacity proxy, and forecasts; ignored by Git.

The bundled legacy price matrix is a software/reproducibility input rather than the intended final market dataset. Use `agriopt data update` and `agriopt forecast build` before publication runs.

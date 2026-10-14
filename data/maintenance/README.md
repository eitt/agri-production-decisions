# Local maintenance inputs

Place `Datos_mantenimiento_calibrados.zip` in the project root. The maintenance
runner automatically validates the complete bundle in a temporary directory,
then imports these four local files into `data/maintenance/`:

- `productividad_actualizada.csv`
- `prices_weekly_model_prefix_observed.csv`
- `price_forecast_104w.csv`
- `forecast_provenance.json`

Numerical inputs are excluded from Git. Keep the bundle privately with the run.
The portable provenance manifest binds the observed history and forecast by SHA256;
the instance loader also requires dated weekly observations and supported crops.
The supplied forecast is a retrospective projection beginning January 2025.
Changing input bytes requires a new experiment ID and matching provenance.
Existing files with different bytes are never overwritten by the importer.
`bundle_import.json` records the ZIP and input hashes after a successful import.

See [local execution and validation](../../docs/MAINTENANCE_LOCAL_EXECUTION.md)
for the `main.py` entry point, status commands and the current data blocker.

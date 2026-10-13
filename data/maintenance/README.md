# Local maintenance inputs

Extract `Datos_mantenimiento_calibrados.zip` in the project root before running
any `maintenance` command. The bundle creates these local files:

- `productividad_actualizada.csv`
- `prices_weekly_model_prefix_observed.csv`
- `price_forecast_104w.csv`
- `forecast_provenance.json`

Numerical inputs are excluded from Git. Keep the bundle privately with the run.
The portable provenance manifest binds the observed history and forecast by SHA256;
the instance loader also requires dated weekly observations and supported crops.
The supplied forecast is a retrospective projection beginning January 2025.
Changing input bytes requires a new experiment ID and matching provenance.

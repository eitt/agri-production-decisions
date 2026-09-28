# Reproducibility guide

## Fast verification

```bash
pip install -e ".[dev]"
pytest
agriopt pipeline --config configs/quick.yaml
```

The quick pipeline is intentionally offline and uses SLSQP for hybrid refinement.

## Publication reproduction

```bash
pip install -e ".[all]"
agriopt doctor
agriopt pipeline --config configs/publication.yaml
```

For a time-limited reproduction, execute the stages separately so failed downloads or exact solves can be resumed without repeating previous work.

## Run immutability

Every experiment creates `results/<run_id>/`. The `results/LATEST` text file only points to the most recent run; previous run directories are never overwritten unless the user deliberately reuses a run identifier.

## Randomness

Instance generation has fixed benchmark seeds. Evolutionary methods use explicit seeds from the YAML configuration. Adding or removing a seed therefore changes only stochastic replication, not the benchmark instance definition.

## External services

DANE publication layouts and network services may change. Raw files are retained, normalization creates audit tables, and the pipeline stops with a diagnostic when required canonical products cannot be recovered. This is preferable to silently imputing a mislabeled market series.

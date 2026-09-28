#!/usr/bin/env bash
set -euo pipefail
python -m pip install -e ".[dev]"
agriopt doctor
pytest
agriopt pipeline --config configs/quick.yaml

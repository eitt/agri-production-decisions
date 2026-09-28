$ErrorActionPreference = "Stop"
python -m pip install -e ".[dev]"
agriopt doctor
pytest
agriopt pipeline --config configs/quick.yaml

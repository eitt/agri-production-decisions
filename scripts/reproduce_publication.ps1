$ErrorActionPreference = "Stop"
python -m pip install -e ".[all]"
agriopt doctor
agriopt data update --config configs/publication.yaml
agriopt forecast build --config configs/publication.yaml
agriopt experiment --config configs/publication.yaml
agriopt report --run latest

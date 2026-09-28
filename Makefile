PYTHON ?= python

.PHONY: install install-all doctor test quick data forecast exact benchmark report clean

install:
	$(PYTHON) -m pip install -e .

install-all:
	$(PYTHON) -m pip install -e ".[all]"

doctor:
	agriopt doctor

test:
	$(PYTHON) -m pytest

quick:
	agriopt pipeline --config configs/quick.yaml

data:
	agriopt data update --config configs/publication.yaml

forecast:
	agriopt forecast build --config configs/publication.yaml

exact:
	agriopt solve exact --config configs/publication.yaml --instance I01_micro

benchmark:
	agriopt experiment --config configs/publication.yaml

report:
	agriopt report --run latest

clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; [shutil.rmtree(p, ignore_errors=True) for p in [Path('results'), Path('artifacts/figures'), Path('artifacts/tables')]]"
	$(PYTHON) -c "from pathlib import Path; [p.mkdir(parents=True, exist_ok=True) for p in [Path('results'), Path('artifacts/figures'), Path('artifacts/tables')]]"

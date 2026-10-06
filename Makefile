# Shortcuts for the README run steps. Run with the virtual environment activated.
PYTHON ?= python

.PHONY: data features test clean

data:  ## download the raw CSVs into data/raw/ and verify their checksums
	$(PYTHON) scripts/download_data.py

features: data  ## build data/processed/pairs.parquet
	$(PYTHON) -m tennis.dataset

test:  ## run all tests (needs `make features` first)
	$(PYTHON) -m pytest

clean:  ## remove generated files (keeps the downloaded data in data/raw/)
	rm -rf data/processed .pytest_cache notebooks/01_eda.html
	find . -name __pycache__ -type d -not -path "./.venv/*" -prune -exec rm -rf {} +

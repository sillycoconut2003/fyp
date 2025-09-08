ENV?=.venv
PY?=$(ENV)/bin/python

setup:
	python -m venv .venv && $(PY) -m pip install --upgrade pip && $(PY) -m pip install -r requirements.txt

build_processed:
	$(PY) src/features.py

train_ml:
	$(PY) src/train_ml.py

train_ts:
	$(PY) src/train_ts.py

train_all: build_processed train_ml train_ts

dashboard:
	streamlit run dashboard/app.py

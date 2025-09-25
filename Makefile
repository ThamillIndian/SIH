VENV=.venv
PY=$(VENV)/bin/python
PIP=$(VENV)/bin/pip

.PHONY: setup split train val export quant dryrun sim report

setup:
	python -m venv $(VENV); $(PIP) install -U pip
	$(PIP) install -r env/requirements.txt

split:
	$(PY) scripts/split_data.py

train:
	$(PY) scripts/train_yolo_cls.py

val:
	yolo task=classify mode=val model=runs/classify/train/weights/best.pt data=waste_cls

export:
	$(PY) scripts/export_onnx.py

quant:
	$(PY) scripts/quantize_onnx.py

dryrun:
	$(PY) scripts/dryrun_infer.py

sim:
	$(PY) scripts/webcam_simulator.py

report:
	$(PY) scripts/eval_report.py

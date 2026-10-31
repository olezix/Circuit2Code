# Circuit2Code -- reproducible pipeline
#
#   make install     create .venv and install the package + dev dependencies
#   make test        full test suite (trains a couple of tiny models)
#   make test-fast   skips the models that need training
#   make all         train -> extract -> decompile -> verify -> export
#   make paper       build paper/main.pdf (requires a TeX distribution)
#   make clean       remove generated artefacts

VENV   ?= .venv
# Use the project virtualenv when it exists, otherwise whatever `python3` is on
# PATH.  Every other target is then runnable without a build step.
PYTHON ?= $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)
BIN     = $(VENV)/bin
C2C     = $(PYTHON) demo/cli.py

P        ?= 59
EPOCHS   ?= 30000
LR       ?= 1e-2
SCHEDULE ?= cosine
WD       ?= 1.0
#: 0.7 reproduces the grokked reference checkpoint; 0.3 gives the grokking-onset
#: run that stalls at 87% held-out accuracy.
FRACTION ?= 0.7
SEED     ?= 3
D_MODEL  ?= 128
N_HEADS  ?= 4
THREADS  ?= 2

#: name of the checkpoint trained/consumed by extract/decompile/verify/export.
#: Override to use a memoriser:  make decompile CHECKPOINT=artifacts/checkpoints/add59.pt FRACTION=0.3
CHECKPOINT ?= artifacts/checkpoints/g$(P).pt
OUT        ?= artifacts
SLACK      ?= 0.0

.PHONY: help install test test-fast lint train extract decompile verify export demo all paper clean

help:
	@echo "Circuit2Code targets"
	@echo "  install    create $(VENV) and install requirements + package"
	@echo "  test       run the full pytest suite"
	@echo "  test-fast  run the suite without the training fixtures"
	@echo "  lint       py_compile every module and check formatting with ruff (if present)"
	@echo "  train      train the reference model   (P=$(P), frac=$(FRACTION), seed=$(SEED), epochs=$(EPOCHS))"
	@echo "  extract    identify the circuit                 (needs a checkpoint)"
	@echo "  decompile  synthesize + emit Python/C           (needs a checkpoint)"
	@echo "  verify     discharge the Z3 proof obligations   (needs a checkpoint)"
	@echo "  export     alias for decompile into $(OUT)/exported"
	@echo "  demo       tiny end-to-end run: train+extract+decompile+verify+export (~10 s)"
	@echo "  all        train + extract + decompile + verify + export"
	@echo "  paper      build paper/main.pdf"
	@echo "  clean      remove __pycache__, artefacts and build outputs"

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt
	$(BIN)/pip install -e .

install: $(BIN)/python
	@echo "environment ready: $(PYTHON)"

test:
	$(PYTHON) -m pytest tests/ -q

test-fast:
	$(PYTHON) -m pytest tests/ -q -m "not slow"

lint:
	$(PYTHON) -m compileall -q src demo tests scripts
	@if $(PYTHON) -c "import ruff" 2>/dev/null; then $(PYTHON) -m ruff check src demo tests scripts; else echo "ruff not installed -- skipping"; fi

train:
	$(C2C) train --p $(P) --d-model $(D_MODEL) --n-heads $(N_HEADS) \
		--epochs $(EPOCHS) --lr $(LR) --lr-schedule $(SCHEDULE) \
		--weight-decay $(WD) --train-fraction $(FRACTION) --seed $(SEED) \
		--torch-threads $(THREADS) --out $(CHECKPOINT)

extract:
	$(C2C) extract-circuit --checkpoint $(CHECKPOINT) --train-fraction $(FRACTION) \
		--json $(OUT)/circuit.json

decompile:
	$(C2C) decompile --checkpoint $(CHECKPOINT) --train-fraction $(FRACTION) \
		--out $(OUT)/decompiled --json $(OUT)/synthesis.json

verify:
	$(C2C) verify --checkpoint $(CHECKPOINT) --train-fraction $(FRACTION) \
		--slack $(SLACK) --json $(OUT)/verification.json

export:
	$(C2C) export --checkpoint $(CHECKPOINT) --train-fraction $(FRACTION) \
		--out $(OUT)/exported --json $(OUT)/export.json

demo:
	$(C2C) demo --checkpoint $(OUT)/checkpoints/demo.pt --out $(OUT)/demo

all: train extract decompile verify export
	@echo "pipeline complete -- see $(OUT)/*.json"

paper:
	@command -v latexmk >/dev/null 2>&1 || { echo "latexmk not found: install a TeX distribution"; exit 1; }
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	rm -rf artifacts/decompiled artifacts/exported artifacts/*.json .pytest_cache
	cd paper && rm -f main.aux main.log main.out main.bbl main.blg main.fls main.fdb_latexmk

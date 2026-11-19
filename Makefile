# Thin wrapper: every target forwards to run.py, which also works on Windows without make.
PY ?= python

TARGETS = env checksums verify-data cache audit labels baseline-quick baseline-full row-index \
	strict split-ceilings imbalance neural analysis early figures tables numbers results \
	export-model lint test

.PHONY: $(TARGETS)

$(TARGETS):
	$(PY) run.py $@

# Detect root (supports monorepo with publish/ subdir or new repo root)
ROOT:=$(shell [ -d publish/code ] && echo publish || echo .)
PYTHONPATH:=$(ROOT)/code
PY:=PYTHONPATH=$(PYTHONPATH) python

.PHONY: test baseline abtest figs logs report e2e serve-report bench-guacamol bench-moses logo

test:
	$(PY) -m pytest -q $(PYTHONPATH)/tests

baseline:
	$(PY) $(PYTHONPATH)/experiments/run_naive_baseline_raw.py

abtest:
	$(PY) -m experiments.abtest_anti_dup_context

figs:
	$(PY) $(PYTHONPATH)/experiments/plot_abtest_summary.py

logs:
	$(PY) $(PYTHONPATH)/experiments/logs_aggregate.py

report: logs figs
	$(PY) $(PYTHONPATH)/experiments/render_report.py

e2e:
	$(PY) $(PYTHONPATH)/experiments/run_e2e.py

serve-report:
	@echo "Serving report on http://127.0.0.1:8000 (Ctrl+C to stop)"
	cd $(ROOT)/report && python -m http.server 8000

bench-guacamol:
	$(PY) $(PYTHONPATH)/experiments/benchmark_guacamol.py

bench-moses:
	$(PY) $(PYTHONPATH)/experiments/benchmark_moses.py

mini-bench:
	$(PY) -m eurekamesh.cli bench

cli-run:
	$(PY) -m eurekamesh.cli run --mode ccad-basic --n 60

cli-report:
	$(PY) -m eurekamesh.cli report

logo:
	python $(ROOT)/tools/make_logo.py


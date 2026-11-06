#!/bin/bash
# Reproduces, on a laptop and without any model or dataset, the numbers and figures computed from the shipped results.
set -e
PY=${PYTHON:-python3}
cd "$(dirname "$0")"
echo "== Leak test, self-check on two synthetic models (leaky: beta ~ 1; causal: beta ~ 0)"
$PY leak_test/leak_test.py demo
echo; echo "== Leak test on OmniNWM's readouts (paper: beta = 1.00)"
$PY leak_test/leak_test.py score leak_test/omninwm_readouts.csv
echo; echo "== Figures 4, 5, 9, 10, 11, 12 -> figures/*.pdf"
cd figures
$PY figs.py
$PY figs_app.py - - scenes,relation,lookahead,phantom
ls -1 *.pdf

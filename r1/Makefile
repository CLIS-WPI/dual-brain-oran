# COMMAG-R1 Dual-Brain reproduction targets (brief: Dual-Brain R1 developer brief, 2 Nov 2026).
# The Harmony canvas "COMMAG-R1 Dual-Brain" is the testbed; these targets only drive it.
PY ?= python3

.PHONY: versions warm smoke

versions:  ## env/versions.json: Ollama version, container settings, per-model digest + quantization
	$(PY) scripts/ollama_env.py versions

warm:  ## one JSON prompt per model, one model resident on the Ollama GPU at a time
	$(PY) scripts/ollama_env.py warm

BACKEND_PY ?= ../testbed-manager/backend/venv/bin/python
smoke:  ## RAN 3 UEs, KPM rate, stub SLICE control accepted, ML engine both backends, ZTO self-checks, Ollama models
	env -i PATH=$$HOME/bin:/usr/local/bin:/usr/bin:/bin HOME=$$HOME DEPLOYMENT_MODE=k8s K8S_TENANT_NAME=test-1 \
	  $(BACKEND_PY) scripts/smoke.py

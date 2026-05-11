#!/bin/bash
# Setup + launch the ERNIE-Image-Turbo HTTP server on a RunPod Pod.
#
# Usage (inside the Pod web terminal or via SSH):
#   git clone https://github.com/Vulkanone/runpod-ernie-pod.git
#   cd runpod-ernie-pod
#   bash start.sh
#
# Optional env vars before running:
#   export POD_API_KEY="some-secret"    # require Bearer auth on /generate
#   export PORT=8000                    # override port (RunPod proxy exposes any port)
#   export HF_HOME=/workspace/hf-cache  # use big workspace volume for HF cache

set -e

# ── 1. System deps ────────────────────────────────────────────────────────
echo "[start] Installing system deps (git, ca-certificates)…"
apt-get update -qq
apt-get install -y --no-install-recommends git ca-certificates >/dev/null

# ── 2. Python deps ────────────────────────────────────────────────────────
echo "[start] Installing Python deps (may take 2-3 min)…"
pip install --upgrade pip
pip install -r requirements.txt

# ── 3. HuggingFace cache on the persistent workspace volume ───────────────
# Pods have a /workspace mount that survives container restarts. Putting
# the HF cache there means the 16GB ERNIE download persists if you stop
# and restart the pod (saving 5-10 min next time).
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME"
echo "[start] HF_HOME=$HF_HOME"

# ── 4. Launch ─────────────────────────────────────────────────────────────
echo "[start] Launching server.py on port ${PORT:-8000}…"
python server.py

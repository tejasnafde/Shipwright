#!/usr/bin/env bash
# Create .venv and install tau2-bench at a pinned commit (editable, so its data/ dir is found).
set -euo pipefail

TAU2_COMMIT="033bf082f48e4a3c6a1256208e93cf2ce9c3bb6f"
AGENTTUNE_VERSION="1.1.3"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/third_party/tau2-bench"

if [ ! -d "$DEST/.git" ]; then
  mkdir -p "$DEST"
  git -C "$DEST" init -q
  git -C "$DEST" remote add origin https://github.com/sierra-research/tau2-bench.git
fi
if [ "$(git -C "$DEST" rev-parse HEAD 2>/dev/null || true)" != "$TAU2_COMMIT" ]; then
  git -C "$DEST" fetch -q --depth 1 origin "$TAU2_COMMIT"
  git -C "$DEST" checkout -q FETCH_HEAD
fi

cd "$ROOT"
[ -d .venv ] || uv venv -q --python 3.12 .venv
uv pip install -q --python .venv/bin/python -e "$DEST" -e ".[dev]"
# AgentTune drives detect, classify, the retrain trigger and the gate (src/shipwright/heal/agenttune_bridge.py).
# --no-deps: its closed-loop classes need only litellm (from tau2) and pyyaml; its full
# dependency list pins torch 2.11 + bitsandbytes, which do not install on a CPU-only Mac.
uv pip install -q --python .venv/bin/python --no-deps "agenttune==$AGENTTUNE_VERSION"
echo "tau2 installed at $TAU2_COMMIT, agenttune $AGENTTUNE_VERSION"

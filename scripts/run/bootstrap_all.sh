#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$ROOT_DIR"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[error] python interpreter not found: $PYTHON_BIN"
  exit 1
fi

cat <<EOF
[deprecated] scripts/run/bootstrap_all.sh is kept only as a compatibility helper.

Preferred setup:
  conda create -n oracle-kv python=3.10 -y
  conda activate oracle-kv
  python -m pip install -U pip setuptools wheel
  python -m pip install -e '.[dev]'

Optional analysis extras:
  python -m pip install -e '.[dev,analysis]'

Optional uv path:
  uv sync --extra dev
  uv sync --extra dev --extra analysis
EOF

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[setup] creating virtual environment at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  echo "[setup] reusing virtual environment at $VENV_DIR"
fi

echo "[setup] ensuring pip is available"
"$VENV_DIR/bin/python" -m ensurepip --upgrade >/dev/null 2>&1 || true

echo "[setup] upgrading pip/setuptools/wheel"
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel

echo "[setup] installing project in editable mode with dev extras"
"$VENV_DIR/bin/python" -m pip install -e '.[dev]'

echo "[verify] checking root package import"
"$VENV_DIR/bin/python" -c "import src; print('src import ok')"

cat <<EOF
[done] compatibility environment is ready

Use it with:
  source "$VENV_DIR/bin/activate"

Primary long-term setup remains conda + editable pip.
EOF

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$ROOT_DIR"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[error] python interpreter not found: $PYTHON_BIN"
  exit 1
fi

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

echo "[setup] installing dependencies from requirements.txt"
"$VENV_DIR/bin/python" -m pip install -r requirements.txt

echo "[setup] installing root project in editable mode"
"$VENV_DIR/bin/python" -m pip install -e .

echo "[verify] checking tau2 command"
"$VENV_DIR/bin/tau2" --help >/dev/null

echo "[verify] checking root package import"
"$VENV_DIR/bin/python" -c "import src; print('src import ok')"

cat <<EOF
[done] environment is ready

Use it with:
  source "$VENV_DIR/bin/activate"

Quick checks:
  which python
  which tau2
  tau2 --help
EOF
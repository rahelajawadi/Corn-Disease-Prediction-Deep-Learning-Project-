#!/usr/bin/env bash
# Launch the Corn Disease Detection Streamlit app.
#
# Some environments block writes to the user's home directory, which makes
# Streamlit crash when it tries to create ~/.streamlit. Pointing HOME at the
# project directory keeps Streamlit's config inside this (writable) folder.
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="venv-mac/bin/streamlit"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="streamlit"
fi

HOME="$(pwd)" "$PYTHON_BIN" run streamlit_app.py "$@"

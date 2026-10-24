#!/bin/sh
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
if [ -n "${LP_PYTHON:-}" ]; then
    LP_EXECUTABLE=$LP_PYTHON
elif [ -x "$PROJECT_DIR/.venv/bin/python" ]; then
    LP_EXECUTABLE="$PROJECT_DIR/.venv/bin/python"
elif [ -x "$PROJECT_DIR/../../work/minimal_lp_env/bin/python" ]; then
    LP_EXECUTABLE="$PROJECT_DIR/../../work/minimal_lp_env/bin/python"
else
    LP_EXECUTABLE=python3
fi
exec "$LP_EXECUTABLE" "$PROJECT_DIR/benchmark.py" "$@"

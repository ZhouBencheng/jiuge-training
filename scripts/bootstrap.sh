#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$project_root/.cache/bootstrap" "$project_root/.deps"
python -m pip download --no-deps --require-hashes --only-binary=:all: \
  -r "$project_root/requirements-bootstrap.txt" --dest "$project_root/.cache/bootstrap"
if ! PYTHONPATH="$project_root/.deps" python -c 'import pybind11; assert pybind11.__version__ == "2.13.6"'; then
  python -m pip install --no-deps --require-hashes --no-index \
    --find-links "$project_root/.cache/bootstrap" --target "$project_root/.deps" \
    -r "$project_root/requirements-bootstrap.txt" \
    --report "$project_root/.cache/bootstrap/install-report.json"
fi
PYTHONPATH="$project_root/.deps" python -c 'import pybind11; assert pybind11.__version__ == "2.13.6"; print("pybind11", pybind11.__version__, pybind11.__file__)'

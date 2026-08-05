#!/usr/bin/env bash
# Idempotent environment bootstrap for the docs site.
# Installs system prerequisites, creates a Python virtualenv, and installs
# the pinned MkDocs dependencies. Safe to run repeatedly.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# python venv support is not always present on the base image.
if ! python3 -c 'import ensurepip' >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3-venv
fi

if [ ! -x ".venv/bin/python" ]; then
  python3 -m venv .venv
fi

./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt

./.venv/bin/mkdocs --version

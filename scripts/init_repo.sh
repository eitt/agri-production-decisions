#!/usr/bin/env bash
set -euo pipefail
[ -f pyproject.toml ] || { echo "Run this script from the repository root." >&2; exit 1; }
git init
git add .
git commit -m "Initial reproducible research release"
git branch -M main
echo "Repository initialized. Add your GitHub remote, then push main."

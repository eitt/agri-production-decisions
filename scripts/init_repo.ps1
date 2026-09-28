$ErrorActionPreference = "Stop"
if (-not (Test-Path "pyproject.toml")) { throw "Run this script from the repository root." }
git init
git add .
git commit -m "Initial reproducible research release"
git branch -M main
Write-Host "Repository initialized. Add your GitHub remote, then push main."

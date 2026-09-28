# GitHub setup

## First publication

1. Unzip the archive locally; the repository root is the folder containing `pyproject.toml`.
2. Replace `<ORG_OR_USER>/<REPOSITORY>` in `pyproject.toml` and `CITATION.cff`.
3. Verify the redistribution status of the versioned seed data described in `data/README.md`.
4. Initialize and push:

```bash
git init
git add .
git commit -m "Initial reproducible research release"
git branch -M main
git remote add origin https://github.com/<ORG_OR_USER>/<REPOSITORY>.git
git push -u origin main
```

The `.github/workflows/ci.yml` workflow tests Python 3.10–3.12 automatically. The manual `quick-reproduction` workflow runs the complete offline pipeline and uploads its results as a GitHub Actions artifact.

## What should remain outside Git

Raw official downloads, normalized working data, full experiment folders, solver logs, and generated publication artifacts are ignored by default. This keeps the repository reviewable while preserving reproducibility through the downloader, configuration files, hashes, and release archives.

For a paper release, attach the final `results/<run_id>` and `artifacts/<run_id>` folders to a tagged GitHub Release or an external research-data repository rather than committing large generated histories to the main branch.

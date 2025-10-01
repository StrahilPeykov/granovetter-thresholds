## Contributing

Thanks for your interest in contributing! This project aims to make replication of Granovetter (1978) threshold cascades easy and reproducible.

Quick guidelines:
- All changes must go through Pull Requests. Direct pushes to `main` are not allowed.
- At least one approval from a code owner is required (CODEOWNERS: @StrahilPeykov).
- Use Python 3.11 and the `environment.conda.yml` (optional) to ensure reproducibility.
- Keep PRs small and focused; include a short motivation and testing notes.
- Follow the CLI contract documented in `README.md` and `RUNME.md`.
- Add or update tests under `tests/` for any new CLI or output change.
- Figures go in `figures/`; data and logs go in `results/`.

To run tests locally:
```bash
conda env create -f environment.conda.yml
conda activate granovetter
pytest -q
```

Branch protection (enable in GitHub repo settings → Branches → Protect `main`):
- Require a pull request before merging
- Require approvals: 1 (or more)
- Require review from Code Owners
- Require status checks to pass (select CI and Protect-main workflows)
- Include administrators

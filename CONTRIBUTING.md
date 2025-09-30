## Contributing

Thanks for your interest in contributing! This project aims to make replication of Granovetter (1978) threshold cascades easy and reproducible.

Quick guidelines:
- Use Python 3.11 and the `environment.yml` to ensure reproducibility.
- Keep PRs small and focused; include a short motivation and testing notes.
- Follow the CLI contract documented in `README.md` and `RUNME.md`.
- Add or update tests under `tests/` for any new CLI or output change.
- Figures go in `figures/`; data and logs go in `results/`.

To run tests locally:
```bash
conda env create -f environment.yml
conda activate granovetter
pytest -q
```


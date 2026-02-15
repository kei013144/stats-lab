# AGENTS Rules for stats-lab

This repository is a statistics learning lab.
You may freely experiment and make messy changes only in:
- `notebooks/`
- `reports/`

Treat all other paths as controlled project infrastructure.

## Data Rules

- `data/` is edit-prohibited.
- Never hand-edit files inside `data/`.
- If data must change, modify generation code in `scripts/` and regenerate.
- Synthetic datasets must live under `data/synthetic/` and are still treated as raw data.

## Reproducibility Rules

- Fix random seeds for all data generation.
- Reproducibility is the top priority.
- Any seed change must be explicit in command, note, or commit message.

## Synthetic Data Design Policy

- Income, price, and sales style variables should be right-skewed (for example lognormal).
- Height-like biometric variables should be approximately normal.
- Include both MCAR missingness and biased missingness with MNAR-like behavior.
- Inject a small amount of outliers.
- Embed testable effects such as A/B differences or group mean differences.

## Recommended Workflow

1. Run `make data` to generate synthetic data.
2. Analyze in notebooks with plots and summary statistics.
3. Save one figure and one-line summary in `reports/`.

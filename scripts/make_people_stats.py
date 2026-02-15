#!/usr/bin/env python
"""Generate a synthetic people-level dataset for statistics exercises."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_people_stats(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    sex = rng.choice(["F", "M"], size=n, p=[0.49, 0.51])
    region = rng.choice(
        ["north", "east", "west", "south", "central"],
        size=n,
        p=[0.16, 0.26, 0.22, 0.18, 0.18],
    )
    education = rng.choice(
        ["high_school", "vocational", "bachelor", "master", "phd"],
        size=n,
        p=[0.32, 0.16, 0.33, 0.15, 0.04],
    )
    age = np.clip(np.rint(rng.normal(42, 13, size=n)).astype(int), 18, 85)

    # Height is mostly normal with group-level center differences.
    height_mean = np.where(sex == "M", 171.0, 158.2)
    height_sd = np.where(sex == "M", 6.2, 5.8)
    height_cm = rng.normal(height_mean, height_sd, size=n)

    # Income is intentionally right-skewed.
    base_income = rng.lognormal(mean=np.log(420.0), sigma=0.55, size=n)
    edu_multiplier = pd.Series(education).map(
        {
            "high_school": 0.85,
            "vocational": 0.95,
            "bachelor": 1.15,
            "master": 1.30,
            "phd": 1.45,
        }
    ).to_numpy()
    sex_multiplier = np.where(sex == "M", 1.08, 1.00)
    age_centered = age - 40
    age_multiplier = 1.0 + 0.012 * age_centered - 0.00018 * (age_centered**2)
    age_multiplier = np.clip(age_multiplier, 0.65, 1.35)
    income_man = base_income * edu_multiplier * sex_multiplier * age_multiplier

    # Outliers: a few extreme high-income cases and abnormal heights.
    income_out_n = max(1, int(n * 0.003))
    income_out_idx = rng.choice(n, size=income_out_n, replace=False)
    income_man[income_out_idx] *= rng.uniform(3.0, 8.5, size=income_out_n)

    height_out_n = max(1, int(n * 0.002))
    height_out_idx = rng.choice(n, size=height_out_n, replace=False)
    height_cm[height_out_idx] = rng.choice(
        np.array([118.0, 124.0, 210.0, 224.0]), size=height_out_n
    ) + rng.normal(0.0, 1.2, size=height_out_n)

    # Missingness: exact-rate MCAR for height, and MCAR + MNAR-like weighted missing for income.
    height_missing_rate = float(rng.uniform(0.005, 0.010))
    height_missing_n = max(1, int(round(n * height_missing_rate)))
    height_missing_idx = rng.choice(n, size=height_missing_n, replace=False)
    height_cm[height_missing_idx] = np.nan

    income_missing_rate = 0.010
    income_missing_n = max(1, int(round(n * income_missing_rate)))
    high_income_cut = np.quantile(income_man, 0.85)
    income_weights = np.full(n, 1.0, dtype=float)
    income_weights += np.where(income_man >= high_income_cut, 2.5, 0.0)
    income_weights /= income_weights.sum()
    income_missing_idx = rng.choice(n, size=income_missing_n, replace=False, p=income_weights)
    income_man[income_missing_idx] = np.nan

    df = pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "age": age,
            "sex": sex,
            "region": region,
            "education": education,
            "height_cm": np.round(height_cm, 1),
            "income_man": np.round(income_man, 1),
        }
    )
    return df


def print_sanity(df: pd.DataFrame) -> None:
    income_non_missing = df["income_man"].dropna()

    print(f"[people_stats] rows: {len(df)}")
    print(f"[people_stats] height_cm missing rate: {df['height_cm'].isna().mean():.4%}")
    print(f"[people_stats] income_man missing rate: {df['income_man'].isna().mean():.4%}")
    print(
        "[people_stats] income_man summary: "
        f"mean={income_non_missing.mean():.1f}, "
        f"median={income_non_missing.median():.1f}, "
        f"p95={income_non_missing.quantile(0.95):.1f}"
    )

    print("[people_stats] mean income_man by sex:")
    print(df.groupby("sex", observed=True)["income_man"].mean().round(1).to_string())

    print("[people_stats] mean income_man by education:")
    print(df.groupby("education", observed=True)["income_man"].mean().round(1).to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic people_stats.csv")
    parser.add_argument("--n", type=int, default=10000, help="Number of rows")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data/synthetic"),
        help="Output directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n <= 0:
        raise ValueError("--n must be positive.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_people_stats(n=args.n, seed=args.seed)
    out_path = out_dir / "people_stats.csv"
    df.to_csv(out_path, index=False)

    print(f"[people_stats] wrote: {out_path}")
    print_sanity(df)


if __name__ == "__main__":
    main()

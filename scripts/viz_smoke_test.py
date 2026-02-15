#!/usr/bin/env python
"""Create reproducible smoke-test plots from synthetic CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FILE_SPECS = {
    "people_stats.csv": ["age", "sex", "region", "education", "height_cm", "income_man"],
    "customers.csv": ["age", "region", "income_man", "price_sens", "quality_pref", "digital_pref"],
    "products.csv": ["category", "price_yen", "quality_score"],
    "orders.csv": ["order_date", "ab_group", "purchased", "revenue_yen", "returned"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate smoke-test PNG plots from synthetic CSVs.")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/synthetic"),
        help="Input CSV directory (default: data/synthetic)",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory (default: auto detect 04_reports/smoke_test or reports/smoke_test)",
    )
    parser.add_argument("--dpi", type=int, default=150, help="PNG DPI")
    return parser.parse_args()


def resolve_data_dir(requested: Path) -> Path:
    requested = Path(requested)
    if requested.exists():
        if not requested.is_dir():
            raise NotADirectoryError(f"data_dir is not a directory: {requested}")
        return requested

    for fallback in [Path("data/synthetic"), Path("01_data/synthetic")]:
        if fallback.exists() and fallback.is_dir():
            print(
                f"[warn] --data_dir '{requested}' not found. "
                f"Falling back to '{fallback}'."
            )
            return fallback

    raise FileNotFoundError(
        f"No synthetic data directory found. Tried: '{requested}', "
        "'data/synthetic', '01_data/synthetic'."
    )


def resolve_out_dir(requested: Optional[Path]) -> Path:
    if requested is not None:
        out_dir = Path(requested)
    else:
        if Path("04_reports").exists():
            out_dir = Path("04_reports") / "smoke_test"
        elif Path("reports").exists():
            out_dir = Path("reports") / "smoke_test"
        else:
            out_dir = Path("reports") / "smoke_test"

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_csvs(data_dir: Path) -> dict[str, pd.DataFrame]:
    loaded: dict[str, pd.DataFrame] = {}

    for file_name in FILE_SPECS:
        csv_path = data_dir / file_name
        if not csv_path.exists():
            print(f"[warn] missing CSV, skip: {csv_path}")
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:  # pragma: no cover - defensive branch
            print(f"[warn] failed to read {csv_path}: {exc}")
            continue

        loaded[file_name] = df
        print(f"[load] {file_name}: rows={len(df):,}")

    if loaded:
        found_list = ", ".join(sorted(loaded.keys()))
        print(f"[info] found CSVs: {found_list}")
    else:
        print("[warn] no expected CSVs found.")

    return loaded


def print_missing_summary(name: str, df: pd.DataFrame, columns: list[str]) -> None:
    print(f"[{name}] rows={len(df):,}")

    major_cols = [c for c in columns if c in df.columns]
    if not major_cols:
        print(f"[{name}] major columns not present for missing-rate summary.")
        return

    missing_pct = (df[major_cols].isna().mean() * 100).sort_values(ascending=False)
    summary = ", ".join([f"{col}={pct:.2f}%" for col, pct in missing_pct.items()])
    print(f"[{name}] missing rate: {summary}")


def save_plot(fig: plt.Figure, out_path: Path, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")


def plot_numeric_hist(
    series: pd.Series,
    out_path: Path,
    title: str,
    xlabel: str,
    dpi: int,
    use_log: bool = False,
) -> bool:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if use_log:
        vals = vals[vals > 0]
        vals = np.log(vals)
    if vals.empty:
        print(f"[warn] skip plot (no valid values): {out_path.name}")
        return False

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(vals, bins=30, color="#4e79a7", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    save_plot(fig, out_path, dpi)
    return True


def plot_people(df: pd.DataFrame, out_dir: Path, dpi: int) -> int:
    created = 0

    missing_targets = ["age", "height_cm", "income_man", "sex", "region", "education"]
    existing_targets = [c for c in missing_targets if c in df.columns]
    if existing_targets:
        miss = (df[existing_targets].isna().mean() * 100).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(9, 5))
        miss.plot(kind="bar", ax=ax, color="#59a14f")
        ax.set_title("People: Missing Rate")
        ax.set_xlabel("Column")
        ax.set_ylabel("Missing rate (%)")
        ax.tick_params(axis="x", rotation=30)
        save_plot(fig, out_dir / "people_missing_rate.png", dpi)
        created += 1
    else:
        print("[warn] people missing-rate plot skipped: target columns not found")

    if "height_cm" in df.columns:
        if plot_numeric_hist(
            series=df["height_cm"],
            out_path=out_dir / "people_height_hist.png",
            title="People: Height Distribution",
            xlabel="height_cm",
            dpi=dpi,
            use_log=False,
        ):
            created += 1
    else:
        print("[warn] people height histogram skipped: height_cm missing")

    if "income_man" in df.columns:
        if plot_numeric_hist(
            series=df["income_man"],
            out_path=out_dir / "people_income_hist_raw.png",
            title="People: Income Distribution (Raw)",
            xlabel="income_man",
            dpi=dpi,
            use_log=False,
        ):
            created += 1

        if plot_numeric_hist(
            series=df["income_man"],
            out_path=out_dir / "people_income_hist_log.png",
            title="People: Income Distribution (log)",
            xlabel="log(income_man)",
            dpi=dpi,
            use_log=True,
        ):
            created += 1
    else:
        print("[warn] people income histograms skipped: income_man missing")

    if {"income_man", "education"}.issubset(df.columns):
        box_df = df[["income_man", "education"]].copy()
        box_df["income_man"] = pd.to_numeric(box_df["income_man"], errors="coerce")
        box_df = box_df.dropna(subset=["income_man", "education"])

        if not box_df.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            box_df.boxplot(column="income_man", by="education", ax=ax, grid=False, rot=30)
            fig.suptitle("")
            ax.set_title("People: Income by Education")
            ax.set_xlabel("education")
            ax.set_ylabel("income_man")
            save_plot(fig, out_dir / "people_income_box_by_education.png", dpi)
            created += 1
        else:
            print("[warn] people income boxplot skipped: no valid rows")
    else:
        print("[warn] people income boxplot skipped: income_man or education missing")

    if {"height_cm", "sex"}.issubset(df.columns):
        box_df = df[["height_cm", "sex"]].copy()
        box_df["height_cm"] = pd.to_numeric(box_df["height_cm"], errors="coerce")
        box_df = box_df.dropna(subset=["height_cm", "sex"])

        if not box_df.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            box_df.boxplot(column="height_cm", by="sex", ax=ax, grid=False)
            fig.suptitle("")
            ax.set_title("People: Height by Sex")
            ax.set_xlabel("sex")
            ax.set_ylabel("height_cm")
            save_plot(fig, out_dir / "people_height_box_by_sex.png", dpi)
            created += 1
        else:
            print("[warn] people height boxplot skipped: no valid rows")
    else:
        print("[warn] people height boxplot skipped: height_cm or sex missing")

    if {"age", "income_man"}.issubset(df.columns):
        scatter_df = df[["age", "income_man"]].copy()
        scatter_df["age"] = pd.to_numeric(scatter_df["age"], errors="coerce")
        scatter_df["income_man"] = pd.to_numeric(scatter_df["income_man"], errors="coerce")
        scatter_df = scatter_df.dropna()

        if not scatter_df.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(scatter_df["age"], scatter_df["income_man"], s=12, alpha=0.35, color="#e15759")
            ax.set_title("People: Age vs Income")
            ax.set_xlabel("age")
            ax.set_ylabel("income_man")
            save_plot(fig, out_dir / "people_age_vs_income_scatter.png", dpi)
            created += 1

    return created


def plot_products(df: pd.DataFrame, out_dir: Path, dpi: int) -> int:
    created = 0

    if "price_yen" in df.columns:
        if plot_numeric_hist(
            series=df["price_yen"],
            out_path=out_dir / "products_price_hist_raw.png",
            title="Products: Price Distribution (Raw)",
            xlabel="price_yen",
            dpi=dpi,
            use_log=False,
        ):
            created += 1
    else:
        print("[warn] products price histogram skipped: price_yen missing")

    if {"price_yen", "category"}.issubset(df.columns):
        box_df = df[["price_yen", "category"]].copy()
        box_df["price_yen"] = pd.to_numeric(box_df["price_yen"], errors="coerce")
        box_df = box_df.dropna(subset=["price_yen", "category"])

        if not box_df.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            box_df.boxplot(column="price_yen", by="category", ax=ax, grid=False, rot=30)
            fig.suptitle("")
            ax.set_title("Products: Price by Category")
            ax.set_xlabel("category")
            ax.set_ylabel("price_yen")
            save_plot(fig, out_dir / "products_price_box_by_category.png", dpi)
            created += 1
        else:
            print("[warn] products boxplot skipped: no valid rows")
    else:
        print("[warn] products boxplot skipped: price_yen or category missing")

    return created


def plot_orders(df: pd.DataFrame, out_dir: Path, dpi: int) -> int:
    created = 0

    if "revenue_yen" in df.columns:
        if plot_numeric_hist(
            series=df["revenue_yen"],
            out_path=out_dir / "orders_revenue_hist_log.png",
            title="Orders: Revenue Distribution (log)",
            xlabel="log(revenue_yen)",
            dpi=dpi,
            use_log=True,
        ):
            created += 1
    else:
        print("[warn] orders revenue histogram skipped: revenue_yen missing")

    if {"order_date", "revenue_yen"}.issubset(df.columns):
        daily_df = df[["order_date", "revenue_yen"]].copy()
        daily_df["order_date"] = pd.to_datetime(daily_df["order_date"], errors="coerce")
        daily_df["revenue_yen"] = pd.to_numeric(daily_df["revenue_yen"], errors="coerce")
        daily_df = daily_df.dropna(subset=["order_date", "revenue_yen"])

        if not daily_df.empty:
            daily_sales = (
                daily_df.groupby(daily_df["order_date"].dt.floor("D"), observed=True)["revenue_yen"]
                .sum()
                .sort_index()
            )

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(daily_sales.index, daily_sales.to_numpy(), color="#f28e2b", linewidth=1.6)
            ax.set_title("Orders: Daily Revenue")
            ax.set_xlabel("order_date")
            ax.set_ylabel("sum(revenue_yen)")
            save_plot(fig, out_dir / "orders_daily_revenue.png", dpi)
            created += 1
        else:
            print("[warn] orders daily revenue plot skipped: no valid rows")
    else:
        print("[warn] orders daily revenue plot skipped: order_date or revenue_yen missing")

    if "ab_group" in df.columns:
        metrics: list[tuple[str, pd.Series, str]] = []

        if "revenue_yen" in df.columns:
            revenue = pd.to_numeric(df["revenue_yen"], errors="coerce")
            revenue_mean = df.assign(revenue_yen=revenue).groupby("ab_group", observed=True)["revenue_yen"].mean()
            revenue_mean = revenue_mean.dropna().sort_index()
            if not revenue_mean.empty:
                metrics.append(("Mean Revenue by A/B", revenue_mean, "mean(revenue_yen)"))

        if "returned" in df.columns:
            returned = pd.to_numeric(df["returned"], errors="coerce")
            returned_rate = (
                df.assign(returned=returned)
                .groupby("ab_group", observed=True)["returned"]
                .mean()
                .dropna()
                .sort_index()
                * 100
            )
            if not returned_rate.empty:
                metrics.append(("Return Rate by A/B", returned_rate, "returned rate (%)"))

        if metrics:
            fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4.8))
            if len(metrics) == 1:
                axes = [axes]

            for ax, (title, vals, ylabel) in zip(axes, metrics):
                ax.bar(vals.index.astype(str), vals.to_numpy(), color="#76b7b2")
                ax.set_title(title)
                ax.set_xlabel("ab_group")
                ax.set_ylabel(ylabel)

            save_plot(fig, out_dir / "orders_ab_comparison.png", dpi)
            created += 1
        else:
            print("[warn] orders A/B plot skipped: no metric columns available")
    else:
        print("[warn] orders A/B plot skipped: ab_group missing")

    return created


def main() -> int:
    args = parse_args()

    if args.dpi <= 0:
        print("[error] --dpi must be positive.")
        return 1

    try:
        data_dir = resolve_data_dir(args.data_dir)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"[error] {exc}")
        return 1

    out_dir = resolve_out_dir(args.out_dir)

    print(f"[info] using data_dir: {data_dir}")
    print(f"[info] using out_dir: {out_dir}")
    print(f"[info] dpi: {args.dpi}")

    loaded = load_csvs(data_dir)

    for file_name, columns in FILE_SPECS.items():
        if file_name in loaded:
            print_missing_summary(file_name, loaded[file_name], columns)

    created = 0

    people_df = loaded.get("people_stats.csv")
    if people_df is not None:
        created += plot_people(people_df, out_dir, args.dpi)

    products_df = loaded.get("products.csv")
    if products_df is not None:
        created += plot_products(products_df, out_dir, args.dpi)

    orders_df = loaded.get("orders.csv")
    if orders_df is not None:
        created += plot_orders(orders_df, out_dir, args.dpi)

    if created == 0:
        print("[error] no plots were created.")
        return 1

    print(f"[done] generated {created} plot(s) into: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

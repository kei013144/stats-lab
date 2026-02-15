#!/usr/bin/env python
"""Generate synthetic retail-style customers/products/orders CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def build_customers(n_customers: int, rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    age = np.clip(np.rint(rng.normal(41, 12, size=n_customers)).astype(int), 18, 82)
    region = rng.choice(
        ["north", "east", "west", "south", "central"],
        size=n_customers,
        p=[0.17, 0.25, 0.22, 0.17, 0.19],
    )

    # Right-skewed income distribution.
    income_true = rng.lognormal(mean=np.log(460.0), sigma=0.60, size=n_customers)
    income_log = np.log(income_true)
    income_z = (income_log - income_log.mean()) / (income_log.std() + 1e-9)

    price_sens = np.clip(0.63 - 0.12 * income_z + rng.normal(0, 0.15, size=n_customers), 0, 1)
    quality_pref = np.clip(0.44 + 0.12 * income_z + rng.normal(0, 0.15, size=n_customers), 0, 1)
    digital_pref = np.clip(0.55 - 0.007 * (age - 40) + rng.normal(0, 0.20, size=n_customers), 0, 1)

    # MCAR + MNAR-like income missingness (high income is slightly more likely to be missing).
    income_missing_prob = np.full(n_customers, 0.010, dtype=float)
    top_decile = np.quantile(income_true, 0.90)
    income_missing_prob += np.where(income_true >= top_decile, 0.040, 0.0)
    income_missing = rng.random(n_customers) < income_missing_prob

    income_observed = income_true.copy()
    income_observed[income_missing] = np.nan

    customers = pd.DataFrame(
        {
            "customer_id": [f"C{i:06d}" for i in range(1, n_customers + 1)],
            "age": age,
            "region": region,
            "income_man": np.round(income_observed, 1),
            "price_sens": np.round(price_sens, 3),
            "quality_pref": np.round(quality_pref, 3),
            "digital_pref": np.round(digital_pref, 3),
        }
    )
    return customers, income_true


def build_products(n_products: int, rng: np.random.Generator) -> pd.DataFrame:
    category_specs = {
        "electronics": {"share": 0.16, "median_price": 42000, "sigma": 0.70, "quality_mean": 4.0},
        "fashion": {"share": 0.20, "median_price": 7800, "sigma": 0.55, "quality_mean": 3.5},
        "grocery": {"share": 0.18, "median_price": 1200, "sigma": 0.50, "quality_mean": 3.2},
        "home": {"share": 0.14, "median_price": 9500, "sigma": 0.62, "quality_mean": 3.7},
        "beauty": {"share": 0.12, "median_price": 3200, "sigma": 0.52, "quality_mean": 3.6},
        "sports": {"share": 0.10, "median_price": 11000, "sigma": 0.60, "quality_mean": 3.8},
        "books": {"share": 0.10, "median_price": 1800, "sigma": 0.45, "quality_mean": 3.4},
    }
    categories = list(category_specs.keys())
    probs = np.array([category_specs[c]["share"] for c in categories], dtype=float)
    probs /= probs.sum()

    category = rng.choice(categories, size=n_products, p=probs)

    price_yen = np.empty(n_products, dtype=float)
    quality_score = np.empty(n_products, dtype=float)
    for cat in categories:
        idx = np.where(category == cat)[0]
        if len(idx) == 0:
            continue
        spec = category_specs[cat]
        price_yen[idx] = rng.lognormal(
            mean=np.log(spec["median_price"]), sigma=spec["sigma"], size=len(idx)
        )
        quality_score[idx] = np.clip(
            rng.normal(loc=spec["quality_mean"], scale=0.45, size=len(idx)), 1.0, 5.0
        )

    products = pd.DataFrame(
        {
            "product_id": [f"P{i:06d}" for i in range(1, n_products + 1)],
            "category": category,
            "price_yen": np.rint(price_yen).astype(int),
            "quality_score": np.round(quality_score, 2),
        }
    )
    return products


def build_orders(
    n_events: int,
    rng: np.random.Generator,
    customers: pd.DataFrame,
    income_true: np.ndarray,
    products: pd.DataFrame,
) -> pd.DataFrame:
    n_customers = len(customers)

    customer_idx = rng.integers(0, n_customers, size=n_events)
    customer_ids = customers["customer_id"].to_numpy()[customer_idx]
    channel = rng.choice(["web", "app", "store"], size=n_events, p=[0.42, 0.38, 0.20])
    ab_group = rng.choice(["A", "B"], size=n_events, p=[0.50, 0.50])

    start = np.datetime64("2025-01-01")
    order_date = pd.to_datetime(start + rng.integers(0, 365, size=n_events).astype("timedelta64[D]"))

    cust_price_sens = customers["price_sens"].to_numpy()[customer_idx]
    cust_quality_pref = customers["quality_pref"].to_numpy()[customer_idx]
    cust_digital_pref = customers["digital_pref"].to_numpy()[customer_idx]

    income_log = np.log(income_true)
    income_z = (income_log - income_log.mean()) / (income_log.std() + 1e-9)
    cust_income_z = income_z[customer_idx]

    channel_effect = np.where(channel == "app", 0.08, np.where(channel == "store", -0.06, 0.0))
    ab_effect = np.where(ab_group == "B", 0.12, 0.0)

    purchase_logit = (
        -1.55
        + 1.10 * cust_quality_pref
        - 0.90 * cust_price_sens
        + 0.20 * cust_digital_pref
        + 0.10 * cust_income_z
        + channel_effect
        + ab_effect
    )
    purchase_prob = np.clip(sigmoid(purchase_logit), 0.01, 0.95)
    purchased = rng.random(n_events) < purchase_prob

    product_category = products["category"].to_numpy()
    premium_pool = np.where(np.isin(product_category, ["electronics", "home", "sports"]))[0]
    budget_pool = np.where(~np.isin(product_category, ["electronics", "home", "sports"]))[0]

    premium_prob = np.clip(0.20 + 0.60 * cust_quality_pref - 0.45 * cust_price_sens, 0.05, 0.90)
    pick_premium = rng.random(n_events) < premium_prob
    product_idx = np.empty(n_events, dtype=int)
    product_idx[pick_premium] = rng.choice(premium_pool, size=pick_premium.sum(), replace=True)
    product_idx[~pick_premium] = rng.choice(budget_pool, size=(~pick_premium).sum(), replace=True)
    product_idx[~purchased] = -1

    purchased_idx = np.where(purchased)[0]

    quantity = np.zeros(n_events, dtype=int)
    if len(purchased_idx) > 0:
        lam = (
            1.00
            + 1.00 * cust_price_sens[purchased_idx]
            + 0.20 * (channel[purchased_idx] == "store").astype(float)
            + 0.15 * (ab_group[purchased_idx] == "B").astype(float)
        )
        quantity[purchased_idx] = 1 + rng.poisson(np.clip(lam, 0.2, 8.0), size=len(purchased_idx))
        quantity[purchased_idx] = np.clip(quantity[purchased_idx], 1, 20)

    outlier_idx = np.array([], dtype=int)
    if len(purchased_idx) > 0:
        n_outliers = max(1, int(len(purchased_idx) * 0.002))
        outlier_idx = rng.choice(purchased_idx, size=n_outliers, replace=False)
        quantity[outlier_idx] *= rng.integers(15, 40, size=n_outliers)

    discount = np.zeros(n_events, dtype=float)
    discount_base = rng.beta(2.0, 12.0, size=n_events) * 0.45
    if len(purchased_idx) > 0:
        discount[purchased_idx] = np.clip(
            discount_base[purchased_idx]
            + 0.10 * cust_price_sens[purchased_idx]
            + np.where(ab_group[purchased_idx] == "B", -0.005, 0.0),
            0.0,
            0.70,
        )

    product_prices = products["price_yen"].to_numpy()
    product_quality = products["quality_score"].to_numpy()

    unit_price = np.zeros(n_events, dtype=float)
    item_quality = np.zeros(n_events, dtype=float)
    if len(purchased_idx) > 0:
        unit_price[purchased_idx] = product_prices[product_idx[purchased_idx]]
        item_quality[purchased_idx] = product_quality[product_idx[purchased_idx]]

    noise = rng.lognormal(mean=0.0, sigma=0.12, size=n_events)
    ab_revenue_multiplier = np.where(ab_group == "B", 1.08, 1.00)

    revenue = np.zeros(n_events, dtype=float)
    if len(purchased_idx) > 0:
        revenue[purchased_idx] = (
            unit_price[purchased_idx]
            * quantity[purchased_idx]
            * (1.0 - discount[purchased_idx])
            * noise[purchased_idx]
            * ab_revenue_multiplier[purchased_idx]
        )

    if len(outlier_idx) > 0:
        revenue[outlier_idx] *= rng.uniform(1.3, 2.8, size=len(outlier_idx))

    returned = np.zeros(n_events, dtype=int)
    return_logit = -2.80 + 3.60 * discount + 0.15 * (channel == "web") + 0.25 * (channel == "app")
    return_prob = sigmoid(return_logit)
    return_flag = rng.random(n_events) < return_prob
    returned[purchased_idx] = return_flag[purchased_idx].astype(int)

    rating = np.full(n_events, np.nan, dtype=float)
    rating_raw = (
        2.90
        + 0.45 * item_quality
        + 0.35 * cust_quality_pref
        - 0.45 * cust_price_sens
        - 1.10 * returned
        + rng.normal(0, 0.60, size=n_events)
    )
    rating_values = np.clip(rating_raw, 1.0, 5.0)
    rating[purchased_idx] = rating_values[purchased_idx]
    rating_missing = (rng.random(n_events) < 0.12) & purchased
    rating[rating_missing] = np.nan

    product_ids = np.full(n_events, "", dtype=object)
    if len(purchased_idx) > 0:
        product_ids[purchased_idx] = products["product_id"].to_numpy()[product_idx[purchased_idx]]

    orders = pd.DataFrame(
        {
            "order_id": [f"O{i:07d}" for i in range(1, n_events + 1)],
            "order_date": order_date.strftime("%Y-%m-%d"),
            "customer_id": customer_ids,
            "product_id": product_ids,
            "channel": channel,
            "ab_group": ab_group,
            "purchased": purchased.astype(int),
            "quantity": quantity,
            "discount": np.round(discount, 4),
            "revenue_yen": np.rint(revenue).astype(int),
            "returned": returned,
            "rating": np.round(rating, 2),
        }
    )
    return orders


def print_sanity(customers: pd.DataFrame, products: pd.DataFrame, orders: pd.DataFrame) -> None:
    purchased_orders = orders[orders["purchased"] == 1]

    print(f"[retail] customers rows: {len(customers)}")
    print(f"[retail] products rows: {len(products)}")
    print(f"[retail] orders rows: {len(orders)}")
    print(f"[retail] customers income_man missing rate: {customers['income_man'].isna().mean():.4%}")

    purchase_rate = orders.groupby("ab_group", observed=True)["purchased"].mean()
    revenue_per_session = orders.groupby("ab_group", observed=True)["revenue_yen"].mean()
    revenue_per_purchase = purchased_orders.groupby("ab_group", observed=True)["revenue_yen"].mean()

    print("[retail] purchase rate by ab_group:")
    print((purchase_rate * 100).round(2).to_string())

    print("[retail] mean revenue per session by ab_group (yen):")
    print(revenue_per_session.round(1).to_string())

    print("[retail] mean revenue per purchased order by ab_group (yen):")
    print(revenue_per_purchase.round(1).to_string())

    if "A" in purchase_rate.index and "B" in purchase_rate.index:
        conv_lift = (purchase_rate["B"] / purchase_rate["A"] - 1.0) * 100
        rps_lift = (revenue_per_session["B"] / revenue_per_session["A"] - 1.0) * 100
        print(f"[retail] estimated conversion lift B vs A: {conv_lift:.2f}%")
        print(f"[retail] estimated revenue/session lift B vs A: {rps_lift:.2f}%")

    print(
        "[retail] rating missing rate among purchased orders: "
        f"{purchased_orders['rating'].isna().mean():.4%}"
    )
    print(
        "[retail] revenue summary (yen): "
        f"median={orders['revenue_yen'].median():.0f}, "
        f"p95={orders['revenue_yen'].quantile(0.95):.0f}, "
        f"max={orders['revenue_yen'].max():.0f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic retail CSV datasets.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data/synthetic"),
        help="Output directory",
    )
    parser.add_argument("--n_customers", type=int, default=3000, help="Number of customers")
    parser.add_argument("--n_products", type=int, default=800, help="Number of products")
    parser.add_argument("--n_sessions", type=int, default=40000, help="Number of session rows")
    parser.add_argument(
        "--n_orders",
        type=int,
        default=None,
        help="Alias of --n_sessions (if set, this value is used for orders row count).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    n_events = args.n_orders if args.n_orders is not None else args.n_sessions
    if args.n_customers <= 0 or args.n_products <= 0 or n_events <= 0:
        raise ValueError("n_customers, n_products, and n_sessions/n_orders must be positive.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    customers, income_true = build_customers(args.n_customers, rng)
    products = build_products(args.n_products, rng)
    orders = build_orders(n_events, rng, customers, income_true, products)

    customers_path = out_dir / "customers.csv"
    products_path = out_dir / "products.csv"
    orders_path = out_dir / "orders.csv"

    customers.to_csv(customers_path, index=False)
    products.to_csv(products_path, index=False)
    orders.to_csv(orders_path, index=False)

    print(f"[retail] wrote: {customers_path}")
    print(f"[retail] wrote: {products_path}")
    print(f"[retail] wrote: {orders_path}")
    print_sanity(customers, products, orders)


if __name__ == "__main__":
    main()

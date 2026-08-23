import pandas as pd
import numpy as np
from econml.dml import LinearDML
from lightgbm import LGBMRegressor


def find_optimal_price():
    df = pd.read_csv("data/pricing_data.csv")

    X = df[
        [
            "competitor_price",
            "demand",
            "seasonality",
            "customer_segment",
        ]
    ]

    treatment = df["price"]
    outcome = df["sales"]

    model_y = LGBMRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        verbosity=-1,
    )

    model_t = LGBMRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        verbosity=-1,
    )

    dml = LinearDML(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=False,
        cv=3,
        random_state=42,
    )

    dml.fit(Y=outcome, T=treatment, X=X)

    # Use average causal price effect
    causal_effect = float(dml.const_marginal_effect(X).mean())

    # Use average observed sales as baseline
    baseline_sales = float(outcome.mean())
    baseline_price = float(treatment.mean())

    prices = np.linspace(50, 150, 101)

    results = []

    for price in prices:
        expected_sales = (
            baseline_sales
            + causal_effect * (price - baseline_price)
        )

        revenue = price * expected_sales

        results.append({
            "price": price,
            "expected_sales": expected_sales,
            "expected_revenue": revenue,
        })

    results_df = pd.DataFrame(results)

    best = results_df.loc[
        results_df["expected_revenue"].idxmax()
    ]

    print("=== EconoCausal Dynamic Pricing ===")
    print(f"Causal price effect: {causal_effect:.4f}")
    print(f"Baseline price: {baseline_price:.2f}")
    print(f"Baseline sales: {baseline_sales:.2f}")
    print()
    print(f"Optimal price: {best['price']:.2f}")
    print(f"Expected sales: {best['expected_sales']:.2f}")
    print(f"Expected revenue: {best['expected_revenue']:.2f}")

    results_df.to_csv(
        "data/price_optimization_results.csv",
        index=False,
    )

    return best


if __name__ == "__main__":
    find_optimal_price()
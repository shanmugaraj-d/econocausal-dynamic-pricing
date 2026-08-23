import numpy as np
import pandas as pd
from pathlib import Path


def generate_pricing_data(n=2000, seed=42):
    rng = np.random.default_rng(seed)

    price = rng.uniform(50, 150, n)
    competitor_price = rng.uniform(50, 150, n)
    demand = rng.uniform(100, 1000, n)
    seasonality = rng.uniform(0, 1, n)
    customer_segment = rng.integers(0, 3, n)

    # Demand response to price + confounders
    treatment = price

    sales = (
        500
        - 2.5 * price
        + 1.8 * competitor_price
        + 0.35 * demand
        + 120 * seasonality
        + 50 * customer_segment
        + rng.normal(0, 50, n)
    )

    data = pd.DataFrame({
        "price": treatment,
        "competitor_price": competitor_price,
        "demand": demand,
        "seasonality": seasonality,
        "customer_segment": customer_segment,
        "sales": sales,
    })

    return data


if __name__ == "__main__":
    output_path = Path("data/pricing_data.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_pricing_data()

    df.to_csv(output_path, index=False)

    print(f"Dataset created: {output_path}")
    print(f"Rows: {len(df)}")
    print(df.head())
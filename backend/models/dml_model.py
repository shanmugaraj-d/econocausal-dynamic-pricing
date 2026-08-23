import pandas as pd
from econml.dml import LinearDML
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split


def run_dml():
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

    dml.fit(
        Y=outcome,
        T=treatment,
        X=X,
    )

    effect = dml.const_marginal_effect(X)

    print("=== EconoCausal DML Results ===")
    print(f"Estimated average price effect: {effect.mean():.4f}")
    print(f"Minimum effect: {effect.min():.4f}")
    print(f"Maximum effect: {effect.max():.4f}")

    return dml


if __name__ == "__main__":
    run_dml()
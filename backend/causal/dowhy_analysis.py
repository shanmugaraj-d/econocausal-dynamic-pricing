import pandas as pd
from dowhy import CausalModel


def run_causal_analysis():
    df = pd.read_csv("data/pricing_data.csv")

    model = CausalModel(
        data=df,
        treatment="price",
        outcome="sales",
        common_causes=[
            "competitor_price",
            "demand",
            "seasonality",
            "customer_segment",
        ],
    )

    identified_estimand = model.identify_effect()

    print("=== DoWhy Causal Identification ===")
    print(identified_estimand)

    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression",
    )

    print("\n=== DoWhy Causal Estimate ===")
    print(f"Estimated causal effect: {estimate.value}")

    return model, identified_estimand, estimate


if __name__ == "__main__":
    run_causal_analysis()
import pandas as pd
from dowhy import CausalModel


# Load dataset
data = pd.read_csv("data/pricing_data.csv")

# Define causal model
model = CausalModel(
    data=data,
    treatment="price",
    outcome="sales",
    common_causes=[
        "competitor_price",
        "demand",
        "seasonality",
        "customer_segment",
    ],
)

# Identify causal effect
identified_estimand = model.identify_effect()

# Estimate causal effect
estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.linear_regression",
)

print("=== DoWhy Causal Audit ===")
print(f"Original causal estimate: {estimate.value:.4f}")

# Refutation 1: Random Common Cause
random_refutation = model.refute_estimate(
    identified_estimand,
    estimate,
    method_name="random_common_cause",
)

print("\n=== Random Common Cause Refutation ===")
print(random_refutation)

# Refutation 2: Placebo Treatment
placebo_refutation = model.refute_estimate(
    identified_estimand,
    estimate,
    method_name="placebo_treatment_refuter",
    placebo_type="permute",
)

print("\n=== Placebo Treatment Refutation ===")
print(placebo_refutation)
# EconoCausal – Dynamic Pricing using Double Machine Learning

## Overview

EconoCausal is a causal machine learning framework for dynamic pricing. The project estimates the causal effect of price on sales using Double Machine Learning (DML) and DoWhy, and uses the estimated causal effect as an input to price optimization.

## Objective

The main objective is to move beyond simple correlation-based pricing models by estimating the causal impact of price changes on sales.

### Key Goals

* Estimate the causal effect of price on sales.
* Control for observed confounding variables.
* Use machine learning for nuisance-function estimation.
* Identify causal relationships using DoWhy.
* Convert causal estimates into an initial dynamic pricing optimization framework.
* Evaluate the framework using real-world pricing data in the next phase.

## Methodology

```text
Pricing Data
     ↓
Feature Preparation
     ↓
Causal Identification (DoWhy)
     ↓
Double Machine Learning (EconML)
     ↓
Causal Price Effect
     ↓
Revenue-Based Price Optimization
     ↓
Optimal Pricing Decision
```

## Variables

| Role          | Variable           |
| ------------- | ------------------ |
| Treatment     | `price`            |
| Outcome       | `sales`            |
| Common Causes | `competitor_price` |
| Common Causes | `demand`           |
| Common Causes | `seasonality`      |
| Common Causes | `customer_segment` |

## Technologies

* Python 3.11
* EconML
* DoWhy
* LightGBM
* pandas
* NumPy
* scikit-learn
* SciPy
* Matplotlib
* seaborn
* Jupyter Notebook

## Current Implementation

### Double Machine Learning

EconML `LinearDML` is used to estimate the causal effect of price on sales. LightGBM models are used for nuisance-function estimation.

### DoWhy

DoWhy is used for causal identification through a backdoor adjustment using the observed common causes.

### Dynamic Pricing

The estimated causal price effect is incorporated into an initial revenue-based pricing optimization framework.

## Dataset

The current prototype uses a synthetic dataset containing **2,000 observations**.

The dataset contains:

* Price
* Competitor price
* Demand
* Seasonality
* Customer segment
* Sales

The synthetic dataset is used to validate the causal estimation pipeline because the underlying price effect is known.

A publicly available real-world pricing and demand dataset will be incorporated in the next development phase.

## Results

| Method                  | Estimated Price Effect |
| ----------------------- | ---------------------: |
| Synthetic Ground Truth  |                ≈ -2.50 |
| Double Machine Learning |                -2.4611 |
| DoWhy                   |                -2.4748 |

The DML and DoWhy estimates are close to the known synthetic causal effect, providing initial validation of the causal estimation pipeline.

### Initial Optimization Result

* Baseline price: **100.28**
* Baseline sales: **725.32**
* Estimated causal price effect: **-2.4611**
* Initial optimal price: **150.00**
* Expected sales: **602.96**
* Expected revenue: **90,443.63**

The current optimizer reaches the upper observed price boundary. This is treated as an initial prototype result rather than a production pricing recommendation.

## Project Structure

```text
econocausal-dynamic-pricing/
│
├── backend/
│   ├── causal/
│   │   └── dowhy_analysis.py
│   ├── data/
│   │   └── generate_data.py
│   ├── models/
│   │   └── dml_model.py
│   └── optimization/
│       └── price_optimizer.py
│
├── data/
│   ├── pricing_data.csv
│   └── price_optimization_results.csv
│
├── notebooks/
│   └── week1_week2_results.ipynb
│
├── tests/
├── docs/
├── .gitignore
└── README.md
```

## Week 1–2 Progress

### Week 1

* Project repository and development branch setup
* Python virtual environment setup
* ML and causal inference dependencies installed
* Project architecture created
* Synthetic pricing dataset generated
* Treatment, outcome and confounding variables defined

### Week 2

* Double Machine Learning implemented using EconML
* LightGBM nuisance models integrated
* DoWhy causal identification implemented
* Causal price effect estimated
* Initial dynamic price optimization implemented
* Results notebook created with analysis and visualizations
* Implementation pushed to the `shanmugaraj-dev` branch

## Future Work

1. Evaluate the framework using a publicly available real-world pricing and demand dataset.
2. Improve the demand-response model.
3. Estimate heterogeneous treatment effects across customer and market conditions.
4. Add realistic pricing constraints.
5. Improve dynamic optimization under changing market conditions.
6. Compare causal pricing against non-causal baseline methods.

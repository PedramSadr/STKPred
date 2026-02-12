import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def check_quantile_calibration(results_df, quantile=0.90):
    """
    results_df columns: ['actual_price', 'q90_pred']
    quantile: The target coverage (e.g., 0.90)
    """
    # 1. Calculate the "Hit" (Did reality fall inside the bound?)
    # For q90, we want actual < prediction
    results_df['is_covered'] = results_df['actual_price'] < results_df['q90_pred']

    # 2. Calculate Empirical Coverage
    empirical_coverage = results_df['is_covered'].mean()

    print(f"--- Calibration Report for q{int(quantile * 100)} ---")
    print(f"Target Coverage:    {quantile:.1%}")
    print(f"Realized Coverage:  {empirical_coverage:.1%}")

    # 3. Diagnose
    diff = empirical_coverage - quantile
    if abs(diff) < 0.02:
        print("VERDICT: ✅ WELL CALIBRATED")
    elif diff > 0:
        print("VERDICT: ⚠️ UNDER-CONFIDENT (Bounds are too wide/safe)")
    else:
        print("VERDICT: 🚨 OVER-CONFIDENT (Bounds are too tight/risky)")

    return empirical_coverage


# Example Usage with Walk-Forward Data
# Assume you collected these lists during your Walk-Forward Loop
data = {
    'actual_price': [100, 102, 98, 105, 110, 108, 99, 101, 103, 97],
    'q90_pred': [104, 104, 104, 106, 108, 112, 105, 105, 105, 102]
    # Notice: At index 4 (110 vs 108), the actual broke the ceiling.
}
df = pd.DataFrame(data)

coverage = check_quantile_calibration(df, quantile=0.90)
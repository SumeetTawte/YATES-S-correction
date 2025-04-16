import numpy as np
import scipy.stats as stats
import pandas as pd

def chi_square_demo():
    print("Enter values for 2×2 contingency table:")
    a = int(input("Enter value for cell [0][0]: "))
    b = int(input("Enter value for cell [0][1]: "))
    c = int(input("Enter value for cell [1][0]: "))
    d = int(input("Enter value for cell [1][1]: "))

    observed = np.array([[a, b],
                         [c, d]])
    
    observed_totals = np.append(observed, np.sum(observed, axis=0).reshape(1, -1), axis=0)
    observed_totals = np.append(observed_totals, np.sum(observed_totals, axis=1).reshape(-1, 1), axis=1)
    
    print("\n🔹 1. Observed Table:")
    df_observed = pd.DataFrame(observed_totals, columns=["Outcome A", "Outcome B", "Total"], index=["Group 1", "Group 2", "Total"])
    print(df_observed)

    chi2, p, dof, expected = stats.chi2_contingency(observed, correction=False)
    
    expected_totals = np.append(expected, np.sum(expected, axis=0).reshape(1, -1), axis=0)
    expected_totals = np.append(expected_totals, np.sum(expected_totals, axis=1).reshape(-1, 1), axis=1)
    
    print("\n🔹 2. Expected Frequencies Table:")
    df_expected = pd.DataFrame(np.round(expected_totals, 4), columns=["Outcome A", "Outcome B", "Total"], index=["Group 1", "Group 2", "Total"])
    print(df_expected)

    o_flat = observed.flatten()
    e_flat = expected.flatten()
    diff_sq = (o_flat - e_flat) ** 2
    diff_sq_e = diff_sq / e_flat

    print("\n🔹 3. Step-by-Step Calculation Table (No Correction):")
    calc_table = pd.DataFrame({
        'O': o_flat,
        'E': np.round(e_flat, 4),
        '(O - E)^2': np.round(diff_sq, 4),
        '(O - E)^2 / E': np.round(diff_sq_e, 4)
    })
    print(calc_table)

    yates_numerator = (np.abs(o_flat - e_flat) - 0.5) ** 2
    yates_components = yates_numerator / e_flat

    print("\n🔹 4. Yates' Correction Calculation Table:")
    yates_table = pd.DataFrame({
        'O': o_flat,
        'E': np.round(e_flat, 4),
        '|O - E| - 0.5': np.round(np.abs(o_flat - e_flat) - 0.5, 4),
        '((|O - E| - 0.5)^2) / E': np.round(yates_components, 4)
    })
    print(yates_table)

    chi2_yates, _, _, _ = stats.chi2_contingency(observed, correction=True)
    critical_value = stats.chi2.ppf(0.95, df=1)

    print("\n🔹 5. Final Chi-Square Results and Conclusion:")
    print(f"Chi-Square WITHOUT Yates' correction: {chi2:.4f}")
    print(f"Chi-Square WITH Yates' correction:    {chi2_yates:.4f}")
    print(f"Critical Chi-Square value at 0.05 significance (df=1): {critical_value:.4f}")

    print("\n📌 Conclusion:")
    if chi2_yates > critical_value:
        print("✔️ With Yates' correction: Reject the null hypothesis (Significant)")
    else:
        print("❌ With Yates' correction: Fail to reject the null hypothesis (Not significant)")

    if chi2 > critical_value:
        print("✔️ Without Yates' correction: Reject the null hypothesis (Significant)")
    else:
        print("❌ Without Yates' correction: Fail to reject the null hypothesis (Not significant)")

chi_square_demo()

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
import numpy as np

username = "put in username here"

# Load data
file_path = f'/Users/{username}/Desktop/Project/3 Data/Merged-Cleaned Data/New_Merged_Cleaned_Data_M.xlsx'
sheet_name = 'Sheet1'
start_date = '2002-01-01'
end_date = '2024-06-01'

# Define dependent variables
dependent_vars = {
    'UK': ['UK_Num_Trans', 'UK_Val_Trans'],
    'DE': ['DE_Num_Trans', 'DE_Val_Trans']
}

# Load data from Excel
data = pd.read_excel(file_path, sheet_name=sheet_name)

# Filter data for the required period
data['Period'] = pd.to_datetime(data['Period'])
filtered_data = data[(data['Period'] >= start_date) & (data['Period'] <= end_date)]

# Extract relevant columns dynamically for UK and DE including lagged variables
uk_columns = ['Period'] + dependent_vars['UK']
de_columns = ['Period'] + dependent_vars['DE']

uk_df = filtered_data[uk_columns].copy()
de_df = filtered_data[de_columns].copy()

# Resample to monthly data and drop the 'Period' column
uk_df.set_index('Period', inplace=True)
de_df.set_index('Period', inplace=True)


# Define function to plot ACF and PACF in a single figure for all variables and output values as text
def plot_acf_pacf_side_by_side(df_uk, df_de, uk_vars, de_vars):
    total_vars = len(uk_vars) + len(de_vars)
    fig, axs = plt.subplots(total_vars, 2, figsize=(14, 3 * total_vars))

    acf_pacf_values = []

    # Function to calculate confidence intervals
    def conf_interval(acf_values, alpha=0.05):
        n = len(acf_values)
        z = 1.96  # 95% confidence interval
        ci = z / np.sqrt(n)
        return np.array([(-ci, ci)] * len(acf_values))

    # Plots UK variables
    for i, var in enumerate(uk_vars):
        plot_acf(df_uk[var], ax=axs[i, 0], lags=15, title=f"ACF of {var}")
        plot_pacf(df_uk[var], ax=axs[i, 1], lags=15, title=f"PACF of {var}")
        axs[i, 0].set_ylim(-1, 1.2)
        axs[i, 0].set_xlim(-0.5, 15.5)
        axs[i, 0].set_xticks(range(0, 16))
        axs[i, 1].set_ylim(-1, 1.2)
        axs[i, 1].set_xlim(-0.5, 15.5)
        axs[i, 1].set_xticks(range(0, 16))

        # Calculates ACF and PACF values
        acf_values, acf_confint = acf(df_uk[var], nlags=15, alpha=0.05)
        pacf_values, pacf_confint = pacf(df_uk[var], nlags=15, alpha=0.05)

        acf_pacf_values.append((var, acf_values, acf_confint, pacf_values, pacf_confint))

    # Plots DE variables
    for i, var in enumerate(de_vars):
        plot_acf(df_de[var], ax=axs[i + len(uk_vars), 0], lags=15, title=f"ACF of {var}")
        plot_pacf(df_de[var], ax=axs[i + len(uk_vars), 1], lags=15, title=f"PACF of {var}")
        axs[i + len(uk_vars), 0].set_ylim(-1, 1.2)
        axs[i + len(uk_vars), 0].set_xlim(-0.5, 15.5)
        axs[i + len(uk_vars), 0].set_xticks(range(0, 16))
        axs[i + len(uk_vars), 1].set_ylim(-1, 1.2)
        axs[i + len(uk_vars), 1].set_xlim(-0.5, 15.5)
        axs[i + len(uk_vars), 1].set_xticks(range(0, 16))

        # Calculates ACF and PACF values
        acf_values, acf_confint = acf(df_de[var], nlags=15, alpha=0.05)
        pacf_values, pacf_confint = pacf(df_de[var], nlags=15, alpha=0.05)

        acf_pacf_values.append((var, acf_values, acf_confint, pacf_values, pacf_confint))

    plt.tight_layout()
    plt.show()

    # Outputs ACF and PACF values with confidence intervals
    for var, acf_vals, acf_conf, pacf_vals, pacf_conf in acf_pacf_values:
        print(f"\n{var} ACF values with 95% confidence intervals:")
        for val, (lower, upper) in zip(acf_vals, acf_conf):
            print(f"{val:.3f} ({lower:.3f}, {upper:.3f})")
        print(f"\n{var} PACF values with 95% confidence intervals:")
        for val, (lower, upper) in zip(pacf_vals, pacf_conf):
            print(f"{val:.3f} ({lower:.3f}, {upper:.3f})")


# Calls function to plot ACF and PACF
plot_acf_pacf_side_by_side(uk_df, de_df, dependent_vars['UK'], dependent_vars['DE'])

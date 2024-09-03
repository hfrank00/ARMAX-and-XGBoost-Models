import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX

username = "put in username here"

# Load data
file_path = f'/Users/{username}/Desktop/Project/3 Data/Merged-Cleaned Data/New_Merged_Cleaned_Data_M.xlsx'
df_final = pd.read_excel(file_path, sheet_name='Sheet1')
output_path = f'/Users/{username}/Desktop/Project/Code/Thesis/Results/ARMAX Models'

# Configuration
legend_location = "High"  # "Low" or "High"
selected_set = 'Set 33076' # Set used for ARMAX model from the pool below

# These are the optimised sets retrieved from the "ARMAX Randomiser.py" file
sets_data = {
    'Set 37009': {
        'variables': ['DE-Prime_lag3', 'DE-cabgdp2_lag3', 'DE-wpir'],
        'model_var': 'DE_Num_Trans',
        'order': (3, 0, 4)
    },
    'Set 39938': {
        'variables': ['DE-HV60d', 'DE-IV91d', 'DE-unemp2', 'DE-wpi3', 'DE-wpi3_lag6'],
        'model_var': 'DE_Val_Trans',
        'order': (4, 0, 2)
    },
    'Set 12726': {
        'variables': ['UK-IV91d', 'UK-txcr'],
        'model_var': 'UK_Num_Trans',
        'order': (4, 0, 1)
    },
    'Set 33076': {
        'variables': ['UK-CCI_lag6', 'UK-Prime_lag3'],
        'model_var': 'UK_Val_Trans',
        'order': (4, 0, 2)
    }
}

# Selects desired set
best_set_name = selected_set
best_set_variables = sets_data[selected_set]['variables']
model_var = sets_data[selected_set]['model_var']
arimax_order = sets_data[selected_set]['order']

# Ensure the 'Period' column is correctly parsed as datetime
df_final['Period'] = pd.to_datetime(df_final['Period'], format='%Y-%m')
df_final.set_index('Period', inplace=True)
# Resample only numeric columns to quarterly frequency and calculate the mean
df_final = df_final.select_dtypes(include=[np.number]).resample('Q').mean()
# Logarithm
df_final = np.log(df_final)
# First differencing
f_final = df_final.diff().iloc[1:]

# Y-axis labels dictionary for graphs
y_axis_labels = {
    'DE_Num_Trans': "Quarterly Number of Transactions in Germany",
    'UK_Num_Trans': "Quarterly Number of Transactions in the UK",
    'DE_Val_Trans': "Quarterly Volume (€M) of Transactions in Germany",
    'UK_Val_Trans': "Quarterly Volume (€M) of Transactions in the UK"
}


# Run the regression and plot the results
def plot_actual_vs_predicted(set_name, variables, model_var, order):
    df_set = df_final[variables + [model_var]].dropna()
    X_set = df_set[variables]
    y_set = df_set[model_var]
    periods = df_set.index

    # Split data into training and testing sets (80/20 split) respecting time series continuity
    train_size = int(len(df_set) * 0.8)
    X_train_set, X_test_set = X_set.iloc[:train_size], X_set.iloc[train_size:]
    y_train_set, y_test_set = y_set.iloc[:train_size], y_set.iloc[train_size:]
    periods_train, periods_test = periods[:train_size], periods[train_size:]

    # Fit ARMAX model with the provided order with constant trend and no differencing
    model_set = SARIMAX(y_train_set, exog=X_train_set, order=order, trend='c')
    result_set = model_set.fit(disp=False)
    y_train_pred_set = result_set.predict(start=0, end=len(y_train_set) - 1, exog=X_train_set)
    y_test_pred_set = result_set.predict(start=len(y_train_set), end=len(y_set) - 1, exog=X_test_set)

    # Define the end date for the training period
    end_date = periods_train[-1]

    # Plot actual vs predicted for both training and testing data
    plt.figure(figsize=(14, 6))
    plt.plot(periods_train, y_train_set, color='blue', label='Actual Train Data', linestyle='-', marker='o')
    plt.plot(periods_train, y_train_pred_set, color='red', label='Predicted Train Data', linestyle='--', marker='x')
    plt.plot(periods_test, y_test_set, color='green', label='Actual Test Data', linestyle='-', marker='o')
    plt.plot(periods_test, y_test_pred_set, color='orange', label='Predicted Test Data', linestyle='--', marker='x')
    plt.xlim([periods_train[0], periods_test[-1]])
    plt.axvspan(periods_train[0], periods_test[0], color='navy', alpha=0.1, label='Train Period')
    plt.axvspan(periods_test[0], periods_test[-1], color='green', alpha=0.1, label='Test Period')

    # Create a custom legend for the best set variables
    if legend_location == "High":
        legend_y = 0.98
        text_y_start = 0.87
    else:  # "Low"
        legend_y = 0.26
        text_y_start = 0.15
    plt.text(0.008, legend_y, 'Independent \nVariables:', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             fontweight='bold', fontstyle='italic', zorder=2)
    for i, var in enumerate(best_set_variables):
        plt.text(0.008, text_y_start - i * 0.05, var, transform=plt.gca().transAxes, fontsize=10, fontstyle='italic', zorder=2)

    # Add text annotation for ARIMAX order
    plt.text(0.845, 0.03 - len(best_set_variables) * 0, f'ARIMAX Order: {order}', transform=plt.gca().transAxes,
             fontsize=10, fontstyle='italic')

    plt.title(f'ARMAX Model ({set_name}): Actual vs Predicted for {model_var}', fontsize=14)
    plt.xlabel(('Date'), fontdict={'style': 'italic'})
    plt.ylabel(y_axis_labels[model_var], fontdict={'style': 'italic'})
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_path}/{set_name}_Prediction Plot_{model_var}.svg", format='svg', bbox_inches='tight')
    plt.show()

    # Calculate and print evaluation metrics
    mae_train = mean_absolute_error(y_train_set, y_train_pred_set)
    mae_test = mean_absolute_error(y_test_set, y_test_pred_set)
    mse_train = mean_squared_error(y_train_set, y_train_pred_set)
    mse_test = mean_squared_error(y_test_set, y_test_pred_set)
    rmse_train = np.sqrt(mse_train)
    rmse_test = np.sqrt(mse_test)
    r2_train = r2_score(y_train_set, y_train_pred_set)
    r2_test = r2_score(y_test_set, y_test_pred_set)
    print(f"MAE (train): {mae_train:.4f}")
    print(f"MAE (test): {mae_test:.4f}")
    print(f"RMSE (train): {rmse_train:.4f}")
    print(f"RMSE (test): {rmse_test:.4f}")
    print(f"R² (train): {r2_train:.4f}")
    print(f"R² (test): {r2_test:.4f}")

    # Save the summary and metrics to a text file
    with open(f"{output_path}/{set_name}_Regression_Summary_{model_var}.txt", 'w') as f:
        f.write(result_set.summary().as_text())
        f.write(
            f"\n\nEvaluation Metrics:\n"
            f"MAE (train): {mae_train:.4f}\n"
            f"MAE (test): {mae_test:.4f}\n"
            f"RMSE (train): {rmse_train:.4f}\n"
            f"RMSE (test): {rmse_test:.4f}\n"
            f"R² (train): {r2_train:.4f}\n"
            f"R² (test): {r2_test:.4f}\n"
        )

# Plotting the best set
plot_actual_vs_predicted(best_set_name, best_set_variables, model_var, arimax_order)

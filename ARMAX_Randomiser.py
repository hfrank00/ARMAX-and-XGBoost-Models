import pandas as pd
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import Parallel, delayed
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

username = "put in username here"

# Load data
file_path = f'/Users/{username}/Desktop/Project/3 Data/Merged-Cleaned Data/New_Merged_Cleaned_Data_M.xlsx'
df_final = pd.read_excel(file_path, sheet_name='Sheet1')
model_var = 'UK_Val_Trans'
region = 'UK'
Iterations = 50000 # Amount of iterations

# Ensure the 'Period' column is correctly parsed as datetime
df_final['Period'] = pd.to_datetime(df_final['Period'], format='%Y-%m')
df_final.set_index('Period', inplace=True)
# Resample only numeric columns to quarterly frequency and calculate the mean
df_final = df_final.select_dtypes(include=[np.number]).resample('Q').mean()
# Logarithm
df_final = np.log(df_final)
# First differencing
f_final = df_final.diff().iloc[1:]


# Generate variable pool for UK and Germany
def generate_variable_pool(region):
    base_variables = [
        'IV91d', 'Prime', 'gdpr2', 'BCI', 'CCI', 'HV60d',
        'IV182d', 'cabgdp2', 'cpi3', 'gdpn2', 'unemp2',
        'wpi3', 'wpir', 'txcr'
    ]

    lag_suffixes = ['lag3', 'lag6']
    lag_variables = [f'{var}_{suffix}' for var in base_variables if var != 'txcr' for suffix in lag_suffixes]

    variable_pool = [f'{region}-{var}' for var in base_variables + lag_variables]

    return variable_pool
variable_pool = generate_variable_pool(region)


# Adjusted R-squared calculation
def adjusted_r2(r2, n, k):
    return 1 - (1 - r2) * ((n - 1) / (n - k - 1))

# Defines randomiser range for ARIMAX order parameters (5,0,5) options
p_range = range(5)  # AR order
d_range = range(1)  # Differencing
q_range = range(5)  # MA order

# Track unique sets of variables and orders
unique_sets_and_orders = set()


# Evaluate random sets using parallel processing
def evaluate_random_sets(num_sets):
    random_sets = {}
    while len(random_sets) < num_sets:
        # Generate a unique set of variables
        set_vars = tuple(sorted(random.sample(variable_pool, random.randint(1, 5))))

        # Randomly select a model order
        order = (random.choice(p_range), random.choice(d_range), random.choice(q_range))

        # Ensure unique combination of variables and order
        if (set_vars, order) not in unique_sets_and_orders:
            unique_sets_and_orders.add((set_vars, order))
            random_sets[f'Set {len(random_sets) + 1}'] = (list(set_vars), order)

    results = Parallel(n_jobs=-1)(
        delayed(evaluate_set)(set_name, variables, order)
        for set_name, (variables, order) in random_sets.items()
    )

    # Filter out None results
    filtered_results = [result for result in results if result is not None]

    # Create the DataFrame and round the values
    results_df = pd.DataFrame(filtered_results).round(3)

    return results_df


# Adjusted function to include model order in evaluation
def evaluate_set(set_name, variables, order):
    df_set = df_final[variables + [model_var]].dropna()

    X_set = df_set[variables]
    y_set = df_set[model_var]
    periods = df_set.index

    # Split data into training and testing sets (80/20 split)
    train_size = int(len(df_set) * 0.8)
    X_train_set, X_test_set = X_set.iloc[:train_size], X_set.iloc[train_size:]
    y_train_set, y_test_set = y_set.iloc[:train_size], y_set.iloc[train_size:]
    periods_train, periods_test = periods[:train_size], periods[train_size:]

    try:
        # Suppress warnings during model fitting
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_set = SARIMAX(y_train_set, exog=X_train_set, order=order, trend='n')
            result_set = model_set.fit(disp=False)

        y_train_pred_set = result_set.predict(start=0, end=len(y_train_set) - 1, exog=X_train_set)
        y_test_pred_set = result_set.predict(start=len(y_train_set), end=len(y_set) - 1, exog=X_test_set)

        mae_train = mean_absolute_error(y_train_set, y_train_pred_set)
        mae_test = mean_absolute_error(y_test_set, y_test_pred_set)

        mse_train = mean_squared_error(y_train_set, y_train_pred_set)
        mse_test = mean_squared_error(y_test_set, y_test_pred_set)

        rmse_train = np.sqrt(mse_train)
        rmse_test = np.sqrt(mse_test)

        r2_train = r2_score(y_train_set, y_train_pred_set)
        r2_test = r2_score(y_test_set, y_test_pred_set)
        r2_adj_set = adjusted_r2(r2_test, len(y_test_set), X_test_set.shape[1])
        aic = result_set.aic

        # Print RMSE values
        print(f"{set_name} - {order}: RMSE (Train) = {rmse_train:.3f}, RMSE (Test) = {rmse_test:.3f}")

        return {
            'Set': set_name,
            'ARIMAX Order': order,
            'Variables': variables,
            'MAE (Train)': mae_train,
            'MAE (Test)': mae_test,
            'RMSE (Train)': rmse_train,
            'RMSE (Test)': rmse_test,
            'R2 (Train)': r2_train,
            'R2 (Test)': r2_test,
            'Adjusted R2 (Test)': r2_adj_set,
            'AIC': aic
        }
    except Exception as e:
        print(f"Skipping {set_name} due to error: {e}")
        return None


# Evaluate random sets
results_df = evaluate_random_sets(Iterations)

# Filter results based on a few conditions to minimise random order possibilities
filtered_results_df = results_df[
    (results_df['RMSE (Train)'] < results_df['RMSE (Test)']) &
    (abs(results_df['R2 (Test)'] - results_df['R2 (Train)']) <= 0.4) &
    (results_df['R2 (Test)'] > 0.25)
]

# Save the filtered results to an Excel file for further investigation
filtered_results_df.to_excel(f'/Users/{username}/Desktop/Project/Code/Thesis/Results/ARMAX Sets/{model_var} XX ARMAX Sets.xlsx', index=False)

# Also displays the sorted filtered results with variables and order for each set
sorted_filtered_results_df = filtered_results_df.sort_values(by='RMSE (Train)', ascending=True)
print(sorted_filtered_results_df[['ARIMAX Order', 'RMSE (Train)', 'RMSE (Test)']])

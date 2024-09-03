import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_squared_error

username = "put in username here"

ivol_de_path = f"/Users/{username}/Desktop/Project/3 Data/Macro-Factors/others/Implied Volatility/Raw/MSCI GERMANY Index.csv"
ivol_uk_path = f"/Users/{username}/Desktop/Project/3 Data/Macro-Factors/Implied Volatility/Raw/MSCI United Kingdom Index.csv"
histvol_de_path = f"/Users/{username}/Desktop/Project/3 Data/Macro-Factors/Historic Volatility/Hist Vol UK.csv"
histvol_uk_path = f"/Users/{username}/Desktop/Project/3 Data/Macro-Factors/Historic Volatility/Hist Vol UK.csv"
macro_path = f"/Users/{username}/Desktop/Project/3 Data/Merged-Cleaned Data/New_Merged_Cleaned_Data_M.xlsx"
MA_data_path = f"/Users/{username}/Desktop/Project/3 Data/M&A Data/DE_UK_Trans.xlsx"
output_path = f"/Users/{username}/Desktop/Project/Code/Thesis/Results/XGBoost"

use_all_features = False


# Y-axis labels dictionary for graphs
y_axis_labels = {
    'DE_Num_Trans': "Quarterly Number of Transactions in Germany",
    'UK_Num_Trans': "Quarterly Number of Transactions in the UK",
    'DE_Val_Trans': "Quarterly Volume (€M) of Transactions in Germany",
    'UK_Val_Trans': "Quarterly Volume (€M) of Transactions in the UK"
}

# Function to filter and aggregate M&A data
# Number of transaction by count and value of transaction by aggregation
def compute_target_vars(target_data):
    target_data_de = target_data[ target_data['Country/Region']=="Germany" ]
    target_data_uk = target_data[target_data['Country/Region']=="United Kingdom"]

    threshold_de = target_data_de['Transaction Value (€M)'].quantile(0.99)
    target_data_de[target_data_de['Transaction Value (€M)']>=threshold_de] = np.nan

    threshold_uk = target_data_uk['Transaction Value (€M)'].quantile(0.99)
    target_data_de[target_data_de['Transaction Value (€M)']>=threshold_uk] = np.nan

    target_de = (target_data_de.groupby('Completion Date').agg({'Country/Region': 'count',
                                                               'Transaction Value (€M)': 'sum'})
                                                          .rename(columns={'Country/Region':'Num_Trans',
                                                                           'Transaction Value (€M)':'Val_Trans'}))
    target_de.columns = [f'DE_{col}' for col in target_de.columns]
    target_uk = (target_data_uk.groupby('Completion Date').agg({'Country/Region': 'count',
                                                               'Transaction Value (€M)': 'sum'})
                                                          .rename(columns={'Country/Region':'Num_Trans',
                                                                           'Transaction Value (€M)':'Val_Trans'}))
    target_uk.columns = [f'UK_{col}' for col in target_uk.columns]
    target_df = target_de.join(target_uk, how='outer')
    target_df = target_df.fillna(0)
    return target_df


def get_merged_data():
    # Read macro data
    macro_df = pd.read_excel(macro_path, 'Sheet1')
    cols = macro_df.columns
    macro_df = macro_df.drop( columns = [col for col in cols if ("HV" in col or "IV" in col or "Period_" in col) ] )
    macro_df = macro_df.drop( columns= ['DE_Num_Trans', 'DE_Val_Trans', 'UK_Num_Trans', 'UK_Val_Trans'] )

    # Read raw data
    ivol_de = pd.read_csv(ivol_de_path, parse_dates=['date'], usecols=['date', 'days', 'impl_volatility', 'cp_flag'])
    ivol_uk = pd.read_csv(ivol_uk_path, parse_dates=['date'], usecols=['date', 'days', 'impl_volatility', 'cp_flag'])
    histvol_de = pd.read_csv(histvol_de_path, parse_dates=['Date'], usecols=['Date', 'Volatility'])
    histvol_uk = pd.read_csv(histvol_uk_path, parse_dates=['Date'], usecols=['Date', 'Volatility'])
    target_data = pd.read_excel(MA_data_path, header=0)
    target_data['Completion Date'] = pd.to_datetime(target_data['Completion Date'], format='%d.%m.%Y')

    # Set index to date
    ivol_de = ivol_de.set_index('date', drop=True)
    ivol_uk = ivol_uk.set_index('date', drop=True)
    histvol_de = histvol_de.set_index('Date', drop=True)
    histvol_uk = histvol_uk.set_index('Date', drop=True)

    # Drop duplicate values for historic volatility dataframes
    histvol_de = histvol_de.drop_duplicates()
    histvol_uk = histvol_uk.drop_duplicates()

    # Compute target data dataframes
    target_df = compute_target_vars(target_data)

    # Only keep call implied volatilities
    ivol_uk = ivol_uk[ ivol_uk['cp_flag']=='C' ]
    ivol_de = ivol_de[ ivol_de['cp_flag']=='C' ]

    # Seperate in days to maturity (91 and 182)
    ivol_de_91 = pd.DataFrame(ivol_de[ivol_de['days'] == 91]['impl_volatility'])
    ivol_de_182 = pd.DataFrame(ivol_de[ivol_de['days'] == 182]['impl_volatility'])
    ivol_uk_91 = pd.DataFrame(ivol_uk[ivol_uk['days'] == 91]['impl_volatility'])
    ivol_uk_182 = pd.DataFrame(ivol_uk[ivol_uk['days'] == 182]['impl_volatility'])

    # Rename columns
    ivol_de_91.columns, ivol_de_182.columns = ['ivol_de_91'], ['ivol_de_182']
    ivol_uk_91.columns, ivol_uk_182.columns = ['ivol_uk_91'], ['ivol_uk_182']
    histvol_de.columns, histvol_uk.columns = ['histvol_de'], ['histvol_uk']

    # Join data and only keep rows where data on all variables is available
    df = target_df.join( [histvol_de, histvol_uk, ivol_de_91, ivol_de_182, ivol_uk_91, ivol_uk_182], how='outer')
    df["Period"] = df.index.to_period('M').astype("str")
    df["date"] = df.index
    df = pd.merge( df, macro_df, on="Period", how="left" )
    df = df.set_index("date", drop=True)
    df = df.drop(columns=["Period"])

    # Sort and return
    return df.sort_index()


# Assumes target column is called 'target'
def get_variables(my_df):
    features = set(my_df.columns)
    features -= { 'target' }
    X, y = my_df[ list(features) ], my_df[ 'target' ]
    return X, y


# XGBoost model
def xgboost_train_regression(dtrain, params, num_boost_round=500, verbose_eval=1, dval=None):
    evals = [(dtrain, "train")]

    # Add validation dataset to the evaluations
    if dval:
        evals.append((dval, "validation"))

    # Train the XGBoost model
    model = xgb.train(
       params=params,
       dtrain=dtrain,
       num_boost_round=num_boost_round,
       evals=evals,
       verbose_eval=verbose_eval,
    )
    return model


# For double-checking that the columns are indeed floats
def to_float(x):
    if type(x) == str: return x
    try: return float(x)
    except: return np.nan


def prepare_data(target_name, country):
    df = get_merged_data()

    # Only keep data where target exists
    df = df.dropna(subset=[target_name], axis='rows')
    df = df.astype(float)

    # Decide on target
    df['target'] = df[target_name]

    # Only keep relevant columns
    assert country in {'DE', 'UK'}
    if country=='DE':
        cols_to_use = {col for col in df.columns if 'de' in col}
        if use_all_features == True:
            # All DE macro variables contain "DE" in their names
            cols_to_use2 = {col for col in df.columns if 'DE' in col}
            cols_to_use = cols_to_use.union(cols_to_use2)
        cols_to_use -= {"DE_Num_Trans", "DE_Val_Trans"}

        df = df[ [*list(cols_to_use), *['target']] ]
        cols_vol = ["histvol_de", "ivol_de_182", "ivol_de_91"]
    else:
        cols_to_use = {col for col in df.columns if 'uk' in col}
        if use_all_features == True:
            # All UK macro variables contain "UK" in their names
            cols_to_use2 = {col for col in df.columns if 'UK' in col}
            cols_to_use = cols_to_use.union(cols_to_use2)
        cols_to_use -= {"UK_Num_Trans", "UK_Val_Trans"}

        df = df[[*list(cols_to_use), *['target']]]
        cols_vol = ["histvol_uk", "ivol_uk_182", "ivol_uk_91"]

    # Computes a series that indicates whether all volatility measures exist
    df["all_vols_avail"] = np.prod(df[cols_vol].fillna(0), axis=1) > 0
    # Computes a series that indicates whether at least one volatility measures exist
    df["one_vols_avail"] = np.sum(df[cols_vol].fillna(0), axis=1) > 0

    # Computes first and last day to consider
    first_day = df[ df["all_vols_avail"] ].index[0]
    last_day = df[ df["all_vols_avail"] ].index[-1]

    # Restricts df to this time frame
    df = df[ (df.index >= first_day) & (df.index <= last_day) ]

    # Carry forward transactions without volatility data to the next date with available volatility
    temp_sum = 0
    my_dates = df.index
    for crt_date in reversed(my_dates):
        row = df.loc[crt_date, :]
        if not row.one_vols_avail:
            temp_sum += row.target
        else:
            df.loc[crt_date, "target"] += temp_sum
            temp_sum = 0

    df = df[ df["one_vols_avail"] ]
    df = df[[*cols_to_use, *['target']]]

    # Split TRAIN - VALIDATION - TEST
    # Perform train-val-test split
    n = len(df)
    test_idx = int(0.8*n)
    val_idx = int(0.8*0.8*n)

    df_test = df.iloc[test_idx:]
    df_val = df.iloc[val_idx:test_idx]
    df_train = df.iloc[:val_idx]

    X_train, y_train = get_variables(df_train)
    X_val, y_val = get_variables(df_val)
    X_test, y_test = get_variables(df_test)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def train_xgboost_with_hyperparameters(train, val, test,
                                       max_depth, rate_drop,
                                       n_boost_rounds=500, verbose_eval=1,
                                       plot=False):

    dtrain, X_train, y_train = train
    dval, X_val, y_val = val
    dtest, X_test, y_test = test

    # Set XGBoost parameters for training using the DART booster and histogram tree method
    # Define the objective function as squared error for regression
    params = {"tree_method": "hist",
              'booster': 'dart',
              'max_depth': max_depth,
              "rate_drop": rate_drop,
              "objective": "reg:squarederror"}

    # Train the XGBoost model with 500 iterations
    model = xgboost_train_regression(dtrain=dtrain,
                                     params=params,
                                     num_boost_round=n_boost_rounds,
                                     dval=dval,
                                     verbose_eval=verbose_eval)

    # Make prediction on test, train and validation sets
    preds_train = model.predict(dtrain)
    preds_val = model.predict(dval)
    preds_test = model.predict(dtest)

    # Create a DataFrame for plotting by combining dates, true values, and predictions
    plot_df = pd.DataFrame({'date': np.concatenate([X_train.index, X_val.index, X_test.index]),
                            'truth': np.concatenate([y_train, y_val, y_test]),
                            'pred': np.concatenate([preds_train, preds_val, preds_test])
                            })
    plot_df = plot_df.set_index('date')

    # Compute dates for train, validation, and test sets
    train_date = X_train.index[0]
    val_date = X_val.index[0]
    test_date = X_test.index[0]
    end_date = X_test.index[-1]

    # Resample and plot the data to match ARMAX models
    plot_df = plot_df.resample('M').sum()
    plot_df = plot_df.resample('Q').mean()

    # Filter the DataFrame to include only the data up to end_date
    filtered_plot_df = plot_df[plot_df.index <= end_date]

    if plot:
        # Plot and save quarterly prediction graph
        plt.figure(figsize=(14, 6))
        plt.plot(filtered_plot_df.index, filtered_plot_df['truth'], color='blue', linestyle='-', marker='o',
                 label='Actual Train Data')
        plt.plot(filtered_plot_df.index, filtered_plot_df['pred'], color='red', linestyle='--', marker='x',
                 label='Predicted Train Data')
        plt.plot(plot_df[test_date:end_date].index, plot_df.loc[test_date:end_date, 'truth'],
                 color='green', linestyle='-', marker='o', label='Actual Test Data')
        plt.plot(plot_df[test_date:end_date].index, plot_df.loc[test_date:end_date, 'pred'],
                 color='orange', linestyle='--', marker='x', label='Predicted Test Data')
        plt.axvspan(train_date, val_date, color='navy', alpha=0.1, label='Train Period')
        plt.axvspan(val_date, test_date, color='yellow', alpha=0.1, label='Validation Period')
        plt.axvspan(test_date, end_date, color='green', alpha=0.1, label='Test Period')
        plt.xlim(train_date, end_date)
        plt.title(f'XGBoost Model: Actual vs Predicted for {country}_{target_name}', fontsize=14)
        plt.xlabel('Date', fontdict={'style': 'italic'})
        plt.ylabel(y_axis_labels[f'{country}_{target_name}'], fontdict={'style': 'italic'})
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_path}/Prediction Plot - {country}_{target_name}.svg", format='svg', bbox_inches='tight')
        plt.show()

        # Plot and save daily prediction graph
        plt.figure(figsize=(14, 6))
        plt.plot(X_train.index, y_train, label='Train (Actual Data)')
        plt.plot(X_train.index, preds_train, label='Train (Predicted Data)')
        plt.plot(X_val.index, y_val, label='Validation (Actual Data)')
        plt.plot(X_val.index, preds_val, label='Validation (Predicted Data)')
        plt.plot(X_test.index, y_test, label='Test (Actual Data)')
        plt.plot(X_test.index, preds_test, label='Test (Predicted Data)')
        plt.axvspan(train_date, val_date, color='navy', alpha=0.1, label='Train Period')
        plt.axvspan(val_date, test_date, color='yellow', alpha=0.1, label='Validation Period')
        plt.axvspan(test_date, end_date, color='green', alpha=0.1, label='Test Period')
        plt.legend()
        plt.grid()
        plt.savefig(f"{output_path}/Daily Prediction Plot - {country}_{target_name}.svg", format='svg', bbox_inches='tight')
        plt.show()

    # Prepare to return errors
    errors = {}

    # Calculate error loss for resampled data to compare to ARMAX models
    # Train set
    train_resampled = plot_df[ plot_df.index <= val_date ]
    train_rmse_resampled = np.sqrt( mean_squared_error(train_resampled.truth, train_resampled.pred) )
    print(f'Resampled RMSE Train: {train_rmse_resampled}')
    errors["RMSE Train"] = train_rmse_resampled

    # Validation set
    val_resampled = plot_df[ (plot_df.index > val_date) & (plot_df.index <= test_date) ]
    val_rmse_resampled = np.sqrt( mean_squared_error(val_resampled.truth, val_resampled.pred) )
    print(f'Resampled RMSE Validation: {val_rmse_resampled}')
    errors["RMSE Validation"] = val_rmse_resampled

    # Test set
    test_resampled = plot_df[ plot_df.index >= test_date ]
    test_rmse_resampled = np.sqrt( mean_squared_error(test_resampled.truth, test_resampled.pred) )
    print(f'Resampled RMSE Test: {test_rmse_resampled}')
    errors["RMSE Test"] = test_rmse_resampled

    return errors

    # To use for optimal configuration after Gridsearch
if __name__ == '__main__':
    # Configuration
    target_name = 'Val_Trans' # either 'Val_Trans' or 'Num_Trans'
    country = 'DE'          # either 'DE' or 'UK'

    # optimal hyperparameters (obtained via grid search)
    optimal_depth = 4
    optimal_drop = 0.3

    train_data, val_data, test_data = prepare_data(target_name=f"{country}_{target_name}", country=country)

    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    my_dtrain = xgb.DMatrix(X_train, y_train, enable_categorical=False)
    my_dval = xgb.DMatrix(X_val, y_val, enable_categorical=False)
    my_dtest = xgb.DMatrix(X_test, y_test, enable_categorical=False)

    my_train = (my_dtrain, X_train, y_train)
    my_val = (my_dval, X_val, y_val)
    my_test = (my_dtest, X_test, y_test)

    errors = train_xgboost_with_hyperparameters(train=my_train,
                                                val=my_val,
                                                test=my_test,
                                                max_depth=optimal_depth,
                                                rate_drop=optimal_drop,
                                                plot=False)

    print(errors)

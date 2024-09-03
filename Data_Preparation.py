import pandas as pd
import numpy as np

# Data Configuration
aggregation_period = 'M'  # 'M' for months, 'Q' for quarters, 'Y' for years
start_period = '1996-01-01'  # Data availability from 1996 onwards

username = "put in username here"

# Paths for input and output files
paths = {
    "MA_data_path": f"/Users/{username}/Desktop/Project/3 Data/M&A Data/DE_UK_Trans.xlsx",
    "macro_path": f'/Users/{username}/Desktop/Project/3 Data/Macro-Factors/Macro_Data.xlsx',
    "histvol_uk_path": f'/Users/{username}/Desktop/Project/3 Data/Macro-Factors/Historic Volatility/Hist Vol UK.csv',
    "histvol_de_path": f'/Users/{username}/Desktop/Project/3 Data/Macro-Factors/Historic Volatility/Hist Vol DE.csv',
    "prime_rates_path": f'/Users/{username}/Desktop/Project/3 Data/Macro-Factors/Prime Rates UK DE.xlsx',
    "sentiment_path": f'/Users/{username}/Desktop/Project/3 Data/Macro-Factors/Sentiments/Sentiment_Indicators.csv',
    "ivol_uk_path": f'/Users/{username}/Desktop/Project/3 Data/Macro-Factors/Implied Volatility/Raw/MSCI United Kingdom Index.csv',
    "ivol_de_path": f'/Users/{username}/Desktop/Project/3 Data/Macro-Factors/Implied Volatility/Raw/MSCI GERMANY Index.csv',

    # File used by other Python scripts
    "output_path": "/Users/username/Desktop/Project/3 Data/Merged-Cleaned Data/New_Merged_Cleaned_Data_M.xlsx"
}


# Function to load prime rates data
def load_prime_rates_data(file_path, skip_rows, columns):
    df = pd.read_excel(file_path, skiprows=skip_rows)
    df.columns = columns
    df['Date'] = pd.to_datetime(df['Date']) + pd.offsets.MonthEnd(0)
    return df


# Function to process volatility data
def process_volatility_data(file_path):
    vol_data = pd.read_csv(file_path)
    vol_data['Date'] = pd.to_datetime(vol_data['Date'])
    return vol_data


# Function to aggregate volatility data
def aggregate_volatility(vol_data, unique_months):
    monthly_volatility = []
    for month in unique_months:
        start_date, end_date = month.start_time, month.end_time
        monthly_data = vol_data[(vol_data['Date'] >= start_date) & (vol_data['Date'] <= end_date)]
        avg_volatility = monthly_data['Volatility'].mean()
        monthly_volatility.append({'Date': month.end_time, 'Volatility': avg_volatility})
    return pd.DataFrame(monthly_volatility)


# Function to filter and aggregate M&A data
def aggregate_MA_data_path(MA_data_path, country, transaction_col, date_col):
    
    # Outlier function
    MA_data_path_country = MA_data_path[ MA_data_path['Country/Region'] == country ].copy()
    threshold = MA_data_path_country[transaction_col].quantile(0.99)
    MA_data_path_country[ MA_data_path_country[transaction_col]>=threshold ] = np.nan
    filtered_data = MA_data_path_country

    # Number of transaction by count and value of transaction by aggregation
    return filtered_data.groupby('Period').agg({
        transaction_col: 'sum',
        date_col: 'count'
    }).reset_index().rename(columns={
        transaction_col: f'{country}_Trans_Val',
        date_col: f'{country}_Num_Trans'
    })


# Function to load and process implied volatility data
def load_and_process_implied_volatility(germany_path, uk_path):
    germany_data = pd.read_csv(germany_path)
    uk_data = pd.read_csv(uk_path)

    germany_data['date'] = pd.to_datetime(germany_data['date'], format='%Y-%m-%d')
    uk_data['date'] = pd.to_datetime(uk_data['date'], format='%Y-%m-%d')

    germany_data = germany_data[germany_data['ticker'] == 'EWG'].dropna()
    uk_data = uk_data[uk_data['ticker'] == 'EWU'].dropna()

    selected_days = [91, 182]
    selected_deltas = [50, -50]

    germany_filtered = germany_data[(germany_data['days'].isin(selected_days)) & (germany_data['delta'].isin(selected_deltas))]
    uk_filtered = uk_data[(uk_data['days'].isin(selected_days)) & (uk_data['delta'].isin(selected_deltas))]

    start_date = max(germany_filtered['date'].min(), uk_filtered['date'].min())
    end_date = min(germany_filtered['date'].max(), uk_filtered['date'].max())

    germany_filtered = germany_filtered[(germany_filtered['date'] >= start_date) & (germany_filtered['date'] <= end_date)]
    uk_filtered = uk_filtered[(uk_filtered['date'] >= start_date) & (uk_filtered['date'] <= end_date)]

    germany_pivot = germany_filtered.pivot_table(index='date', columns=['days', 'delta'], values='impl_volatility')
    uk_pivot = uk_filtered.pivot_table(index='date', columns=['days', 'delta'], values='impl_volatility')

    germany_avg = germany_pivot.T.groupby(level=0).mean().T
    uk_avg = uk_pivot.T.groupby(level=0).mean().T

    germany_avg.columns = [f'DE-IV_{col}d' for col in germany_avg.columns.get_level_values(0)]
    uk_avg.columns = [f'UK-IV_{col}d' for col in uk_avg.columns.get_level_values(0)]

    germany_avg = germany_avg[['DE-IV91d', 'DE-IV182d']]
    uk_avg = uk_avg[['UK-IV91d', 'UK-IV182d']]

    germany_monthly = germany_avg.resample(aggregation_period).mean()
    uk_monthly = uk_avg.resample(aggregation_period).mean()

    combined_iv_data = pd.concat([germany_monthly, uk_monthly], axis=1)
    combined_iv_data.reset_index(inplace=True)
    return combined_iv_data


# Load prime rates data
prime_rates_data = load_prime_rates_data(paths["prime_rates_path"], skip_rows=1, columns=['Date', 'UK Prime Rate', 'Germany Prime Rate'])
prime_rates_data.rename(columns={
    'Date': 'datadate',
    'UK Prime Rate': 'UK-Prime',
    'Germany Prime Rate': 'DE-Prime',
}, inplace=True)

# Load sentiment data
sentiment_data = pd.read_csv(paths["sentiment_path"])
sentiment_data['DateTime'] = pd.to_datetime(sentiment_data['DateTime'], format='%Y-%m')

# Load and process main macro data
macro_data = pd.read_excel(paths["macro_path"])
macro_data = macro_data[macro_data['econiso'].isin(['DEU', 'GBR'])]
macro_data['datadate'] = pd.to_datetime(macro_data['datadate'], dayfirst=True)
macro_data = macro_data[macro_data['datadate'].dt.year >= 1991]
macro_data['econiso'] = macro_data['econiso'].replace({'DEU': 'DE', 'GBR': 'UK'})
columns_to_keep = ['datadate', 'gdpn2', 'gdpr2', 'unemp2', 'cpi3', 'cabgdp2', 'wpi3', 'wpir', 'txcr', 'econiso']
macro_data = macro_data[columns_to_keep]

# Pivot and transform macro data
pivoted_data = pd.DataFrame()
for col in columns_to_keep:
    if col != 'datadate' and col != 'econiso':
        df = macro_data.pivot(index='datadate', columns='econiso', values=col)
        df.columns = [f'{country}-{col}' for country in df.columns]
        pivoted_data = pd.concat([pivoted_data, df], axis=1)
pivoted_data.reset_index(inplace=True)
pivoted_data.columns = ['datadate'] + list(pivoted_data.columns[1:])
pivoted_data.dropna(axis=1, how='all', inplace=True)

# Process and aggregate historic volatility data
histvol_uk_path = process_volatility_data(paths["histvol_uk_path"])
histvol_de_path = process_volatility_data(paths["histvol_de_path"])
unique_months = pd.Series(pivoted_data['datadate'].dt.to_period(aggregation_period).unique())
histvol_uk_path_monthly = aggregate_volatility(histvol_uk_path, unique_months)
histvol_de_path_monthly = aggregate_volatility(histvol_de_path, unique_months)
histvol_uk_path_monthly.rename(columns={'Date': 'datadate', 'Volatility': 'UK-HV60d'}, inplace=True)
histvol_de_path_monthly.rename(columns={'Date': 'datadate', 'Volatility': 'DE-HV60d'}, inplace=True)

# Ensure datadate is in the correct format in all dataframes
pivoted_data['datadate'] = pd.to_datetime(pivoted_data['datadate'])
histvol_de_path_monthly['datadate'] = pd.to_datetime(histvol_de_path_monthly['datadate'])
prime_rates_data['datadate'] = pd.to_datetime(prime_rates_data['datadate'])

# Merge historic volatility data
min_date = pd.to_datetime(start_period)
pivoted_data = pivoted_data[pivoted_data['datadate'] >= min_date]
pivoted_data = pd.merge(pivoted_data, histvol_uk_path_monthly, on='datadate', how='outer')
pivoted_data = pd.merge(pivoted_data, histvol_de_path_monthly, on='datadate', how='outer')

# Merge prime rates data
pivoted_data = pd.merge(pivoted_data, prime_rates_data, on='datadate', how='outer')

# Merge sentiment data
sentiment_data.rename({'DateTime': 'datadate'}, inplace=True)
pivoted_data = pd.merge(pivoted_data, sentiment_data, on='datadate', how='outer')

# Remove duplicate columns
pivoted_data = pivoted_data.loc[:, ~pivoted_data.columns.duplicated()]

# Merge implied volatility data
iv_data = load_and_process_implied_volatility(paths["ivol_de_path"], paths["ivol_uk_path"])
pivoted_data.rename({'date': 'datadate'}, inplace=True)
pivoted_data = pd.merge_asof(pivoted_data.sort_values('datadate'), iv_data.sort_values('datadate'), on='datadate', direction='nearest', tolerance=pd.Timedelta(days=2))

# Forward-fill the few missing values in the implied volatility columns
iv_columns_correct = ['DE-IV91d', 'DE-IV182d', 'UK-IV91d', 'UK-IV182d']
pivoted_data[iv_columns_correct] = pivoted_data[iv_columns_correct].ffill()

# Load and process M&A data
MA_data_path = pd.read_excel(paths["MA_data_path"])
MA_data_path['Completion Date'] = pd.to_datetime(MA_data_path['Completion Date'], format='%d.%m.%Y')
pivoted_data['Period'] = pivoted_data['datadate'].dt.to_period(aggregation_period)
MA_data_path['Period'] = MA_data_path['Completion Date'].dt.to_period(aggregation_period)

# Aggregate M&A data
ma_uk_aggregated = aggregate_MA_data_path(MA_data_path, 'United Kingdom', 'Transaction Value (€M)', 'Completion Date')
ma_de_aggregated = aggregate_MA_data_path(MA_data_path, 'Germany', 'Transaction Value (€M)', 'Completion Date')

# Rename M&A data columns
ma_uk_aggregated.rename(columns={'United Kingdom_Trans_Val': 'UK_Val_Trans', 'United Kingdom_Num_Trans': 'UK_Num_Trans'}, inplace=True)
ma_de_aggregated.rename(columns={'Germany_Trans_Val': 'DE_Val_Trans', 'Germany_Num_Trans': 'DE_Num_Trans'}, inplace=True)

# Aggregate macro factors
macro_factors_aggregated = pivoted_data.groupby('Period').mean(numeric_only=True).reset_index()

# Create lagged variables for 3 months and 6 months
lags = [3, 6]
lagged_data = pd.concat([macro_factors_aggregated.shift(lag).add_suffix(f'_lag{lag}') for lag in lags], axis=1)

# Combine the original and lagged data
combined_data = pd.concat([macro_factors_aggregated, lagged_data], axis=1)

# Merge all data into a final dataset
merged_data = pd.merge(combined_data, ma_uk_aggregated, on='Period', how='outer')
merged_data = pd.merge(merged_data, ma_de_aggregated, on='Period', how='outer')

# Add Period to column ordering and organize by countries, excluding lagged columns
ordered_columns = ['Period'] + sorted([col for col in merged_data.columns if 'UK' in col and 'lag' not in col]) + sorted([col for col in merged_data.columns if 'DE' in col and 'lag' not in col])

# Ensure all columns are included, including lagged columns
final_columns = ordered_columns + sorted([col for col in merged_data.columns if 'lag' in col])

# Dropping duplicate columns from merging process
final_data = merged_data.loc[:, ~merged_data.columns.duplicated()]

# Reorder the columns
final_data = final_data[final_columns]

# Save the final cleaned and merged data
final_data.to_excel(paths["output_path"], index=False)
print(f"Data cleaned and saved to: {paths['output_path']}")

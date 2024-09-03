import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.dates as mdates

username = "put in username here"

# Path to the file
file_path = f'/Users/{username}/Desktop/Project/3 Data/Merged-Cleaned Data/New_Merged_Cleaned_Data_M.xlsx'
output_dir = f'/Users/{username}/Desktop/Project/3 Data/Data Exploration/'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load data
data = pd.read_excel(file_path, sheet_name='Sheet1')

# Ensure the 'Period' column is correctly parsed as datetime
data['Period'] = pd.to_datetime(data['Period'], format='%Y-%m')
data.set_index('Period', inplace=True)

# Drops lag columns as not necessary for data exploration
data_filtered = data.loc[:, ~data.columns.str.contains('Lag')]

# Select relevant columns for UK and Germany
uk_columns = [
    'UK-HV60d', 'UK-IV182d', 'UK-IV91d', 'UK-Prime', 'UK-cabgdp2',
    'UK-cpi3', 'UK-gdpn2', 'UK-gdpr2', 'UK-txcr', 'UK-unemp2', 'UK-wpi3',
    'UK-wpir', 'UK-CCI', 'UK-BCI', 'UK_Num_Trans', 'UK_Val_Trans'
]

de_columns = [
    'DE-HV60d', 'DE-IV182d', 'DE-IV91d', 'DE-Prime', 'DE-cabgdp2',
    'DE-cpi3', 'DE-gdpn2', 'DE-gdpr2', 'DE-txcr', 'DE-unemp2', 'DE-wpi3',
    'DE-wpir', 'DE-CCI', 'DE-BCI', 'DE_Num_Trans', 'DE_Val_Trans'
]


# Function to rename the columns by removing "UK_" or "DE_" and "UK-" or "DE-" prefixes for the graphs
def rename_column(column_name):
    if column_name.startswith('UK_') or column_name.startswith('DE_'):
        return column_name[3:]  # Remove the first three characters for underscore format
    if column_name.startswith('UK-') or column_name.startswith('DE-'):
        return column_name[3:]  # Remove the first three characters for dash format
    return column_name


uk_data_filtered = data_filtered[uk_columns]
de_data_filtered = data_filtered[de_columns]

# Calculate correlation matrices
uk_correlation_matrix = uk_data_filtered.corr()
de_correlation_matrix = de_data_filtered.corr()

# Plots and saves UK correlation heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(uk_correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=.5)
plt.title('UK Correlation Matrix Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'UK_Correlation_Matrix_Heatmap.svg'))
plt.show()

# Plots and saves DE correlation heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(de_correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=.5)
plt.title('DE Correlation Matrix Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'DE_Correlation_Matrix_Heatmap.svg'))
plt.show()

# Saves summary statistics
summary_stats = data_filtered.describe()
summary_stats.to_csv(os.path.join(output_dir, 'Summary_Statistics.csv'))

# Histograms for UK data
num_cols_uk = len(uk_data_filtered.columns)
fig, axes = plt.subplots(nrows=num_cols_uk // 3 + 1, ncols=3, figsize=(18, 6*num_cols_uk//3))
axes = axes.flatten()
for i, col in enumerate(uk_data_filtered.columns):
    sns.histplot(uk_data_filtered[col].dropna(), kde=True, ax=axes[i])
    axes[i].set_title(f'Histogram of {col}')
for ax in axes[num_cols_uk:]:
    ax.remove()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'UK_Histograms.svg'))
plt.show()

# Histograms for DE data
num_cols_de = len(de_data_filtered.columns)
fig, axes = plt.subplots(nrows=num_cols_de // 3 + 1, ncols=3, figsize=(18, 6*num_cols_de//3))
axes = axes.flatten()
for i, col in enumerate(de_data_filtered.columns):
    sns.histplot(de_data_filtered[col].dropna(), kde=True, ax=axes[i])
    axes[i].set_title(f'Histogram of {col}')
for ax in axes[num_cols_de:]:
    ax.remove()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'DE_Histograms.svg'))
plt.show()

# Define the start and end date for the x-axis based on the actual data range
start_date = uk_data_filtered.index.min()
end_date = uk_data_filtered.index.max()

# Plotting time series for UK and DE data (first half)
fig1, axes1 = plt.subplots(nrows=len(uk_columns[:len(uk_columns) // 2]), ncols=1,
                           figsize=(15, 3 * (len(uk_columns) // 2)), sharex=True)
for i, ax in enumerate(axes1):
    uk_column = uk_columns[i]
    de_column = de_columns[i]

    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())

    ax.plot(uk_data_filtered.index, uk_data_filtered[uk_column], label='UK', color='blue')
    ax.plot(de_data_filtered.index, de_data_filtered[de_column], label='DE', color='orange')

    # Rename the variable to remove country prefix
    variable_name = rename_column(uk_column)

    ax.set_title(f'Time Series of {variable_name}')
    ax.set_xlim([start_date, end_date])
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'UK_DE_Combined_Time_Series_Part1.svg'))
plt.show()

# Plotting time series for UK and DE data (second half)
fig2, axes2 = plt.subplots(nrows=len(uk_columns[len(uk_columns) // 2:]), ncols=1,
                           figsize=(15, 3 * (len(uk_columns) - len(uk_columns) // 2)), sharex=True)
for i, ax in enumerate(axes2):
    uk_column = uk_columns[len(uk_columns) // 2 + i]
    de_column = de_columns[len(uk_columns) // 2 + i]

    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())

    ax.plot(uk_data_filtered.index, uk_data_filtered[uk_column], label='UK', color='blue')
    ax.plot(de_data_filtered.index, de_data_filtered[de_column], label='DE', color='orange')

    # Rename the variable to remove country prefix
    variable_name = rename_column(uk_column)

    ax.set_title(f'Time Series of {variable_name}')
    ax.set_xlim([start_date, end_date])
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'UK_DE_Combined_Time_Series_Part2.svg'))
plt.show()
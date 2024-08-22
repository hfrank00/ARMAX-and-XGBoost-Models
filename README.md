# ARMAX-and-XGBoost-Models

This repository contains the Python scripts developed for the thesis, focusing on ARMAX and XGBoost models for forecasting M&A activity.

## Files Overview

1. **Data Exploration.py**  
   Provides initial analysis of the dataset, including plots and statistical summaries.

2. **Data Preparation.py**  
   Prepares data for modeling, including cleaning, merging, and transformation.

3. **Model Diagnostics.py**  
   Calculates and plots model diagnostics, such as ACF and PACF plots for time series analysis.

4. **ARMAX Randomiser.py**  
   Performs hyperparameter tuning for the ARMAX model configurations, returning unique sets of optimal models.

5. **ARMAX Model.py**  
   Implements and runs the ARMAX models, utilising the cleaned dataset from `Data Preparation.py` and the optimal sets from `ARMAX Randomiser.py`.

6. **XGBoost Validation.py**  
   Validates XGBoost models and tunes hyperparameters.

7. **XGBoost Model.py**  
   Utilises `XGBoost Validation.py` to train and evaluate the XGBoost models.

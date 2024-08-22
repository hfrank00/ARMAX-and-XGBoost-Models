# ARMAX-and-XGBoost-Models

This repository contains the Python scripts for the thesis, focusing on the ARMAX and XGBoost models for forecasting M&A activity.

**Files Overview**

_1. Data Exploration.py_
Provides initial analysis of the dataset, including plots and statistical summaries.

_2. Data Preparation.py_
Prepares data for modeling, including cleaning, merging, and transformation.

_3. Model Diagnostics.py_
Calculates and plots model diagnostics, such as ACF and PACF plots for time series analysis.

_4. ARMAX Randomiser.py_
Hyperparameter tuning for the ARMAX model configurations. Returns unique sets of optimal models.

_5. ARMAX Model.py_
Implements and runs the ARMAX models, utilising the cleaned dataset from Data Preparation.py file and the optimal sets from the ARMAX Randomiser.py file.

_6. XGBoost Validation.py_
Validates XGBoost models and tunes hyperparameters.

_7. XGBoost Model.py_
Utilises XGBoost Validation.py to train and evaluate the XGBoost models.

# ARMAX-and-XGBoost-Models

This repository contains the Python scripts developed for the thesis, focusing on ARMAX and XGBoost models for forecasting M&A activity.

## Files Overview

1. **Data_Exploration.py**  
   Provides initial analysis of the dataset, including plots and statistical summaries.

2. **Data_Preparation.py**  
   Prepares data for modelling, including cleaning, merging, and transformation.

3. **Model_Diagnostics.py**  
   Calculates and plots model diagnostics, such as ACF and PACF plots, for time series analysis.

4. **ARMAX_Randomiser.py**  
   Performs hyperparameter tuning for the ARMAX model configurations, returning unique sets of optimal models.

5. **ARMAX_Model.py**  
   Implements and runs the ARMAX models, utilising the cleaned dataset from `Data_Preparation.py` and the optimal sets from `ARMAX_Randomiser.py`.

6. **XGBoost_Gridsearch.py**  
   Finetunes hyperparameters using Gridsearch methodology and validates XGBoost models.

7. **XGBoost_Model.py**  
   Utilises `XGBoost_Gridsearch.py` to train and evaluate the XGBoost models.

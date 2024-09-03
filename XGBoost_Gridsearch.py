import time
import pandas as pd
import itertools
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
# Import the XGBoost modules
import XGBoost_Model


def perform_grid_search(country, target_name, depths, rates_drop, n_boost_rounds=500):
    start_gridsearch = time.time()

    param_list_training = [depths, rates_drop]

    depths_list = []
    rates_list = []
    rmse_train_list = []
    rmse_val_list = []

    train_data, val_data, test_data = XGBoost_Model.prepare_data(target_name=f"{country}_{target_name}",
                                                                 country=country)

    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    my_dval = xgb.DMatrix(X_val, y_val, enable_categorical=False)
    my_dtest = xgb.DMatrix(X_test, y_test, enable_categorical=False)
    my_dtrain = xgb.DMatrix(X_train, y_train, enable_categorical=False)

    my_train = (my_dtrain, X_train, y_train)
    my_val = (my_dval, X_val, y_val)
    my_test = (my_dtest, X_test, y_test)

    param_combinations_training = itertools.product(*param_list_training)
    for (depth, rate_drop) in param_combinations_training:

        print("##############################")
        print("------------------------------")
        print( f"[ depth = {depth} , rate_drop = {rate_drop} ] " )
        print("------------------------------")

        errors = XGBoost_Model.train_xgboost_with_hyperparameters(train=my_train,
                                                                  val=my_val,
                                                                  test=my_test,
                                                                  max_depth=depth,
                                                                  rate_drop=rate_drop,
                                                                  n_boost_rounds=n_boost_rounds,
                                                                  # We don't want to print any boost rounds
                                                                  # during grid search --> disable
                                                                  verbose_eval=0,
                                                                  plot=False)

        # Add MSEs to dataset
        depths_list.append(depth)
        rates_list.append(rate_drop)
        rmse_train_list.append( errors["RMSE Train"] )
        rmse_val_list.append( errors["RMSE Validation"] )


    results_df = pd.DataFrame({'depth': depths_list, 'rate': rates_list,
                               'rmse_train': rmse_train_list,
                               'rmse_val': rmse_val_list})
    results_df.to_excel(f'Gridsearch_result.xlsx')
    end_gridsearch = time.time()
    print(f'time for gridsearch: {end_gridsearch - start_gridsearch:.3f}')


if __name__ == '__main__':
    my_depths = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    my_rates_drop = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    target_name = 'Val_Trans' # Either 'Val_Trans' or 'Num_Trans'
    country = 'DE'          # Either 'DE' or 'UK'

    # Perform Gridsearch
    perform_grid_search(country, target_name, depths=my_depths, rates_drop=my_rates_drop, n_boost_rounds=500)

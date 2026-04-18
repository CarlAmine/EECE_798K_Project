# %%
from matplotlib import rcParams
from matplotlib import rc
import sklearn
import random
import numpy as np
#!define device
RANDOM_STATE = 2021
# !plot paramerters
rcParams['font.style'] = 'normal'
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'Arial'
rc('pdf', fonttype=42)
#!seed
sklearn.random.seed(2021)
# torch.manual_seed(2021)
random.seed(2021)
np.random.seed(2021)
RANDOM_STATE = 2021
BOOST_ROUND = 50000
EARLY_STOP = 500
DATA_PATH = './data/'
# %%
#!param setting for lgbm
PARAM_ALL_SENSOR = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'learning_rate': 0.008811081555269881,
    'max_depth': 20,
    'num_leaves': 2,
    'max_bin': 199,
    'feature_fraction': 0.22669272164500454,
    'bagging_fraction': 0.08774975143857956,
    'lambda_l1': 4.251920518275901,
    'lambda_l2': 35.37817764952771,
    # 'min_sum_hessian_in_leaf': 0.09690794881031235,
    # 'min_data_in_leaf': 351,
    'metric': 'rmse',
    # 'bagging_freq': 5,
    'seed': RANDOM_STATE,
    'num_threads': 80
}
PARAM_SELECTED_SENSOR = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'learning_rate': 0.06254860781028972,
    'max_depth': 41,
    'num_leaves': 323,
    'max_bin': 199,
    'min_data_in_leaf': 173,
    'feature_fraction': 0.06677188904447798,
    'bagging_fraction': 0.16256032275820181,
    'metric': 'rmse',
    'lambda_l1': 5.217287920571612,
    'lambda_l2': 1.5417980688095783,
    'min_sum_hessian_in_leaf': 0.003148238153083795,
    'seed': RANDOM_STATE,
    'num_threads': 80}
PARAM_SELECTED_STATISTIC = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'max_bin': 199,
    # 'bagging_fraction': 0.5403874395856009,
    # 'feature_fraction': 0.6254237693170765,
    'lambda_l1': 0.5619147100961293,
    'lambda_l2': 4.84317679103711,
    'learning_rate': 0.06224241882611581,
    'max_depth': 5,
    # 'min_data_in_leaf': 141,
    # 'min_gain_to_split': 0.2814928668986977,
    # 'min_sum_hessian_in_leaf': 0.010131407329740303,
    'num_leaves': 5,
    # # 'bagging_freq': 10,
    'metric': 'rmse',
    'seed': RANDOM_STATE,
    'num_threads': 80
}
PARAM_SELECTED_STATISTIC_8_10 = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'learning_rate': 0.04757086536248228,
    'max_depth': 5,
    'num_leaves': 256,
    'max_bin': 179,
    'min_data_in_leaf': 104,
    'feature_fraction': 0.11374386151228788,
    # 'bagging_fraction': 0.7249110918075498,
    'metric': 'rmse',
    # 'bagging_freq': 2,
    # 'lambda_l1': 1.2926450391605209,
    'lambda_l2': 4.39076135579039,
    # 'min_sum_hessian_in_leaf': 0.8042375917550351,
    'seed': RANDOM_STATE,
    'num_threads': 80
}
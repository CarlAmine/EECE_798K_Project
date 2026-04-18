# %%
import numpy as np
from submisstion_setting import *
import re
from mpl_toolkits import axisartist
from mpl_toolkits.axes_grid1 import host_subplot
import scipy.stats as st
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error, mean_squared_error
import random
from matplotlib.pyplot import MultipleLocator
from sklearn.preprocessing import StandardScaler
from matplotlib import rc
from matplotlib import rcParams
import shap
import copy
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import GridSearchCV
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
#! import pandas as pd
import seaborn as sns
import gc
import sklearn
from sklearn.model_selection import KFold
from sklearn import metrics
from bayes_opt import BayesianOptimization
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


def rmse(y_predict, y_true):
    sumrmse = 0
    y_predict = np.asarray(y_predict)
    y_true = np.asarray(y_true)
    for i in range(len(y_true)):
        sumrmse += (y_predict[i] - y_true[i])**2
    sumrmse = np.square(sumrmse / len(y_true))
    return sumrmse


def coef(x_test, y_test):
    n = len(y_test)
    r = 0
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    sx = np.std(x_test)
    sy = np.std(y_test)
    x_mean = x_test.mean()
    y_mean = y_test.mean()
    for i in range(n):
        r += (x_test[i] - x_mean) / sx * (y_test[i] - y_mean) / sy
    return 1/(n - 1)*r


def coefficient_determination(y_predict, y_true):
    y_predict = np.asarray(y_predict)
    y_true = np.asarray(y_true)
    R2 = 0
    numerator = 0
    denominator = 0
    y_true_mean = y_true.mean()
    for i in range(len(y_true)):
        numerator += (y_true[i] - y_predict[i])**2
        denominator += (y_true[i] - y_true_mean)**2
    R2 = 1 - (numerator / denominator)
    return R2


def bias_variance(preds, y):
    preds = np.array(preds)
    y = np.array(y)
    bias = np.square(preds.mean() - y)
    variance = np.square(preds - preds.mean())
    return bias.mean(), variance.mean()
# ?bayes_opt allsensors data


def lgb_cv(max_depth, num_leaves, min_data_in_leaf, feature_fraction, lambda_l1, lambda_l2, min_sum_hessian_in_leaf, learning_rate, bagging_fraction):
    flods = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(x_train_sd.shape[0])
    predictions = np.zeros(x_test_sd.shape[0])
    for one_fold, (train_idx, val_idx) in enumerate(flods.split(x_train_sd, y_train)):
        print('fold = {}'.format(one_fold))
        train_data = lgb.Dataset(
            x_train_sd[train_idx], label=y_train[train_idx], )
        val_data = lgb.Dataset(
            x_train_sd[val_idx], label=y_train[val_idx], )
        param = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'learning_rate': learning_rate,
            'max_depth': int(max_depth),
            'num_leaves': int(num_leaves),
            'min_sum_hessian_in_leaf': int(min_sum_hessian_in_leaf),
            # 'min_gain_to_split': min_gain_to_split,
            'max_bin': 199,
            'min_data_in_leaf': int(min_data_in_leaf),
            'feature_fraction': feature_fraction,
            'bagging_fraction': bagging_fraction,
            # 'bagging_freq': 20,
            'metric': 'rmse',
            'lambda_l1': lambda_l1,
            'lambda_l2': lambda_l2,
            'seed': RANDOM_STATE,
            'verbosity': -1,
            'num_threads': 80
        }
        reg = lgb.train(params=param, train_set=train_data, num_boost_round=5000, valid_sets=[
                        train_data, val_data], valid_names=['Training', 'Validation'], verbose_eval=False, early_stopping_rounds=200)
        oof[val_idx] = reg.predict(
            x_train_sd[val_idx], num_iteration=reg.best_iteration)
        predictions += reg.predict(x_test_sd,
                                   num_iteration=reg.best_iteration) / flods.n_splits
        del reg, train_idx, val_idx
        gc.collect()
    return metrics.r2_score(predictions, y_test)
# %%
#!function build the new features
Time_window = 300
# Time_window = 20  # min failure
# Time_window = 100  # max failure


def mean_windows(one_feature):
    one_feature = np.array(one_feature)
    feature_mean = []
    feature_mean = [one_feature[i] for i in range(int(Time_window/2))]
    for i in range(int(Time_window/2), int(len(one_feature)-int(Time_window/2))):
        feature_mean.append(
            np.mean(one_feature[i-int(Time_window/2):i+int(Time_window/2)]))
    for last in one_feature[-int(Time_window/2):]:
        feature_mean.append(last)
    return feature_mean


def variance_windows(one_feature):
    one_feature = np.array(one_feature)
    feature_variance = []
    feature_variance = [0 for i in range(int(Time_window/2))]
    for i in range(int(Time_window/2), int(len(one_feature))-int(Time_window/2)):
        feature_variance.append(
            np.var(one_feature[i-int(Time_window/2):i+int(Time_window/2)]))
    for last in one_feature[-int(Time_window/2):]:
        feature_variance.append(0)
    return feature_variance


def skewness_windows(one_feature):
    one_feature = np.array(one_feature)
    feature_skewness = []
    feature_skewness = [one_feature[i] for i in range(int(Time_window/2))]
    for i in range(int(Time_window/2), int(int(len(one_feature)-int(Time_window/2)))):
        skewness = 0
        data = pd.DataFrame(
            one_feature[i-int(Time_window/2):i+int(Time_window/2)])
        skewness = data.skew().values[0]
        # for x_j in one_feature[i-int(Time_window/2):i+int(Time_window/2)]:
        #     skewness += ((x_j - np.mean(one_feature[i-int(Time_window/2):i+int(
        #         Time_window/2)]))**3)/(np.std(one_feature[i-int(Time_window/2):i+int(Time_window/2)])**3)
        # skewness = (1/(Time_window-1))*skewness
        feature_skewness.append(skewness)
    for last in one_feature[-int(Time_window/2):]:
        feature_skewness.append(last)
    return feature_skewness


def kurtosis_windows(one_feature):
    one_feature = np.array(one_feature)
    feature_kurtosis = []
    feature_kurtosis = [0 for i in range(int(Time_window/2))]
    for i in range(int(Time_window/2), int(len(one_feature)-int(Time_window/2))):
        kurtosis = 0
        data = pd.DataFrame(
            one_feature[i-int(Time_window/2):i+int(Time_window/2)])
        kurtosis = data.kurtosis().values[0]
        # for x_j in one_feature[i-int(Time_window/2):i+int(Time_window/2)]:
        #     kurtosis += ((x_j - np.mean(one_feature[i-int(Time_window/2):i+int(Time_window/2)]))**4)/(
        #         np.std(one_feature[i-int(Time_window/2):i+int(Time_window/2)])**4)
        # kurtosis = (1/(Time_window-1))*kurtosis - 3
        feature_kurtosis.append(kurtosis)
    for last in one_feature[-int(Time_window/2):]:
        feature_kurtosis.append(0)
    return feature_kurtosis


def interquartile_windows(one_feature):
    one_feature = np.array(one_feature)
    feature_interquartile = []
    feature_interquartile = [one_feature[i] for i in range(int(Time_window/2))]
    for i in range(int(Time_window/2), int(len(one_feature)-int(Time_window/2))):
        interquartile = 0
        data = one_feature[i-int(Time_window/2):i+int(Time_window/2)]
        lower = np.quantile(data, 0.25, interpolation='lower')
        higher = np.quantile(data, 0.75, interpolation='higher')
        interquartile = higher-lower
        feature_interquartile.append(interquartile)
    for last in one_feature[-int(Time_window/2):]:
        feature_interquartile.append(last)
    return feature_interquartile


def percentile_windows(one_feature):
    one_feature = np.array(one_feature)
    feature_percentile_1 = []
    feature_percentile_91 = []
    feature_percentile_1 = [one_feature[i] for i in range(int(Time_window/2))]
    feature_percentile_91 = [one_feature[i] for i in range(int(Time_window/2))]
    for i in range(int(Time_window/2), int(len(one_feature)-int(Time_window/2))):
        percentile_1 = 0
        percentile_91 = 0
        data = one_feature[i-int(Time_window/2):i+int(Time_window/2)]
        percentile_1 = np.percentile(data, 1, interpolation='midpoint')
        percentile_91 = np.percentile(data, 91, interpolation='midpoint')
        feature_percentile_1.append(percentile_1)
        feature_percentile_91.append(percentile_91)
    for last in one_feature[-int(Time_window/2):]:
        feature_percentile_1.append(last)
        feature_percentile_91.append(last)
    return feature_percentile_1, feature_percentile_91


def median_windows(one_feature):
    one_feature = np.array(one_feature)
    feature_median = []
    feature_median = [one_feature[i] for i in range(int(Time_window/2))]
    for i in range(int(Time_window/2), int(len(one_feature))-int(Time_window/2)):
        median = 0
        data = one_feature[i-int(Time_window/2):i+int(Time_window/2)]
        median = np.median(data)
        feature_median.append(median)
    for last in one_feature[-int(Time_window/2):]:
        feature_median.append(last)
    return feature_median


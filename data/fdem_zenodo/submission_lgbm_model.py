# %%
# TODO 20221021 modified
from submission_function_define import *
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
from tqdm.autonotebook import tqdm
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
# %%
data_gather = np.fromfile(DATA_PATH+'p28_data.bin',
                          dtype=np.float64).reshape(25000, 8814)
all_features = data_gather[:, :-2]
all_features = pd.DataFrame(all_features)
t = data_gather[:, -2]
nss = data_gather[:, -1]
# %%
# TODO all sensor model
# ? preapre data in time split
Ground_truth = nss
train_all_features = all_features.iloc[:int(len(all_features) * 0.67)]
test_all_features = all_features.iloc[int(len(all_features) * 0.67):]
train_label = Ground_truth[:int(len(Ground_truth) * 0.67)]
test_label = Ground_truth[int(len(Ground_truth) * 0.67):]
train_features_names = list(all_features.columns)
test_features_names = list(all_features.columns)
# %%
# ! standard the data or not
x_train = train_all_features
y_train = train_label
x_test = test_all_features
y_test = test_label
t_train = t[:int(len(t) * 0.67)]
t_predct = t[int(len(t) * 0.67):]
standard = StandardScaler()
standard.fit(all_features)
# x_train_sd = standar.transform(x_train)
# x_test_sd = standar.transform(x_test)
x_train_sd = np.array(x_train)
x_test_sd = np.array(x_test)
y_train = y_train
y_test = y_test
# %%
#!all sensor prediction
# ? preapre data in time split
Ground_truth = nss
Friction_coef = nss
train_all_features = all_features.iloc[:int(len(all_features) * 0.67)]
test_all_features = all_features.iloc[int(len(all_features) * 0.67):]
train_label = Ground_truth[:int(len(Ground_truth) * 0.67)]
test_label = Ground_truth[int(len(Ground_truth) * 0.67):]
train_features_names = list(all_features.columns)
test_features_names = list(all_features.columns)
lgb_train = lgb.Dataset(train_all_features, train_label)
lgb_test = lgb.Dataset(test_all_features, test_label)
# %%
boost_round = BOOST_ROUND
early_stop = EARLY_STOP
param_final = PARAM_ALL_SENSOR
# %%
results_all_features = {}
lgbm_estimator_all_features = lgb.train(params=param_final, train_set=lgb_train, num_boost_round=boost_round, valid_sets=[
                                        lgb_train, lgb_test], valid_names=('Training', 'Testing'), early_stopping_rounds=early_stop, evals_result=results_all_features, verbose_eval=100)
# %%
# *predict
y_predict_all_features = lgbm_estimator_all_features.predict(
    test_all_features, num_iteration=lgbm_estimator_all_features.best_iteration)
y_predict_alltime_all_features = lgbm_estimator_all_features.predict(
    all_features, num_iteration=lgbm_estimator_all_features.best_iteration)
# %%
#!select features
#!get the importance of time split
features_importance_timesplit = lgbm_estimator_all_features.feature_importance()
features_names_timesplit = lgbm_estimator_all_features.feature_name()
features_importance_timesplit = pd.DataFrame(features_importance_timesplit)
features_names_timesplit = pd.DataFrame(features_names_timesplit)
all_features_importance_timesplit = pd.concat(
    [features_names_timesplit, features_importance_timesplit], axis=1)
all_features_importance_timesplit.columns = [
    'Sensor_features', 'Importance_values']
all_features_importance_timesplit.sort_values(
    by='Importance_values', axis=0, ascending=False, inplace=True)
all_features_importance_timesplit = all_features_importance_timesplit.reset_index(
    drop=True)
# %%
# TODO choose the selected features
selected_features_name = all_features_importance_timesplit.iloc[0:134, 0]
selected_features_data = [all_features[i] for i in selected_features_name]
selected_features_data = pd.DataFrame(selected_features_data).T
# %%
#!build selected plus data
# *build mean selected  features data
selected_mean_features_data = []
for i in range(selected_features_data.shape[1]):
    selected_mean_features_data.append(
        mean_windows(selected_features_data.iloc[:, i]))
selected_mean_features_data = pd.DataFrame(selected_mean_features_data).T
# *name meanselected features
selected_mean_features_name = [i+' Mean' for i in selected_features_name]
selected_mean_features_data.columns = selected_mean_features_name
# %%
# *build variance selected features data
selected_variance_features_data = []
for i in range(selected_features_data.shape[1]):
    selected_variance_features_data.append(
        variance_windows(selected_features_data.iloc[:, i]))
selected_variance_features_data = pd.DataFrame(
    selected_variance_features_data).T
# *name var selected features
selected_variance_features_name = [
    i+' Variance' for i in selected_features_name]
selected_variance_features_data.columns = selected_variance_features_name
# %%
selected_skewness_features_data = []
k = list(range(134))
for i in tqdm(k, leave=True, ncols=100, mininterval=1, desc='processing'):
    selected_skewness_features_data.append(
        skewness_windows(selected_features_data.iloc[:, i]))
selected_skewness_features_data = pd.DataFrame(
    selected_skewness_features_data).T
# *name skewness features
selected_skewness_features_name = [
    i+' Skewness' for i in selected_features_name]
selected_skewness_features_data.columns = selected_skewness_features_name
# %%
selected_kurtosis_features_data = []
k = list(range(134))
for i in tqdm(k, leave=True, ncols=100, mininterval=1, desc='processing'):
    selected_kurtosis_features_data.append(
        kurtosis_windows(selected_features_data.iloc[:, i]))
selected_kurtosis_features_data = pd.DataFrame(
    selected_kurtosis_features_data).T
# *name features
selected_kurtosis_features_name = [
    i+' Kurtosis' for i in selected_features_name]
selected_kurtosis_features_data.columns = selected_kurtosis_features_name
# %%
selected_interquartile_features_data = []
k = list(range(134))
for i in tqdm(k, leave=True, ncols=100, mininterval=1, desc='processing'):
    selected_interquartile_features_data.append(
        interquartile_windows(selected_features_data.iloc[:, i]))
selected_interquartile_features_data = pd.DataFrame(
    selected_interquartile_features_data).T
# *name features
selected_interquartile_features_name = [
    i+' Interquartile' for i in selected_features_name]
selected_interquartile_features_data.columns = selected_interquartile_features_name
# %%
# %%
# * build the percentile selected features data
selected_percentile_1_features_data = []
selected_percentile_91_features_data = []
k = list(range(134))
for i in tqdm(k, leave=True, ncols=100, mininterval=1, desc='processing'):
    temp1, temp2 = percentile_windows(selected_features_data.iloc[:, i])
    selected_percentile_1_features_data.append(temp1)
    selected_percentile_91_features_data.append(temp2)
selected_percentile_1_features_data = pd.DataFrame(
    selected_percentile_1_features_data).T
selected_percentile_91_features_data = pd.DataFrame(
    selected_percentile_91_features_data).T
# *name features
selected_percentile_1_features_name = [
    i+' Percentile1' for i in selected_features_name]
selected_percentile_91_features_name = [
    i+' Percentile91' for i in selected_features_name]
selected_percentile_1_features_data.columns = selected_percentile_1_features_name
selected_percentile_91_features_data.columns = selected_percentile_91_features_name
# %%
# *build the median selected features data
selected_median_features_data = []
k = list(range(134))
for i in tqdm(k, leave=True, ncols=100, mininterval=1, desc='processing'):
    selected_median_features_data.append(
        median_windows(selected_features_data.iloc[:, i]))
selected_median_features_data = pd.DataFrame(
    selected_median_features_data).T
# *name features
selected_median_features_name = [
    i+' Median' for i in selected_features_name]
selected_median_features_data.columns = selected_median_features_name
# %%
# TODO selected features model
# ? preapre data in time split
train_all_features = selected_features_data.iloc[:int(
    len(all_features) * 0.67)]
test_all_features = selected_features_data.iloc[int(len(all_features) * 0.67):]
train_label = Ground_truth[:int(len(Ground_truth) * 0.67)]
test_label = Ground_truth[int(len(Ground_truth) * 0.67):]
train_features_names = list(selected_features_data.columns)
test_features_names = list(selected_features_data.columns)
# ! standard the data or not
x_train = train_all_features
y_train = train_label
x_test = test_all_features
y_test = test_label
t_train = t[:int(len(t) * 0.67)]
t_predct = t[int(len(t) * 0.67):]
standard = StandardScaler()
standard.fit(all_features)
# x_train_sd = standar.transform(x_train)
# x_test_sd = standar.transform(x_test)
x_train_sd = np.array(x_train)
x_test_sd = np.array(x_test)
y_train = y_train
y_test = y_test
lgb_train = lgb.Dataset(x_train, y_train)
lgb_test = lgb.Dataset(x_test, y_test)
# %%
boost_round = BOOST_ROUND
early_stop = EARLY_STOP
param_final = PARAM_SELECTED_SENSOR
# %%
results_all_features = {}
lgbm_estimator_all_features = lgb.train(params=param_final, train_set=lgb_train, num_boost_round=boost_round, valid_sets=[
                                        lgb_train, lgb_test], valid_names=('Training', 'Testing'), early_stopping_rounds=early_stop, evals_result=results_all_features, verbose_eval=100)
# %%
# *predict
y_predict_all_features = lgbm_estimator_all_features.predict(
    x_test, num_iteration=lgbm_estimator_all_features.best_iteration)
y_predict_alltime_all_features = lgbm_estimator_all_features.predict(
    selected_features_data, num_iteration=lgbm_estimator_all_features.best_iteration)
# %%
# TODO selected & statistic features model
selected_plus_features = pd.concat([selected_features_data, selected_mean_features_data, selected_variance_features_data, selected_skewness_features_data, selected_kurtosis_features_data,
                                    selected_interquartile_features_data, selected_percentile_1_features_data, selected_percentile_91_features_data, selected_median_features_data], axis=1)
# ?data prepare for training
# %%
# split and name
train_selected_plus_features = selected_plus_features.iloc[:16750]
test_selected_plus_features = selected_plus_features.iloc[16750:]
train_selected_plus_label = np.array(Friction_coef[:16750].squeeze())
test_selected_plus_label = np.array(Friction_coef[16750:].squeeze())
selected_plus_features_names = list(train_selected_plus_features.columns)
#!build the selected Plus features in lgb style timesplit
lgb_train_selected_plus = lgb.Dataset(
    train_selected_plus_features, train_selected_plus_label, feature_name=selected_plus_features_names)
lgb_val_selected_plus = lgb.Dataset(
    test_selected_plus_features, test_selected_plus_label, feature_name=selected_plus_features_names)
# ! standard the data or not
x_train = train_selected_plus_features
y_train = train_selected_plus_label
x_test = test_selected_plus_features
y_test = test_selected_plus_label
t_train = t[:int(len(t) * 0.67)]
t_predct = t[int(len(t) * 0.67):]
standard = StandardScaler()
standard.fit(all_features)
# x_train_sd = standar.transform(x_train)
# x_test_sd = standar.transform(x_test)
x_train_sd = np.array(x_train)
x_test_sd = np.array(x_test)
y_train = y_train
y_test = y_test
lgb_train = lgb.Dataset(x_train, y_train)
lgb_test = lgb.Dataset(x_test, y_test)
# %%
boost_round = BOOST_ROUND
early_stop = EARLY_STOP
param_timesplit_final = PARAM_SELECTED_STATISTIC
# %%
results_selected_plus_final = {}
lgbm_estimator_selected_plus_timesplit = lgb.train(param_timesplit_final, lgb_train_selected_plus, num_boost_round=boost_round, valid_sets=[
    lgb_train_selected_plus, lgb_val_selected_plus], valid_names=('Training', 'Testing'), early_stopping_rounds=early_stop, evals_result=results_selected_plus_final, verbose_eval=50)
# %%
# *predict
y_predict_selected_plus = lgbm_estimator_selected_plus_timesplit.predict(
    test_selected_plus_features, num_iteration=lgbm_estimator_selected_plus_timesplit.best_iteration)
y_predict_alltimeseries_selected_plus = lgbm_estimator_selected_plus_timesplit.predict(
    selected_plus_features, num_iteration=lgbm_estimator_selected_plus_timesplit.best_iteration)
print('The R2=', coefficient_determination(
    y_predict_selected_plus, Ground_truth[16750:]))
# %%
# TODO selected & statistic features model train with 20000 time step (8/10)
# ? data prepare for training
# split and name
train_selected_plus_features = selected_plus_features.iloc[:20000]
test_selected_plus_features = selected_plus_features.iloc[20000:]
train_selected_plus_label = np.array(Friction_coef[:20000].squeeze())
test_selected_plus_label = np.array(Friction_coef[20000:].squeeze())
selected_plus_features_names = list(train_selected_plus_features.columns)
#!build the selected Plus features in lgb style timesplit
lgb_train_selected_plus = lgb.Dataset(
    train_selected_plus_features, train_selected_plus_label, feature_name=selected_plus_features_names)
lgb_val_selected_plus = lgb.Dataset(
    test_selected_plus_features, test_selected_plus_label, feature_name=selected_plus_features_names)
#! standard the data or not
x_train = train_selected_plus_features
y_train = train_selected_plus_label
x_test = test_selected_plus_features
y_test = test_selected_plus_label
t_train = t[:int(len(t) * 0.80)]
t_predct = t[int(len(t) * 0.80):]
standard = StandardScaler()
standard.fit(all_features)
# x_train_sd = standar.transform(x_train)
# x_test_sd = standar.transform(x_test)
x_train_sd = np.array(x_train)
x_test_sd = np.array(x_test)
y_train = y_train
y_test = y_test
lgb_train = lgb.Dataset(x_train, y_train)
lgb_test = lgb.Dataset(x_test, y_test)
# %%
boost_round = BOOST_ROUND
early_stop = EARLY_STOP
param_timesplit_final = PARAM_SELECTED_STATISTIC_8_10
# %%
results_selected_plus_final = {}
lgbm_estimator_selected_plus_timesplit = lgb.train(param_timesplit_final, lgb_train_selected_plus, num_boost_round=boost_round, valid_sets=[
                                                   lgb_train_selected_plus, lgb_val_selected_plus], valid_names=('Training', 'Testing'), early_stopping_rounds=early_stop, evals_result=results_selected_plus_final, verbose_eval=50)
# %%

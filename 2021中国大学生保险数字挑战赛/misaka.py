import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import gc
import os
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from itertools import product
from gensim.models import Word2Vec
from collections import OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
import time
from itertools import combinations
from datetime import datetime

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
warnings.filterwarnings('ignore')
seed = 2021

df_train = pd.read_csv('/home/mw/input/pre8881/train.csv')
df_test = pd.read_csv('/home/mw/input/pretest_b6354/test_b.csv')


# 平均编码类定义
class MeanEncoder:
    def __init__(self, categorical_features, n_splits=5, target_type='classification', prior_weight_func=None):
        """
        :param categorical_features: list of str, the name of the categorical columns to encode

        :param n_splits: the number of splits used in mean encoding

        :param target_type: str, 'regression' or 'classification'

        :param prior_weight_func:
        a function that takes in the number of observations, and outputs prior weight
        when a dict is passed, the default exponential decay function will be used:
        k: the number of observations needed for the posterior to be weighted equally as the prior
        f: larger f --> smaller slope
        """

        self.categorical_features = categorical_features
        self.n_splits = n_splits
        self.learned_stats = {}

        if target_type == 'classification':
            self.target_type = target_type
            self.target_values = []
        else:
            self.target_type = 'regression'
            self.target_values = None

        if isinstance(prior_weight_func, dict):
            self.prior_weight_func = eval('lambda x: 1 / (1 + np.exp((x - k) / f))', dict(prior_weight_func, np=np))
        elif callable(prior_weight_func):
            self.prior_weight_func = prior_weight_func
        else:
            self.prior_weight_func = lambda x: 1 / (1 + np.exp((x - 2) / 1))

    @staticmethod
    def mean_encode_subroutine(X_train, y_train, X_test, variable, target, prior_weight_func):
        X_train = X_train[[variable]].copy()
        X_test = X_test[[variable]].copy()

        if target is not None:
            nf_name = '{}_pred_{}'.format(variable, target)
            X_train['pred_temp'] = (y_train == target).astype(int)  # classification
        else:
            nf_name = '{}_pred'.format(variable)
            X_train['pred_temp'] = y_train  # regression
        prior = X_train['pred_temp'].mean()

        col_avg_y = X_train.groupby(by=variable, axis=0)['pred_temp'].agg({'mean': 'mean', 'beta': 'size'})
        col_avg_y['beta'] = prior_weight_func(col_avg_y['beta'])
        col_avg_y[nf_name] = col_avg_y['beta'] * prior + (1 - col_avg_y['beta']) * col_avg_y['mean']
        col_avg_y.drop(['beta', 'mean'], axis=1, inplace=True)

        nf_train = X_train.join(col_avg_y, on=variable)[nf_name].values
        nf_test = X_test.join(col_avg_y, on=variable).fillna(prior, inplace=False)[nf_name].values

        return nf_train, nf_test, prior, col_avg_y

    def fit_transform(self, X, y):
        """
        :param X: pandas DataFrame, n_samples * n_features
        :param y: pandas Series or numpy array, n_samples
        :return X_new: the transformed pandas DataFrame containing mean-encoded categorical features
        """
        X_new = X.copy()
        if self.target_type == 'classification':
            skf = StratifiedKFold(self.n_splits)
        else:
            skf = KFold(self.n_splits)

        if self.target_type == 'classification':
            self.target_values = sorted(set(y))
            self.learned_stats = {'{}_pred_{}'.format(variable, target): [] for variable, target in
                                  product(self.categorical_features, self.target_values)}
            for variable, target in product(self.categorical_features, self.target_values):
                nf_name = '{}_pred_{}'.format(variable, target)
                X_new.loc[:, nf_name] = np.nan
                for large_ind, small_ind in skf.split(y, y):
                    nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
                        X_new.iloc[large_ind], y.iloc[large_ind], X_new.iloc[small_ind], variable, target,
                        self.prior_weight_func)
                    X_new.iloc[small_ind, -1] = nf_small
                    self.learned_stats[nf_name].append((prior, col_avg_y))
        else:
            self.learned_stats = {'{}_pred'.format(variable): [] for variable in self.categorical_features}
            for variable in self.categorical_features:
                nf_name = '{}_pred'.format(variable)
                X_new.loc[:, nf_name] = np.nan
                for large_ind, small_ind in skf.split(y, y):
                    nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
                        X_new.iloc[large_ind], y.iloc[large_ind], X_new.iloc[small_ind], variable, None,
                        self.prior_weight_func)
                    X_new.iloc[small_ind, -1] = nf_small
                    self.learned_stats[nf_name].append((prior, col_avg_y))
        return X_new

    def transform(self, X):
        """
        :param X: pandas DataFrame, n_samples * n_features
        :return X_new: the transformed pandas DataFrame containing mean-encoded categorical features
        """
        X_new = X.copy()

        if self.target_type == 'classification':
            for variable, target in product(self.categorical_features, self.target_values):
                nf_name = '{}_pred_{}'.format(variable, target)
                X_new[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    X_new[nf_name] += X_new[[variable]].join(col_avg_y, on=variable).fillna(prior, inplace=False)[
                        nf_name]
                X_new[nf_name] /= self.n_splits
        else:
            for variable in self.categorical_features:
                nf_name = '{}_pred'.format(variable)
                X_new[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    X_new[nf_name] += X_new[[variable]].join(col_avg_y, on=variable).fillna(prior, inplace=False)[
                        nf_name]
                X_new[nf_name] /= self.n_splits

        return X_new


def kfold_stats_feature(train, test, feats, k):
    folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=6666)  # 这里最好和后面模型的K折交叉验证保持一致

    train['fold'] = None
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['y1_is_purchase'])):
        train.loc[val_idx, 'fold'] = fold_

    kfold_features = []
    for feat in feats:
        nums_columns = ['y1_is_purchase']
        for f in nums_columns:
            colname = feat + '_' + f + '_kfold_mean'
            kfold_features.append(colname)
            train[colname] = None
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['y1_is_purchase'])):
                tmp_trn = train.iloc[trn_idx]
                order_label = tmp_trn.groupby([feat])[f].mean()
                tmp = train.loc[train.fold == fold_, [feat]]
                train.loc[train.fold == fold_, colname] = tmp[feat].map(order_label)
                # fillna
                global_mean = train[f].mean()
                train.loc[train.fold == fold_, colname] = train.loc[train.fold == fold_, colname].fillna(global_mean)
            train[colname] = train[colname].astype(float)

        for f in nums_columns:
            colname = feat + '_' + f + '_kfold_mean'
            test[colname] = None
            order_label = train.groupby([feat])[f].mean()
            test[colname] = test[feat].map(order_label)
            # fillna
            global_mean = train[f].mean()
            test[colname] = test[colname].fillna(global_mean)
            test[colname] = test[colname].astype(float)
    del train['fold']
    return train, test


# 数据处理
df_train['p2_client_grade'] = df_train['p2_client_grade'].map({'车主俱乐部-黄铜客户-2': 1,
                                                               '车主俱乐部-白银客户-2': 2,
                                                               '车主俱乐部-黄金客户-2': 3,
                                                               '车主俱乐部-铂金客户-2': 4,
                                                               '车主俱乐部-钻石客户-2': 5,
                                                               '车主俱乐部-黑钻客户-2': 6})
df_test['p2_client_grade'] = df_test['p2_client_grade'].map({'车主俱乐部-黄铜客户-2': 1,
                                                             '车主俱乐部-白银客户-2': 2,
                                                             '车主俱乐部-黄金客户-2': 3,
                                                             '车主俱乐部-铂金客户-2': 4,
                                                             '车主俱乐部-钻石客户-2': 5,
                                                             '车主俱乐部-黑钻客户-2': 6})


# %%

def car_level(x):
    if x <= 1:
        return 1
    elif x < 1 and x <= 1.6:
        return 2
    elif 1.6 < x <= 2.5:
        return 3
    elif 2.5 < x <= 4:
        return 4
    elif 4 < x <= 6:
        return 5
    else:
        return 0


def age_level(x):
    if x < 18:
        return 1
    elif 18 < x <= 23:
        return 2
    elif 23 < x <= 27:
        return 3
    elif 27 < x <= 35:
        return 4
    elif 35 < x <= 45:
        return 5
    elif 45 < x <= 55:
        return 6
    elif 55 < x <= 65:
        return 7
    elif 65 < x < 80:
        return 8
    elif pd.isna(x):
        return 0
    else:
        return 9


def seats(x):
    if pd.isna(x):
        return 0
    elif x < 4:
        return 1
    elif 4 <= x < 7:
        return 2
    elif 7 < x < 10:
        return 3
    else:
        return 4


def house_cust(x):
    if x == 0:
        return 0
    elif 0 < x <= 50:
        return 1
    elif 50 <= x <= 128:
        return 2
    elif 128 < x <= 288:
        return 3
    elif 288 < x <= 500:
        return 4
    elif 500 < x <= 888:
        return 5
    elif 888 < x <= 1288:
        return 6
    elif 1288 <= x < 2888:
        return 7
    elif pd.isna(x):
        return np.nan
    elif 2888 <= x <= 10000:
        return 8
    else:
        return 9


def insurance_01(x):
    if x == 0:
        return 0
    elif 0 < x <= 100:
        return 1
    elif 100 < x < 1000:
        return 2
    elif 1000 <= x <= 2000:
        return 3
    elif 2000 < x <= 5000:
        return 4
    elif 5000 < x <= 10000:
        return 5
    elif pd.isna(x):
        return np.nan
    elif 10000 < x < 54000:
        return 6
    elif 54000 <= x < 100000:
        return 7
    else:
        return 8


def score_bin(x):
    if pd.isna(x):
        return np.nan
    elif x == 0:
        return 0
    elif 0 < x <= 5:
        return 1
    elif 5 < x < 20:
        return 2
    elif 20 <= x < 50:
        return 3
    elif 50 <= x < 150:
        return 4
    elif 150 <= x <= 500:
        return 5
    elif 500 < x <= 1000:
        return 6
    elif 1000 < x <= 1500:
        return 7
    elif 1500 < x < 5000:
        return 8
    elif 5000 <= x < 10000:
        return 9
    elif 10000 <= x < 20000:
        return 10
    elif 20000 <= x <= 40000:
        return 11
    elif 40000 < x <= 100000:
        return 12
    else:
        return 13


df_train['capab_level'] = df_train['capab'].map(car_level)
df_train['age_level'] = df_train['p1_age'].map(age_level)
df_test['capab_level'] = df_test['capab'].map(car_level)
df_test['age_level'] = df_test['p1_age'].map(age_level)
df_train['seats_level'] = df_train['seats'].map(seats)
df_train['f2_cust_housing_price_level'] = df_train['f2_cust_housing_price_total'].map(house_cust)
df_train['dur_personal_insurance_90_level'] = df_train['dur_personal_insurance_90'].map(insurance_01)
df_train['service_score_available_level'] = df_train['service_score_available'].map(score_bin)
df_test['seats_level'] = df_test['seats'].map(seats)
df_test['f2_cust_housing_price_level'] = df_test['f2_cust_housing_price_total'].map(house_cust)
df_test['dur_personal_insurance_90_level'] = df_test['dur_personal_insurance_90'].map(insurance_01)
df_test['service_score_available_level'] = df_test['service_score_available'].map(score_bin)


class_list = [
    'trademark_cn',
    'brand_cn',
    'change_owner',
    'p1_is_bank_eff',
    'w1_pc_wx_use_flag',
    'p2_client_grade',
    'p2_is_enterprise_owner',
    'p1_age',
    'capab_level',
    'age_level',
    'dur_personal_insurance_90_level',
    'f2_cust_housing_price_level',
    'service_score_available_level',
    'suiche_nonauto_amount_20',
    'si_od',
]

MeanEnocodeFeature = class_list  # 声明需要平均数编码的特征
ME = MeanEncoder(MeanEnocodeFeature, target_type='classification')  # 声明平均数编码的类

train = ME.fit_transform(
    df_train,
    df_train['y1_is_purchase'])  # 对训练数据集的X和y进行拟合
# x_train_fav = ME.fit_transform(x_train,y_train_fav)#对训练数据集的X和y进行拟合
test = ME.transform(df_test)  # 对测试集进行编码
print('num0:mean_encode train.shape', df_train.shape, df_test.shape)

df_train, df_test = kfold_stats_feature(train, test, class_list, 5)
print('num1:target_encode train.shape', train.shape, test.shape)

df_feature = df_train.append(df_test, sort=False)
df_feature['p1_gender'].fillna(0, inplace=True)
df_feature['p1_age'].fillna(0, inplace=True)


def set_nan(x):
    if x < 0:
        return np.nan
    else:
        return x


df_feature['p1_prior_days_to_insure'] = df_feature['p1_prior_days_to_insure'].map(set_nan)
df_feature['nprem_ly'] = df_feature['nprem_ly'].map(set_nan)


def judge(n):
    n1 = str(float(n))
    n2 = n1.split('.')
    if n == 0:
        return 0
    elif pd.isna(n):
        return np.nan
    elif n2[1] == '0':
        return 2
    else:
        return 1


df_feature['suiche_nonauto_nprem_20_level'] = df_feature['suiche_nonauto_nprem_20'].map(judge)
df_feature['suiche_nonauto_nprem_sum_level'] = df_feature[['suiche_nonauto_nprem_16',
                                                           'suiche_nonauto_nprem_17',
                                                           'suiche_nonauto_nprem_18',
                                                           'suiche_nonauto_nprem_19',
                                                           'suiche_nonauto_nprem_20']].sum(axis=1).map(judge)


def reduce_mem(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('{:.2f} Mb, {:.2f} Mb ({:.2f} %)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    gc.collect()
    return df


df_feature = reduce_mem(df_feature)

df_feature['client_no_suiche_nonauto_amount_20_count'] = df_feature.groupby('client_no')[
    'suiche_nonauto_amount_20'].transform('count').values
df_feature['suiche_nonauto_amount_20_client_no_count'] = df_feature.groupby('suiche_nonauto_amount_20')[
    'client_no'].transform('count').values

df_feature['client_no_suiche_nonauto_amount_20_nunique'] = df_feature.groupby('client_no')[
    'suiche_nonauto_amount_20'].transform('nunique').values
df_feature['suiche_nonauto_amount_20_client_no_nunique'] = df_feature.groupby('suiche_nonauto_amount_20')[
    'client_no'].transform('nunique').values

df_feature['client_no_suiche_nonauto_amount_20_count_nunique'] = df_feature[
                                                                     'client_no_suiche_nonauto_amount_20_count'].values / \
                                                                 df_feature[
                                                                     'client_no_suiche_nonauto_amount_20_nunique'].values
df_feature['suiche_nonauto_amount_20_client_no_count_nunique'] = df_feature[
                                                                     'suiche_nonauto_amount_20_client_no_count'].values / \
                                                                 df_feature[
                                                                     'suiche_nonauto_amount_20_client_no_nunique'].values

# %%

df_feature['client_no_dpt_count'] = df_feature.groupby('client_no')['dpt'].transform('count').values
df_feature['dpt_client_no_count'] = df_feature.groupby('dpt')['client_no'].transform('count').values

df_feature['client_no_dpt_nunique'] = df_feature.groupby('client_no')['dpt'].transform('nunique').values
df_feature['dpt_client_no_nunique'] = df_feature.groupby('dpt')['client_no'].transform('nunique').values

df_feature['client_no_dpt_count_nunique'] = df_feature['client_no_dpt_count'].values / df_feature[
    'client_no_dpt_nunique'].values
df_feature['dpt_client_no_count_nunique'] = df_feature['dpt_client_no_count'].values / df_feature[
    'dpt_client_no_nunique'].values

df_feature['client_no_p1_prior_days_to_insure_count'] = df_feature.groupby('client_no')[
    'p1_prior_days_to_insure'].transform('count').values
df_feature['p1_prior_days_to_insure_client_no_count'] = df_feature.groupby('p1_prior_days_to_insure')[
    'client_no'].transform('count').values

df_feature['client_no_p1_prior_days_to_insure_nunique'] = df_feature.groupby('client_no')[
    'p1_prior_days_to_insure'].transform('nunique').values
df_feature['p1_prior_days_to_insure_client_no_nunique'] = df_feature.groupby('p1_prior_days_to_insure')[
    'client_no'].transform('nunique').values

df_feature['client_no_p1_prior_days_to_insure_count_nunique'] = df_feature[
                                                                    'client_no_p1_prior_days_to_insure_count'].values / \
                                                                df_feature[
                                                                    'client_no_p1_prior_days_to_insure_nunique'].values
df_feature['p1_prior_days_to_insure_client_no_count_nunique'] = df_feature[
                                                                    'p1_prior_days_to_insure_client_no_count'].values / \
                                                                df_feature[
                                                                    'p1_prior_days_to_insure_client_no_nunique'].values

# %%

df_feature['dur_personal_insurance_avg'] = df_feature['dur_personal_insurance_90'] / \
                                           (df_feature['active_90'] + 1e-10)
df_feature['p3_service_use_avg'] = df_feature['p3_service_use_cnt'] / \
                                   (df_feature['active_30'] + 1e-10)
df_feature['p1_service_offer_avg'] = df_feature['p1_service_offer_cnt'] / \
                                     (df_feature['active_30'] + 1e-10)
df_feature['vld/vlp'] = df_feature['nprem_vld'] / (df_feature['nprem_vlp'] + 1e-10)
df_feature['use/offer'] = df_feature['p3_service_use_cnt'] / \
                          (df_feature['p1_service_offer_cnt'] + 1e-10)

df_feature['p2_owner'] = (df_feature['p2_is_enterprise_owner'].map(
    {'是': 1, '否': 0}) + df_feature['p2_is_smeowner'].map({'是': 1, '否': 0}) + df_feature['f2_posses_house_flag'].map(
    {'是': 1, '否': 0})) / 3
df_feature['p2_family'] = (df_feature['p2_is_child_under_15_family'].map({'是': 1, '否': 0}) + df_feature[
    'p2_is_adult_over_55_family'].map({'是': 1, '否': 0})) / 2

df_feature['90/7'] = df_feature['active_90'] / (1e-10 + df_feature['active_7'])
df_feature['365/7'] = df_feature['active_365'] / (1e-10 + df_feature['active_7'])
df_feature['30/7'] = df_feature['active_30'] / (1e-10 + df_feature['active_7'])
df_feature['90/30'] = df_feature['active_90'] / (1e-10 + df_feature['active_30'])
df_feature['365/30'] = df_feature['active_365'] / (1e-10 + df_feature['active_30'])
df_feature['365/90'] = df_feature['active_365'] / (1e-10 + df_feature['active_90'])

# %%

# 随车非车险费额比
df_feature['20_ratio'] = df_feature['suiche_nonauto_nprem_20'] / \
                         df_feature['suiche_nonauto_amount_20']
df_feature['19_ratio'] = df_feature['suiche_nonauto_nprem_19'] / \
                         df_feature['suiche_nonauto_amount_19']
df_feature['17_ratio'] = df_feature['suiche_nonauto_nprem_17'] / \
                         df_feature['suiche_nonauto_amount_17']
df_feature['16_ratio'] = df_feature['suiche_nonauto_nprem_16'] / \
                         df_feature['suiche_nonauto_amount_16']

# 今年保费
df_feature['nprem_now'] = df_feature['nprem_ly'] * df_feature['ncd_ly']
# 商三险费额比
df_feature['od_ratio'] = df_feature['nprem_od'] / df_feature['si_od']
df_feature['tp_ratio'] = df_feature['nprem_tp'] / df_feature['si_tp']
df_feature['bt_ratio'] = df_feature['nprem_bt'] / df_feature['si_bt']
df_feature['vld_ratio'] = df_feature['nprem_vld'] / df_feature['si_vld']
df_feature['vlp_ratio'] = df_feature['nprem_vlp'] / df_feature['si_vlp']

df_feature['now/ly'] = df_feature['nprem_now'] / (df_feature['nprem_ly'] + 1e-10)
df_feature['nprem_all_total'] = df_feature[['nprem_od', 'nprem_tp', 'nprem_bt', 'nprem_vld', 'nprem_vlp']].sum(axis=1)
df_feature['nprem_ratio'] = df_feature['nprem_all_total'] / (df_feature['nprem_ly'] + 1e-10)
# 总保费
df_feature['nprem_total'] = df_feature['nprem_ly'] + \
                            df_feature['suiche_nonauto_nprem_20']
# 总保额
df_feature['si_total'] = df_feature['si_od'] + df_feature['si_tp'] + \
                         df_feature['si_bt'] + df_feature['si_vld'] + df_feature['si_vlp']
df_feature['total_ratio'] = df_feature['nprem_total'] / df_feature['si_total']
df_feature['people_ratio'] = (df_feature['nprem_vld'] +
                              df_feature['nprem_vlp']) / df_feature['nprem_total']
df_feature['nonauto/nprem_ly'] = df_feature['suiche_nonauto_nprem_20'] / \
                                 df_feature['nprem_ly']
df_feature['nonauto_ratio'] = df_feature['suiche_nonauto_nprem_20'] / \
                              df_feature['nprem_total']
df_feature['vld/vlp'] = df_feature['nprem_vld'] / (df_feature['nprem_vlp'] + 1e-10)
# 平均保费
df_feature['nprem_avg'] = df_feature['nprem_ly'] / (1e-10 + df_feature['clmnum'])

df_feature['birth_year'] = 2021 - df_feature['p1_age']
df_feature['birth_month'] = df_feature['birth_month'].apply(
    lambda x: int(x[:-1]) if not isinstance(x, float) else 0)
df_feature['birth_year_month'] = 100 * \
                                 df_feature['birth_year'] + df_feature['birth_month']
df_feature['suiche_nonauto_nprem_21'] = df_feature['suiche_nonauto_nprem_20'] + \
                                        (df_feature['suiche_nonauto_nprem_20'] -
                                         df_feature['suiche_nonauto_nprem_16']) / 5

df_feature['car_day'] = (datetime.now() - pd.to_datetime(df_feature['regdate'])).dt.days
df_feature['regdate_year'] = pd.to_datetime(df_feature['regdate']).dt.year
df_feature['regdate_month'] = pd.to_datetime(df_feature['regdate']).dt.month
df_feature['regdate_date'] = df_feature['regdate_year'] * 100 + df_feature['regdate_month']

# %%

df_feature['brand_series'] = df_feature['brand_cn'].astype(str) + '_' + df_feature['series'].astype(str)
df_feature['trademark_cn_series'] = df_feature['trademark_cn'].astype(str) + '_' + df_feature['series'].astype(str)
df_feature['p1_prior_days_to_insure_active_365'] = df_feature['p1_prior_days_to_insure'].astype(str) + '_' + df_feature[
    'active_365'].astype(str)
df_feature['p1_prior_days_to_insure_active_90'] = df_feature['p1_prior_days_to_insure'].astype(str) + '_' + df_feature[
    'active_90'].astype(str)
df_feature['p1_prior_days_to_insure_active_30'] = df_feature['p1_prior_days_to_insure'].astype(str) + '_' + df_feature[
    'active_30'].astype(str)
df_feature['p1_prior_days_to_insure_active_7'] = df_feature['p1_prior_days_to_insure'].astype(str) + '_' + df_feature[
    'active_7'].astype(str)
df_feature['p1_prior_days_to_insure_suiche_nonauto_nprem_20'] = df_feature['p1_prior_days_to_insure'].astype(
    str) + '_' + df_feature['suiche_nonauto_nprem_20'].astype(str)
df_feature['dur_personal_insurance_90_suiche_nonauto_nprem_20'] = df_feature['dur_personal_insurance_90'].astype(
    str) + '_' + df_feature['suiche_nonauto_nprem_20'].astype(str)
df_feature['p1_prior_days_to_insure_suiche_nonauto_nprem_20'] = df_feature['p1_prior_days_to_insure'].astype(
    str) + '_' + df_feature['suiche_nonauto_nprem_20'].astype(str)

df_feature['20-19'] = df_feature['suiche_nonauto_nprem_20'] - df_feature['suiche_nonauto_nprem_19']
df_feature['20-18'] = df_feature['suiche_nonauto_nprem_20'] - df_feature['suiche_nonauto_nprem_18']
df_feature['20-17'] = df_feature['suiche_nonauto_nprem_20'] - df_feature['suiche_nonauto_nprem_17']
df_feature['20-16'] = df_feature['suiche_nonauto_nprem_20'] - df_feature['suiche_nonauto_nprem_16']
df_feature['19-18'] = df_feature['suiche_nonauto_nprem_19'] - df_feature['suiche_nonauto_nprem_18']
df_feature['19-17'] = df_feature['suiche_nonauto_nprem_19'] - df_feature['suiche_nonauto_nprem_17']
df_feature['19-16'] = df_feature['suiche_nonauto_nprem_19'] - df_feature['suiche_nonauto_nprem_16']
df_feature['18-17'] = df_feature['suiche_nonauto_nprem_18'] - df_feature['suiche_nonauto_nprem_17']
df_feature['17-16'] = df_feature['suiche_nonauto_nprem_17'] - df_feature['suiche_nonauto_nprem_16']

df_feature['car_photo'] = df_feature['xz'].map({'商交': 1, '单交': 2}) + \
                          10 * df_feature['xb'].map({'交三': 1, '单交': 2, '主全': 2}) + \
                          100 * df_feature['use_type'].map({'营业': 1, '非营业': 2}) + \
                          1000 * df_feature['change_owner'].map({'非过户投保': 1, '过户投保': 2})

loace_map = dict([(name, label) for label, name in enumerate(list(df_feature['p1_census_register'].unique()))])
df_feature['base_photo'] = 100 * df_feature['p2_marital_status'].map(
    {'eNP+WqbTmmD3bj49nIcSew==': 1, '3x1ph65jyhpx1f7Xu3TELA==': 2, 'VW5/NBm6sxbhUnzkEzW5rg==': 3}) + \
                           10 * df_feature['p1_gender'].map(
    {'jh4mxXNEalwumcCWUJdnBw==': 1, 'yUh7960km3oydK6Km9rqRA==': 2, 0: 0}) + \
                           0.01 * df_feature['p1_census_register'].map(loace_map) + \
                           1000 * df_feature['f1_child_flag'].map({'是': 1, '否': 0})

df_feature['famliy_photo'] = df_feature['p2_marital_status'].map(
    {'eNP+WqbTmmD3bj49nIcSew==': 1, '3x1ph65jyhpx1f7Xu3TELA==': 2, 'VW5/NBm6sxbhUnzkEzW5rg==': 3}) + \
                             10 * df_feature['f1_child_flag'].map({'是': 1, '否': 0}) + \
                             100 * df_feature['p2_is_child_under_15_family'].map({'是': 1, '否': 0}) + \
                             1000 * df_feature['p2_is_adult_over_55_family'].map({'是': 1, '否': 0})

df_feature['property_photo'] = df_feature['p1_is_bank_eff'].map({'是': 1, '否': 0}) + \
                               10 * df_feature['p2_is_enterprise_owner'].map({'是': 1, '否': 0}) + \
                               100 * df_feature['p2_is_smeowner'].map({'是': 1, '否': 0}) + \
                               1000 * df_feature['f2_posses_house_flag'].map({'是': 1, '否': 0})

df_feature['man_photo'] = df_feature['p2_marital_status'].map(
    {'eNP+WqbTmmD3bj49nIcSew==': 1, '3x1ph65jyhpx1f7Xu3TELA==': 2, 'VW5/NBm6sxbhUnzkEzW5rg==': 3}) + \
                          100 * df_feature['age_level'] + \
                          10 * df_feature['f1_child_flag'].map({'是': 1, '否': 0}) + \
                          10000 * df_feature['p2_is_adult_over_55_family'].map({'是': 1, '否': 0}) + \
                          1000 * df_feature['f2_posses_house_flag'].map({'是': 1, '否': 0})

df_feature['young_photo'] = df_feature['p2_marital_status'].map(
    {'eNP+WqbTmmD3bj49nIcSew==': 1, '3x1ph65jyhpx1f7Xu3TELA==': 2, 'VW5/NBm6sxbhUnzkEzW5rg==': 3}) + \
                            100 * df_feature['age_level'] + \
                            10 * df_feature['f2_posses_house_flag'].map({'是': 1, '否': 0})

df_feature['19_avg'] = df_feature.groupby('suiche_nonauto_amount_19')['suiche_nonauto_nprem_19'].transform('mean')
df_feature['18_avg'] = df_feature.groupby('suiche_nonauto_amount_18')['suiche_nonauto_nprem_18'].transform('mean')
df_feature['17_avg'] = df_feature.groupby('suiche_nonauto_amount_17')['suiche_nonauto_nprem_17'].transform('mean')
df_feature['16_avg'] = df_feature.groupby('suiche_nonauto_amount_16')['suiche_nonauto_nprem_16'].transform('mean')

df_feature['car_photo_avg'] = df_feature.groupby('car_photo')['suiche_nonauto_nprem_20'].transform('mean')
df_feature['base_photo_avg'] = df_feature.groupby('base_photo')['suiche_nonauto_nprem_20'].transform('mean')
df_feature['famliy_photo_avg'] = df_feature.groupby('famliy_photo')['suiche_nonauto_nprem_20'].transform('mean')
df_feature['property_photo_avg'] = df_feature.groupby('property_photo')['suiche_nonauto_nprem_20'].transform('mean')

df_feature['p1_census_register_len'] = df_feature['p1_census_register'].astype('str').map(lambda x: len(x))

df_feature['brand_series'] = df_feature['brand_cn'].astype(str) + '_' + df_feature['series'].astype(str)
df_feature['trademark_cn_series'] = df_feature['trademark_cn'].astype(str) + '_' + df_feature['series'].astype(str)
df_feature['p1_prior_days_to_insure_active_365'] = df_feature['p1_prior_days_to_insure'].astype(str) + '_' + df_feature[
    'active_365'].astype(str)
df_feature['p1_prior_days_to_insure_active_90'] = df_feature['p1_prior_days_to_insure'].astype(str) + '_' + df_feature[
    'active_90'].astype(str)
df_feature['p1_prior_days_to_insure_active_30'] = df_feature['p1_prior_days_to_insure'].astype(str) + '_' + df_feature[
    'active_30'].astype(str)
df_feature['p1_prior_days_to_insure_active_7'] = df_feature['p1_prior_days_to_insure'].astype(str) + '_' + df_feature[
    'active_7'].astype(str)
df_feature['p1_prior_days_to_insure_suiche_nonauto_amount_20'] = df_feature['p1_prior_days_to_insure'].astype(
    str) + '_' + df_feature['suiche_nonauto_amount_20'].astype(str)
df_feature['dur_personal_insurance_90_suiche_nonauto_amount_20'] = df_feature['dur_personal_insurance_90'].astype(
    str) + '_' + df_feature['suiche_nonauto_amount_20'].astype(str)
df_feature['p1_prior_days_to_insure_suiche_nonauto_amount_20'] = df_feature['p1_prior_days_to_insure'].astype(
    str) + '_' + df_feature['suiche_nonauto_amount_20'].astype(str)

df_feature['suiche_nonauto_nprem_20_base_photo_avg'] = df_feature['suiche_nonauto_nprem_20'] - df_feature[
    'base_photo_avg']
df_feature['suiche_nonauto_amount_20_p3_service_use_cnt'] = df_feature.groupby('suiche_nonauto_amount_20')[
    'p3_service_use_cnt'].transform('mean')
df_feature['suiche_nonauto_amount_20_clmnum'] = df_feature.groupby('suiche_nonauto_amount_20')['clmnum'].transform(
    'mean')

df_feature['client_no_20-19'] = df_feature.groupby('client_no')['20-19'].transform('mean')
df_feature['client_no_bi_renewal_year'] = df_feature.groupby('client_no')['bi_renewal_year'].transform('mean')
df_feature['client_no_clmnum'] = df_feature.groupby('client_no')['clmnum'].transform('mean')
df_feature['client_no_car_day'] = df_feature.groupby('client_no')['car_day'].transform('mean')

df_feature['suiche_nonauto_amount_20_dur_personal_insurance_90'] = df_feature.groupby('suiche_nonauto_amount_20')[
    'dur_personal_insurance_90'].transform('mean')
df_feature['suiche_nonauto_amount_20_20-19'] = df_feature.groupby('suiche_nonauto_amount_20')['20-19'].transform('mean')
df_feature['suiche_nonauto_amount_20_clmnum'] = df_feature.groupby('suiche_nonauto_amount_20')['clmnum'].transform(
    'mean')
df_feature['suiche_nonauto_amount_20_car_day'] = df_feature.groupby('suiche_nonauto_amount_20')['car_day'].transform(
    'mean')

df_feature['base_photo_suiche_nonauto_nprem_20'] = df_feature['base_photo'].astype(str) + '_' + df_feature[
    'suiche_nonauto_nprem_20'].astype(str)


# 01分箱
def bin_0_1(x):
    if x > 0:
        return 1
    else:
        return 0


df_feature['nprem_count_01'] = df_feature['suiche_nonauto_nprem_16'].map(bin_0_1) + 10 * df_feature[
    'suiche_nonauto_nprem_17'].map(
    bin_0_1) + 100 * df_feature['suiche_nonauto_nprem_18'].map(bin_0_1) + 1000 * df_feature[
                                   'suiche_nonauto_nprem_19'].map(bin_0_1) + 10000 * df_feature[
                                   'suiche_nonauto_nprem_20'].map(bin_0_1)
df_feature['nprem_count_01_avg'] = df_feature.groupby('nprem_count_01')['suiche_nonauto_nprem_20'].transform('mean')
df_feature['nprem_count_01_suiche_nonauto_nprem_20'] = df_feature['nprem_count_01'].astype(str) + '_' + df_feature[
    'suiche_nonauto_nprem_20'].astype(str)

# 计数
for f in [['dpt'], ['trademark_cn'], ['brand_cn'], ['make_cn'], ['brand_series'], ['series'],
          ['suiche_nonauto_amount_20'], ['p1_census_register'], ['seats'], ['dur_personal_insurance_90_level'],
          ['f2_cust_housing_price_level']]:
    df_temp = df_feature.groupby(f).size().reset_index()
    df_temp.columns = f + ['{}_count'.format('_'.join(f))]
    df_feature = df_feature.merge(df_temp, how='left')

ratio = {}
for i in df_feature.trademark_cn.unique():
    ratio[i] = len(df_feature.loc[(df_feature.trademark_cn == i) & (df_feature.y1_is_purchase == 1)]) / len(
        df_feature.loc[df_feature.trademark_cn == i])

df_feature['ratio_trademark_cn'] = df_feature.trademark_cn.map(ratio)
del ratio

# 简单统计
def stat(df, df_merge, group_by, agg):
    group = df.groupby(group_by).agg(agg)

    columns = []
    for on, methods in agg.items():
        for method in methods:
            columns.append('{}_{}_{}'.format('_'.join(group_by), on, method))
    group.columns = columns
    group.reset_index(inplace=True)
    df_merge = df_merge.merge(group, on=group_by, how='left')

    del (group)
    gc.collect()

    return df_merge


def statis_feat(df_know, df_unknow):
    for f in tqdm(['p1_census_register', 'dpt', 'suiche_nonauto_amount_20', 'si_od', 'client_no', 'nprem_count_01',
                   'p2_client_grade', 'p1_age', 'trademark_cn', 'dur_personal_insurance_90_level']):
        df_unknow = stat(df_know, df_unknow, [f], {
            'y1_is_purchase': ['mean'],
            'active_365': ['mean', 'std'],
            'p1_prior_days_to_insure': ['mean'],
            'newvalue': ['mean', 'std'],
            'suiche_nonauto_nprem_20': ['mean', 'std'],
            'nprem_now': ['mean', 'max'],
            'active_90': ['mean', 'std'],
            'nprem_all_total': ['mean'],
            'nprem_avg': ['mean'],
            'nprem_total': ['mean'],
        })
    return df_unknow

# 5折交叉
df_train = df_feature[~df_feature['y1_is_purchase'].isnull()]
df_train = df_train.reset_index(drop=True)
df_test = df_feature[df_feature['y1_is_purchase'].isnull()]

df_stas_feat = None
kfold = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
for train_index, val_index in kfold.split(df_train, df_train['y1_is_purchase']):
    df_fold_train = df_train.iloc[train_index]
    df_fold_val = df_train.iloc[val_index]

    df_fold_val = statis_feat(df_fold_train, df_fold_val)
    df_stas_feat = pd.concat([df_stas_feat, df_fold_val], axis=0)

    del df_fold_train
    del df_fold_val
    gc.collect()

df_test = statis_feat(df_train, df_test)
df_feature = pd.concat([df_stas_feat, df_test], axis=0)

del df_stas_feat
del df_train
del df_test
gc.collect()

# %%

for f in list(df_feature.select_dtypes('object')):
    if f in ['carid', 'regdate']:
        continue
    le = LabelEncoder()
    df_feature[f] = le.fit_transform(
        df_feature[f].astype('str')).astype('int')

df_train = df_feature[df_feature['y1_is_purchase'].notnull()]
df_test = df_feature[df_feature['y1_is_purchase'].isnull()]

ycol = 'y1_is_purchase'
feature_names = list(
    filter(lambda x: x not in [ycol, 'regdate', 'carid', 'use_type', 'suiche_nonauto_amount_18',
                               'suiche_nonauto_nprem_18'], df_train.columns))
del df_feature, train, test

model = lgb.LGBMClassifier(num_leaves=512,
                           max_depth=12,
                           learning_rate=0.01,
                           n_estimators=10000,
                           subsample=0.8,
                           feature_fraction=0.8,
                           reg_alpha=0.5,
                           reg_lambda=0.5,
                           random_state=seed,
                           n_jobs=4,
                           metric=None, )

oof = []
df_importance_list = []
prediction = df_test[['carid']]
prediction['label'] = 0

kfold = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
for fold_id, (trn_idx, val_idx) in enumerate(kfold.split(
        df_train[feature_names], df_train[ycol])):
    X_train = df_train.iloc[trn_idx][feature_names]
    Y_train = df_train.iloc[trn_idx][ycol]

    X_val = df_train.iloc[val_idx][feature_names]
    Y_val = df_train.iloc[val_idx][ycol]

    print('\nFold_{} Training ================================\n'.format(fold_id + 1))

    lgb_model = model.fit(X_train,
                          Y_train,
                          eval_names=['valid'],
                          eval_set=[(X_val, Y_val)],
                          verbose=500,
                          eval_metric='auc',
                          early_stopping_rounds=500,
                          categorical_feature=[0])

    pred_val = lgb_model.predict_proba(
        X_val, num_iteration=lgb_model.best_iteration_)[:, 1]
    df_oof = df_train.iloc[val_idx][[
        'carid', ycol]].copy()
    df_oof['pred'] = pred_val

    pred_test = lgb_model.predict_proba(df_test[feature_names], num_iteration=lgb_model.best_iteration_)[:, 1]
    prediction['label'] += pred_test / 5
    oof.append(df_oof)
    df_importance = pd.DataFrame({
        'column': feature_names,
        'importance': lgb_model.feature_importances_,
    })
    df_importance_list.append(df_importance)

    del lgb_model, pred_val, X_train, Y_train, X_val, Y_val
    gc.collect()

df_importance = pd.concat(df_importance_list)
df_importance = df_importance.groupby(['column'])['importance'].agg(
    'mean').sort_values(ascending=False).reset_index()

df_oof = pd.concat(oof)
score = roc_auc_score(df_oof['y1_is_purchase'], df_oof['pred'])
df_oof.head(20)
prediction.head()

os.makedirs('sub', exist_ok=True)
prediction.to_csv(f'sub/{score}.csv', index=False)
prediction.to_csv(f'sub/sub.csv', index=False)

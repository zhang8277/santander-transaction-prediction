# load library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from pandas.core.common import SettingWithCopyWarning
import time
import datetime
import os
import warnings
import gc

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore')

train_df = pd.read_csv('input/train.csv')
test_df = pd.read_csv('input/test.csv')

train_length = train_df.shape[0]
# test_ID = test_df['ID_code'].values
train_label = train_df['target'].values
train_df = train_df.drop(['target'], axis=1)
# test_df = test_df.drop(['ID_code'], axis=1)

print('Get real_samples and synthetic_samples index ...', end=' ')
unique_count = np.zeros((test_df.shape[0], 200))
for i in range(200):
    col = 'var_{}'.format(i)
    _, index_, count_ = np.unique(test_df[col].values, return_counts=True, return_index=True)
    unique_count[index_[count_ == 1], i] += 1

# Samples which have unique values are real the others are fake
real_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) > 0)[:, 0]
synthetic_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) == 0)[:, 0]

real_data = pd.concat([train_df, test_df.iloc[real_samples_indexes]], axis=0, sort=False, ignore_index=True)

print('Consturcting count feature with real data ...', end=' ')
for i in range(200):
    feat_name = 'var_{}'.format(i)
    new_feat_name = 'count_' + feat_name
    if real_data[feat_name].nunique() > 5:
        tmp = real_data[[feat_name, 'ID_code']].groupby([feat_name])['ID_code'].agg(['count']).reset_index().rename(columns={'count': new_feat_name})
        train_df = train_df.merge(tmp, on=feat_name, how='left')
        test_df = test_df.merge(tmp, on=feat_name, how='left')
    else:
        continue
print('Done ...', end=' ')

train_df = train_df.drop(['ID_code'], axis=1)
test_df = test_df.drop(['ID_code'], axis=1)
del real_data
gc.collect()

"""
print('Consturcting rolling statistic feature ...', end=' ')
rolling_step = 10
for i in list(np.arange(0,200,rolling_step)):
    feat_name = 'var_{}'.format(i) + '_to_' +'var_{}'.format(i+rolling_step-1)
    data[feat_name + '_mean'] = data.iloc[:,i:(i+rolling_step)].mean(axis=1)
    data[feat_name + '_max'] = data.iloc[:,i:(i+rolling_step)].max(axis=1)
    data[feat_name + '_min'] = data.iloc[:,i:(i+rolling_step)].min(axis=1)
    data[feat_name + '_median'] = data.iloc[:,i:(i+rolling_step)].median(axis=1)
    data[feat_name + '_std'] = data.iloc[:,i:(i+rolling_step)].std(axis=1)
    data[feat_name + '_skew'] = data.iloc[:,i:(i+rolling_step)].skew(axis=1)
    data[feat_name + '_kurtosis'] = data.iloc[:,i:(i+rolling_step)].kurtosis(axis=1)
print('Done ...', end=' ')
"""

"""
rolling_step_2 = 20
for i in list(np.arange(0,200,rolling_step_2)):
    feat_name = 'var_{}'.format(i) + '_to_' +'var_{}'.format(i+rolling_step_2-1)
    data[feat_name + '_mean'] = data.iloc[:,i:(i+rolling_step_2)].mean(axis=1)
    data[feat_name + '_max'] = data.iloc[:,i:(i+rolling_step_2)].max(axis=1)
    data[feat_name + '_min'] = data.iloc[:,i:(i+rolling_step_2)].min(axis=1)
    data[feat_name + '_median'] = data.iloc[:,i:(i+rolling_step_2)].median(axis=1)
    data[feat_name + '_std'] = data.iloc[:,i:(i+rolling_step_2)].std(axis=1)
    data[feat_name + '_skew'] = data.iloc[:,i:(i+rolling_step_2)].skew(axis=1)
    data[feat_name + '_kurtosis'] = data.iloc[:,i:(i+rolling_step_2)].kurtosis(axis=1)
print('Done ...', end=' ')
"""

"""
rolling_step_3 = 40
for i in list(np.arange(0,200,rolling_step_3)):
    feat_name = 'var_{}'.format(i) + '_to_' +'var_{}'.format(i+rolling_step_3-1)
    data[feat_name + '_mean'] = data.iloc[:,i:(i+rolling_step_3)].mean(axis=1)
    data[feat_name + '_max'] = data.iloc[:,i:(i+rolling_step_3)].max(axis=1)
    data[feat_name + '_min'] = data.iloc[:,i:(i+rolling_step_3)].min(axis=1)
    data[feat_name + '_median'] = data.iloc[:,i:(i+rolling_step_3)].median(axis=1)
    data[feat_name + '_std'] = data.iloc[:,i:(i+rolling_step_3)].std(axis=1)
    data[feat_name + '_skew'] = data.iloc[:,i:(i+rolling_step_3)].skew(axis=1)
    data[feat_name + '_kurtosis'] = data.iloc[:,i:(i+rolling_step_3)].kurtosis(axis=1)
print('Done.')
"""

"""
print('Consturcting 2-poly feature ...', end=' ')
for i in range(200):
    col = 'var_{}'.format(i)
    feat_name = col + '_square'
    data[feat_name] = data[col] ** 2
print('Done.')
"""

"""
print('Consturcting roll multiply feature ...', end=' ')
for i in range(199):
    col_1 = 'var_{}'.format(i)
    col_2 = 'var_{}'.format(i+1)
    feat_name = col_1 + '_multiply_' + col_2
    data[feat_name] = data[col_2] * data[col_1]
print('Done.')
"""

#train_df = data[:train_length]
#test_df = data[train_length:]
train_df['target'] = train_label
# del data
del train_label
gc.collect()


def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance",
                                                                                                   ascending=False)[
           :200].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances.png')


def kfold_lightgbm(train_df, test_df, num_folds, stratified=False, debug=False):
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))

    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['target']]

    # k-fold
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['target'])):
        print('Fold {}'.format(n_fold + 1))
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx]
        
        # set data structure
        lgb_train = lgb.Dataset(train_x,
                                label=train_y,
                                free_raw_data=False)
        lgb_test = lgb.Dataset(valid_x,
                               label=valid_y,
                               free_raw_data=False)

        # params optimized by optuna
        params = {'num_leaves': 6,
                  'min_data_in_leaf': 42,
                  'objective': 'binary',
                  'max_depth': -1,
                  'learning_rate': 0.01,
                  'boosting': 'gbdt',
                  'bagging_freq': 5,
                  'bagging_fraction': 0.8,
                  'feature_fraction': 0.8201,
                  'bagging_seed': 11,
                  'reg_alpha': 1.728910519108444,
                  'reg_lambda': 4.9847051755586085,
                  'random_state': 42,
                  'metric': 'auc',
                  'verbosity': -1,
                  'min_gain_to_split': 0.01077313523861969,
                  'min_child_weight': 19.428902804238373,
                  'num_threads': -1,
                  # 'is_unbalance': True,
                  # 'seed':int(3**n_fold),
                  # 'bagging_seed':int(3**n_fold),
                  }

        reg = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_test],
            valid_names=['train', 'test'],
            num_boost_round=20000,
            early_stopping_rounds=200,
            verbose_eval=300
        )

        oof_preds[valid_idx] = reg.predict(valid_x, num_iteration=reg.best_iteration)
        #sub_preds += reg.predict(test_df[feats], num_iteration=reg.best_iteration) / folds.n_splits
        sub_preds += reg.predict(test_df[feats], num_iteration=reg.best_iteration)
        
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = np.log1p(
            reg.feature_importance(importance_type='gain', iteration=reg.best_iteration))
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        del reg, train_x, train_y, valid_x, valid_y
        gc.collect()

    print("CV score: {:<8.5f}".format(roc_auc_score(train_df['target'].values, oof_preds)))
    # display importances
    display_importances(feature_importance_df)

    # if not debug:
    # save submission file
    sub = pd.read_csv('input/sample_submission.csv', usecols=['ID_code'])
    sub['target'] = sub_preds
    sub.to_csv('submission1.csv', index=False)


if __name__ == "__main__":
    kfold_lightgbm(train_df, test_df, num_folds=6, stratified=True, debug=False)
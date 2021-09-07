import gc
import os
from pathlib import Path
import random
import sys
from os.path import join as pjoin

from tqdm import tqdm_notebook as tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

from IPython.core.display import display, HTML

# --- plotly ---
from plotly import tools, subplots
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff

# --- models ---
from sklearn import preprocessing
from sklearn.model_selection import KFold, train_test_split
import lightgbm as lgb
import xgboost as xgb
import catboost as cb





RAW_DATA_DIR = '/kaggle/input/ashrae-energy-prediction/'

weather_dtypes = {
    'site_id': np.uint8,
    'air_temperature': np.float32,
    'cloud_coverage': np.float32,
    'dew_temperature': np.float32,
    'precip_depth_1_hr': np.float32,
    'sea_level_pressure': np.float32,
    'wind_direction': np.float32,
    'wind_speed': np.float32,
}

weather_train = pd.read_csv(pjoin(RAW_DATA_DIR, 'weather_train.csv'),dtype=weather_dtypes,
    parse_dates=['timestamp'])
weather_test = pd.read_csv(pjoin(RAW_DATA_DIR, 'weather_test.csv'),dtype=weather_dtypes,
    parse_dates=['timestamp'])

weather = pd.concat([weather_train,weather_test],ignore_index=True)
del weather_train, weather_test
weather_key = ['site_id', 'timestamp']




temp_skeleton = weather[weather_key + ['air_temperature']].drop_duplicates(subset=weather_key).sort_values(by=weather_key).copy()
data_to_plot = temp_skeleton.copy()
data_to_plot["hour"] = data_to_plot["timestamp"].dt.hour
count = 1
plt.figure(figsize=(25, 15))
for site_id, data_by_site in data_to_plot.groupby('site_id'):
    by_site_by_hour = data_by_site.groupby('hour').mean()
    ax = plt.subplot(4, 4, count)
    plt.plot(by_site_by_hour.index,by_site_by_hour['air_temperature'],'xb-')
    ax.set_title('site: '+str(site_id))
    count += 1
plt.tight_layout()
plt.show()




# calculate ranks of hourly temperatures within date/site_id chunks
temp_skeleton['temp_rank'] = temp_skeleton.groupby(['site_id', temp_skeleton.timestamp.dt.date])['air_temperature'].rank('average')

# create a dataframe of site_ids (0-16) x mean hour rank of temperature within day (0-23)
df_2d = temp_skeleton.groupby(['site_id', temp_skeleton.timestamp.dt.hour])['temp_rank'].mean().unstack(level=1)

# Subtract the columnID of temperature peak by 14, getting the timestamp alignment gap.
site_ids_offsets = pd.Series(df_2d.values.argmax(axis=1) - 14)
site_ids_offsets.index.name = 'site_id'

def timestamp_align(df):
    df['offset'] = df.site_id.map(site_ids_offsets)
    df['timestamp_aligned'] = (df.timestamp - pd.to_timedelta(df.offset, unit='H'))
    df['timestamp'] = df['timestamp_aligned']
    del df['timestamp_aligned']
    return df


# Original code from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage by @gemartin
# Modified to support timestamp type, categorical type
# Modified to add option to use float16 or not. feather format does not support float16.
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype


def reduce_mem_usage(df, use_float16=False):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            # skip datetime type or categorical type
            continue
        col_type = df[col].dtype

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
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

# !ls.. / input


# %%time
root = Path('../input/ashrae-feather-format-for-fast-loading')

train_df = pd.read_feather(root/'train.feather')
weather_train_df = pd.read_feather(root/'weather_train.feather')
building_meta_df = pd.read_feather(root/'building_metadata.feather')


def preprocess(df):
    df["hour"] = df["timestamp"].dt.hour
    df["weekend"] = df["timestamp"].dt.weekday
    df["month"] = df["timestamp"].dt.month
    df["dayofweek"] = df["timestamp"].dt.dayofweek


def add_lag_feature(weather_df, window=3):
    group_df = weather_df.groupby('site_id')
    cols = ['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
            'wind_direction', 'wind_speed']
    rolled = group_df[cols].rolling(window=window, min_periods=0)
    lag_mean = rolled.mean().reset_index().astype(np.float16)
    lag_max = rolled.max().reset_index().astype(np.float16)
    lag_min = rolled.min().reset_index().astype(np.float16)
    lag_std = rolled.std().reset_index().astype(np.float16)
    for col in cols:
        weather_df[f'{col}_mean_lag{window}'] = lag_mean[col]
        weather_df[f'{col}_max_lag{window}'] = lag_max[col]
        weather_df[f'{col}_min_lag{window}'] = lag_min[col]
        weather_df[f'{col}_std_lag{window}'] = lag_std[col]


train_df['date'] = train_df['timestamp'].dt.date
train_df['meter_reading_log1p'] = np.log1p(train_df['meter_reading'])
building_meta_df[building_meta_df.site_id == 0]
train_df = train_df.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')
debug = False
preprocess(train_df)

# https://www.kaggle.com/ryches/simple-lgbm-solution
df_group = train_df.groupby('building_id')['meter_reading_log1p']
building_mean = df_group.mean().astype(np.float16)
building_median = df_group.median().astype(np.float16)
building_min = df_group.min().astype(np.float16)
building_max = df_group.max().astype(np.float16)
building_std = df_group.std().astype(np.float16)

train_df['building_mean'] = train_df['building_id'].map(building_mean)
train_df['building_median'] = train_df['building_id'].map(building_median)
train_df['building_min'] = train_df['building_id'].map(building_min)
train_df['building_max'] = train_df['building_id'].map(building_max)
train_df['building_std'] = train_df['building_id'].map(building_std)

weather_train_df = timestamp_align(weather_train_df)
weather_train_df = weather_train_df.groupby('site_id').apply(lambda group: group.interpolate(limit_direction='both'))

add_lag_feature(weather_train_df, window=3)
add_lag_feature(weather_train_df, window=72)

primary_use_list = building_meta_df['primary_use'].unique()
primary_use_dict = {key: value for value, key in enumerate(primary_use_list)}
print('primary_use_dict: ', primary_use_dict)
building_meta_df['primary_use'] = building_meta_df['primary_use'].map(primary_use_dict)

gc.collect()

reduce_mem_usage(train_df, use_float16=True)
reduce_mem_usage(building_meta_df, use_float16=True)
reduce_mem_usage(weather_train_df, use_float16=True)



category_cols = ['building_id', 'site_id', 'primary_use']  # , 'meter'
feature_cols = ['square_feet', 'year_built'] + [
    'hour', 'weekend', # 'month' , 'dayofweek'
    'building_median'] + [
    'air_temperature', 'cloud_coverage',
    'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
    'wind_direction', 'wind_speed', 'air_temperature_mean_lag72',
    'air_temperature_max_lag72', 'air_temperature_min_lag72',
    'air_temperature_std_lag72', 'cloud_coverage_mean_lag72',
    'dew_temperature_mean_lag72', 'precip_depth_1_hr_mean_lag72',
    'sea_level_pressure_mean_lag72', 'wind_direction_mean_lag72',
    'wind_speed_mean_lag72', 'air_temperature_mean_lag3',
    'air_temperature_max_lag3',
    'air_temperature_min_lag3', 'cloud_coverage_mean_lag3',
    'dew_temperature_mean_lag3',
    'precip_depth_1_hr_mean_lag3', 'sea_level_pressure_mean_lag3',
    'wind_direction_mean_lag3', 'wind_speed_mean_lag3']


def create_X_y(train_df, target_meter):
    target_train_df = train_df[train_df['meter'] == target_meter]
    target_train_df = target_train_df.merge(building_meta_df, on='building_id', how='left')
    target_train_df = target_train_df.merge(weather_train_df, on=['site_id', 'timestamp'], how='left')
    X_train = target_train_df[feature_cols + category_cols]
    y_train = target_train_df['meter_reading_log1p'].values

    del target_train_df
    return X_train, y_train



metric = 'l2'
early_stop = 50
verbose_eval = 50
iteration=1500
lr=0.001
print('training CatBoostRegressor:')
cbr = cb.CatBoostRegressor(iterations=iteration, learning_rate=lr,
                           num_leaves=31,
                           depth=256, loss_function='MAE',
                           random_seed=42, eval_metric=metric,
                           bagging_temperature=0.9, boosting_type='gbdt',
                           sampling_frequency=5, objective='regression',
                           cat_features=cat_features, task_type='GPU',
                           devices='0:1', n_estimators=5000)

# for loading the model
cbr = cbr.load_model('')

for i in range(10):
    model = cbr.fit(X=X_train, y=y_train,  cat_features=cat_features,
                    verbose=1, plot=True,
                    verbose_eval=verbose_eval,
                    early_stopping_rounds=early_stop,
                    init_model=None, eval_set=())

    # predictions
    y_pred_valid = model.predict(X_valid)

    print('best_score', model.best_score_)
    log = {'train/mae': model.best_score_['training']['l2'],
           'valid/mae': model.best_score_['valid_1']['l2']}
    model.save_model()
    model.predict(X_valid, prediction_type=)



target_meter = 0

X_train, y_train = create_X_y(train_df, target_meter=target_meter)
# y_valid_pred_total = np.zeros(X_train.shape[0])
gc.collect()
print('target_meter', target_meter, X_train.shape)

cat_features = [X_train.columns.get_loc(cat_col) for cat_col in category_cols]
print('cat_features', cat_features)

models0 = []
x_train, x_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=42, shuffle=True)

model, y_pred_valid, log = fit_lgbm(train_data, valid_data, cat_features=category_cols,
                                    num_rounds=15000, lr=0.3, bf=0.9)
os.chdir('/kaggle/working')
model.save_model('ASHRAE_LGBM-01-{}.txt'.format(target_meter))
print('===========================================model saved: ASHRAE_LGBM-01-{}.txt'.format(target_meter))
gc.collect()


sns.distplot(y_train)
del X_train, y_train
gc.collect()




from IPython.display import FileLink
import glob, os

os.chdir('/kaggle/working')
# files = [file for f in glob.glob("*.txt")]
# print(files)


FileLink('ASHRAE_LGBM-00-0.txt')



target_meter = 1

X_train, y_train = create_X_y(train_df, target_meter=target_meter)
# y_valid_pred_total = np.zeros(X_train.shape[0])
gc.collect()
print('target_meter', target_meter, X_train.shape)

cat_features = [X_train.columns.get_loc(cat_col) for cat_col in category_cols]
print('cat_features', cat_features)

models0 = []
x_tr, x_ts, y_tr, y_ts = train_test_split(X_train, y_train, test_size=0.3, random_state=87, shuffle=True)
train_data = [x_tr, y_tr ]
valid_data = [x_ts, y_ts]
model, y_pred_valid, log = fit_lgbm(train_data, valid_data, cat_features=category_cols,
                                    num_rounds=5000, lr=0.01, bf=0.7)
os.chdir('/kaggle/working')
model.save_model('ASHRAE_LGBM-01-{}.txt'.format(target_meter))
print('===========================================model saved: ASHRAE_LGBM-01-{}.txt'.format(target_meter))
gc.collect()


sns.distplot(y_train)
del X_train, y_train



target_meter = 2

X_train, y_train = create_X_y(train_df, target_meter=target_meter)
# y_valid_pred_total = np.zeros(X_train.shape[0])
gc.collect()
print('target_meter', target_meter, X_train.shape)

cat_features = [X_train.columns.get_loc(cat_col) for cat_col in category_cols]
print('cat_features', cat_features)

models0 = []
x_tr, x_ts, y_tr, y_ts = train_test_split(X_train, y_train, test_size=0.25, random_state=59, shuffle=True)
train_data = [x_tr, y_tr ]
valid_data = [x_ts, y_ts]
model, y_pred_valid, log = fit_lgbm(train_data, valid_data, cat_features=category_cols,
                                    num_rounds=5000, lr=0.03, bf=0.8)
os.chdir('/kaggle/working')
model.save_model('ASHRAE_LGBM-01-{}.txt'.format(target_meter))
print('===========================================model saved: ASHRAE_LGBM-01-{}.txt'.format(target_meter))
gc.collect()


sns.distplot(y_train)
del X_train, y_train






target_meter = 3

X_train, y_train = create_X_y(train_df, target_meter=target_meter)
# y_valid_pred_total = np.zeros(X_train.shape[0])
gc.collect()
print('target_meter', target_meter, X_train.shape)

cat_features = [X_train.columns.get_loc(cat_col) for cat_col in category_cols]
print('cat_features', cat_features)

models0 = []
x_tr, x_ts, y_tr, y_ts = train_test_split(X_train, y_train, test_size=0.15, random_state=199, shuffle=True)
train_data = [x_tr, y_tr ]
valid_data = [x_ts, y_ts]
model, y_pred_valid, log = fit_lgbm(train_data, valid_data, cat_features=category_cols,
                                    num_rounds=5000, lr=0.05, bf=0.9)
os.chdir('/kaggle/working')
model.save_model('ASHRAE_LGBM-01-{}.txt'.format(target_meter))
print('===========================================model saved: ASHRAE_LGBM-01-{}.txt'.format(target_meter))
gc.collect()


sns.distplot(y_train)
del X_train, y_train












print('loading...')
test_df = pd.read_feather(root/'test.feather')
weather_test_df = pd.read_feather(root/'weather_test.feather')

print('preprocessing building...')
test_df['date'] = test_df['timestamp'].dt.date
preprocess(test_df)
test_df['building_mean'] = test_df['building_id'].map(building_mean)
test_df['building_median'] = test_df['building_id'].map(building_median)
test_df['building_min'] = test_df['building_id'].map(building_min)
test_df['building_max'] = test_df['building_id'].map(building_max)
test_df['building_std'] = test_df['building_id'].map(building_std)

print('preprocessing weather...')
weather_test_df = timestamp_align(weather_test_df)
weather_test_df = weather_test_df.groupby('site_id').apply(lambda group: group.interpolate(limit_direction='both'))
weather_test_df.groupby('site_id').apply(lambda group: group.isna().sum())

add_lag_feature(weather_test_df, window=3)
add_lag_feature(weather_test_df, window=72)

print('reduce mem usage...')
reduce_mem_usage(test_df, use_float16=True)
reduce_mem_usage(weather_test_df, use_float16=True)

gc.collect()






sample_submission = pd.read_feather(os.path.join(root, 'sample_submission.feather'))
reduce_mem_usage(sample_submission)




def create_X(test_df, target_meter):
    target_test_df = test_df[test_df['meter'] == target_meter]
    target_test_df = target_test_df.merge(building_meta_df, on='building_id', how='left')
    target_test_df = target_test_df.merge(weather_test_df, on=['site_id', 'timestamp'], how='left')
    X_test = target_test_df[feature_cols + category_cols]
    return X_test



def pred(X_test, models, batch_size=1000000):
    iterations = (X_test.shape[0] + batch_size -1) // batch_size
    print('iterations', iterations)

    y_test_pred_total = np.zeros(X_test.shape[0])
    for i, model in enumerate(models):
        print(f'predicting {i}-th model')
        for k in tqdm(range(iterations)):
            y_pred_test = model.predict(X_test[k*batch_size:(k+1)*batch_size], num_iteration=model.best_iteration)
            y_test_pred_total[k*batch_size:(k+1)*batch_size] += y_pred_test

    y_test_pred_total /= len(models)
    return y_test_pred_total


models = [models0, modesl1, models2, models3]
y_test = []
for i, model in enumerate(models):
    # % % time
    X_test = create_X(test_df, target_meter=i)
    gc.collect()
    y_test.append(pred(X_test, model))
    sns.distplot(y_test0)

del X_test
gc.collect()




sample_submission.loc[test_df['meter'] == 0, 'meter_reading'] = np.expm1(y_test[0])
sample_submission.loc[test_df['meter'] == 1, 'meter_reading'] = np.expm1(y_test[1])
sample_submission.loc[test_df['meter'] == 2, 'meter_reading'] = np.expm1(y_test[2])
sample_submission.loc[test_df['meter'] == 3, 'meter_reading'] = np.expm1(y_test[3])




sample_submission.to_csv('submission.csv', index=False, float_format='%.4f')
sample_submission.head()



from tensorflow.python.keras.layers import PReLU
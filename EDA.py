import pandas as pd
import datetime
import math
import numpy as np
import sys

from sklearn import model_selection
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold

from sklearn.neural_network import MLPRegressor

from pandas.plotting import scatter_matrix

# --------------------------
# ---- Helper functions ----
# --------------------------

def convert_to_date(time_string):
    try:
        format_str = '%d-%m-%y'
        return datetime.datetime.strptime(time_string, format_str)
    except ValueError:
        format_str = '%Y-%m-%d'
        return datetime.datetime.strptime(time_string, format_str)

def convert_timestamp_to_int(df, column_name='travel_date'):
    #df['year'] = df['travel_date'].apply(lambda x: (convert_to_date(x)).year)
    df['month'] = df['travel_date'].apply(lambda x: (convert_to_date(x)).month)
    df['day'] = df[column_name].apply(lambda x: (convert_to_date(x)).day)
    df.drop([column_name], axis=1, inplace=True)
    return df

def hot_encode(df):
    obj_df = df.select_dtypes(include=['object'])
    return pd.get_dummies(df, columns=obj_df.columns).values

def time_split(row):
    a,b = row.split(':')
    a = int(a)
    b = int(b)
    return a,b

def processing_pipeline(data):
    data = convert_timestamp_to_int(data)
    data['hour'], data['minutes'] = zip(*data['travel_time'].apply(time_split))
    data.drop(['travel_time'], axis=1, inplace=True)

    trips = data.groupby(['travel_from', 'month']).size().reset_index(name='trips_scheduled')
    trips = trips.sort_values(['trips_scheduled'])
    data = data.merge(trips, how='outer', left_on=['travel_from', 'month'], right_on=['travel_from', 'month'])

    data = data.replace(['Kendu Bay', 'Oyugis', 'Keumbu', 'Sori'], 'other')
    encoding = data.groupby('travel_from').size()
    encoding = encoding/len(data)
    data['travel_encoding'] = data.travel_from.map(encoding)
    data.drop('travel_from', axis=1, inplace=True)
    
    data['car_type_is_bus'] = np.where(data['car_type'] == 'Bus', 1, 0)
    data.drop('car_type', axis=1, inplace=True)

    #data = pd.concat([data, pd.get_dummies(data['travel_from'], prefix='travel_from')], axis=1).drop(['travel_from'], axis=1)
    return data

def process_uber_data(df):
    df['day'] = pd.DatetimeIndex(df['Date']).day

    df.drop(
        ['Date', 'Origin Movement ID', 'Destination Movement ID', 'Origin Display Name', 
        'Destination Display Name'], axis=1, inplace=True)

    df = df.groupby('day').mean()
    return df


# -----------------------------
# ---- data pre processing ----
# -----------------------------

# import and handle train data
url = 'C:\\Users\\brendon.pitcher\\Documents\\Brendon\\Dev\\PlayTime\\Zindi\\TrafficJam\\train_revised.csv'
df = pd.read_csv(url)
columns = ['ride_id', 'travel_date', 'travel_time', 'travel_from', 'car_type', 'max_capacity']
df = df.groupby(columns).size().reset_index(name='counts')
print(df.sort_values('travel_date', ascending=False).head(5))

sys.exit()


columns = ['ride_id', 'travel_date', 'travel_time', 'travel_from', 'car_type', 'max_capacity']
data = df.groupby(columns).size().reset_index(name='counts')
train = processing_pipeline(data)


import matplotlib.pyplot as plt

scatter_matrix(train)
plt.show()

sys.exit()

# import and handle test data
url = 'C:\\Users\\brendon.pitcher\\Documents\\Brendon\\Dev\\PlayTime\\Zindi\\TrafficJam\\test_questions.csv'
test = pd.read_csv(url)
X_test = processing_pipeline(test)
X_test.drop(['travel_to'],axis=1, inplace=True)


# add columns to test set if they dont exist after feature encoding
if ((len(train.columns)) > (len(X_test.columns) + 1)):
    missing_columns = list(set(train.columns) - set(X_test.columns) - set(['counts']))

    for miss in missing_columns:
        X_test[miss] = 0

# reorder columns
X_test = X_test[train.drop('counts', axis=1).columns]

# create X/Y training and validation sets
X_train = train.drop(['counts'], axis=1)
y_train = train['counts']
#X_train, X_test, y_train, y_test = model_selection.train_test_split(X_train, y_train, test_size=0.25, shuffle=False)


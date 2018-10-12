
import pandas as pd
import datetime
import math
import numpy as np

from sklearn import model_selection
from sklearn.metrics import mean_absolute_error

#from xgboost.sklearn import XGBRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor, BaggingRegressor

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
    #df['month'] = df['travel_date'].apply(lambda x: (convert_to_date(x)).month)
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
    data.drop(['travel_time', 'minutes'], axis=1, inplace=True)

    data = data.replace(['Kendu Bay', 'Oyugis', 'Keumbu', 'Sori'], 'other')
    data['car_type_is_bus'] = np.where(data['car_type'] == 'Bus', 1, 0)
    data.drop('car_type', axis=1, inplace=True)

    return pd.concat([data, pd.get_dummies(data['travel_from'], prefix='travel_from')], axis=1).drop(['travel_from'], axis=1)


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
data = df.groupby(columns).size().reset_index(name='counts')
train = processing_pipeline(data)

# import and handle test data
url = 'C:\\Users\\brendon.pitcher\\Documents\\Brendon\\Dev\\PlayTime\\Zindi\\TrafficJam\\test_questions.csv'
test = pd.read_csv(url)
X_test = processing_pipeline(test)
X_test.drop(['travel_to'],axis=1, inplace=True)

# import and handle uber travel data
url = 'C:\\Users\\brendon.pitcher\\Documents\\Brendon\\Dev\\PlayTime\\Zindi\\TrafficJam\\westlands_travel_times_daily.csv'
wtt = process_uber_data(pd.read_csv(url))
train = train.merge(wtt, how='left', left_on='day', right_on='day')

# AM - 7 to 10
# midday - 10 to 4
# PM - 4 to 7
# evening - 7 to 12
# morning - 12 to 7

def drop_uber_columns(df):
    return df.drop([       'AM Mean Travel Time (Seconds)',
       'AM Range - Lower Bound Travel Time (Seconds)',
       'AM Range - Upper Bound Travel Time (Seconds)',
       'PM Mean Travel Time (Seconds)',
       'PM Range - Lower Bound Travel Time (Seconds)',
       'PM Range - Upper Bound Travel Time (Seconds)',
       'Midday Mean Travel Time (Seconds)',
       'Midday Range - Lower Bound Travel Time (Seconds)',
       'Midday Range - Upper Bound Travel Time (Seconds)',
       'Evening Mean Travel Time (Seconds)',
       'Evening Range - Lower Bound Travel Time (Seconds)',
       'Evening Range - Upper Bound Travel Time (Seconds)',
       'Early Morning Mean Travel Time (Seconds)',
       'Early Morning Range - Lower Bound Travel Time (Seconds)',
       'Early Morning Range - Upper Bound Travel Time (Seconds)',], axis=1)


def concat_travel_columns(x):
    if x['hour'] >= 7 and x['hour'] < 10:
        return x['AM Mean Travel Time (Seconds)'], x['AM Range - Lower Bound Travel Time (Seconds)'], x['AM Mean Travel Time (Seconds)']
    if x['hour'] >= 10 and x['hour'] < 14:
        return x['Midday Mean Travel Time (Seconds)'], x['Midday Range - Lower Bound Travel Time (Seconds)'], x['Midday Range - Upper Bound Travel Time (Seconds)']
    if x['hour'] >= 14 and x['hour'] < 19:
        return x['PM Mean Travel Time (Seconds)'], x['PM Range - Lower Bound Travel Time (Seconds)'], x['PM Range - Upper Bound Travel Time (Seconds)']
    if x['hour'] >= 19 and x['hour'] <= 23:
        return x['Evening Mean Travel Time (Seconds)'], x['Evening Range - Lower Bound Travel Time (Seconds)'], x['Evening Range - Upper Bound Travel Time (Seconds)']
    if x['hour'] >= 1 and x['hour'] < 7:
        return x['Early Morning Mean Travel Time (Seconds)'], x['Early Morning Range - Lower Bound Travel Time (Seconds)'], x['Early Morning Range - Upper Bound Travel Time (Seconds)']

train['hour_mean_time'], train['hour_min_time'], train['hour_max_time'] = zip(*train.apply(concat_travel_columns, axis=1))
train = drop_uber_columns(train)

print(train.columns)

#print(train['hour'].head(100))

import sys
sys.exit()

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
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_train, y_train, test_size=0.25, random_state=1)


# -----------------------
# --- training model ----
# -----------------------

models=[ 
        ######## First level ########
        [
            #BaggingRegressor(RandomForestRegressor(n_estimators=150, criterion="mae", max_depth=50, max_features=0.7), max_samples=0.5, max_features=0.5),
            #SVR(max_iter=-1, kernel='rbf'),
            #KNeighborsRegressor(n_neighbors=2),
            #MLPRegressor(hidden_layer_sizes=(100, 100), learning_rate='adaptive', learning_rate_init=0.01, alpha=0.000001),
            #XGBRegressor(n_estimators=100, criterion="mae", max_depth=25, subsample=0.5, learning_rate=0.1, colsample_bytree=0.9),
            #ExtraTreesRegressor(n_estimators=120, criterion="mae", max_depth=20, max_features=0.5, random_state=1),
            GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=50, max_features=0.5, random_state=1),
            #RandomForestRegressor(n_estimators=150, criterion="mae", max_depth=50, max_features=0.7, random_state=1, oob_score=True),
        ],
        ######## Second level ########
        #[
            #XGBRegressor(n_estimators=100, criterion="mae", max_depth=5, max_features=0.5, random_state=1),
        #]
        # [
        #     XGBRegressor(n_estimators=100, criterion="mae", max_depth=3, max_features=0.5, random_state=1),
        # ]
    ]

from pystacknet.pystacknet import StackNetRegressor

model = StackNetRegressor(models, metric="mae", folds=4, restacking=False, use_retraining=False, random_state=12345, verbose=1)

model.fit(X_train, y_train)
print(model.score(X_train.values, y_train))

predictions = model.predict(X_test.as_matrix())
print(mean_absolute_error(y_test, predictions))


'''
# save results
url = 'C:\\Users\\brendon.pitcher\\Documents\\Brendon\\Dev\\PlayTime\\Zindi\\TrafficJam\\results.csv'

predictions[predictions < 0] = 0
X_test['numer_of_ticket'] = predictions
X_test[['ride_id', 'numer_of_ticket']].to_csv(url, index=False)
'''
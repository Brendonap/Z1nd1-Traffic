import pandas as pd
import datetime
import math
import numpy as np

from sklearn import model_selection
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold


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

def convert_timestamp_to_int(df):
    df['year'] = df['travel_date'].apply(lambda x: (convert_to_date(x)).year)
    df['month'] = df['travel_date'].apply(lambda x: (convert_to_date(x)).month)
    df['day'] = df['travel_date'].apply(lambda x: (convert_to_date(x)).day)
    df.drop(['travel_date'], axis=1, inplace=True)
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
    data.drop('travel_time', axis=1, inplace=True)

    data = pd.concat([data, pd.get_dummies(data['car_type'], prefix='car_type')], axis=1).drop(['car_type'], axis=1)
    return pd.concat([data, pd.get_dummies(data['travel_from'], prefix='travel_from')], axis=1).drop(['travel_from'], axis=1)


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
X_test = processing_pipeline(pd.read_csv(url))
X_test.drop('travel_to',axis=1, inplace=True )

# add columns to test set if they dont exist after feature encoding
if ((len(train.columns)) > (len(X_test.columns) + 1)):
    missing_columns = list(set(train.columns) - set(X_test.columns) - set(['counts']))

    for miss in missing_columns:
        X_test[miss] = 0

# reorder columns
X_test = X_test[train.drop('counts', axis=1).columns]

# create X/Y training and validation sets
X_train = train.drop('counts', axis=1)
y_train = train['counts']
#X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size=0.2, random_state=1)

from keras.models import Sequential
from keras.layers import Dense
from keras import metrics
import numpy

# create model
model = Sequential()
model.add(Dense(12, input_dim=26, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='linear'))
# Compile model
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=[metrics.mae])
# Fit the model
model.fit(X_train, y_train, epochs=150, batch_size=10,  verbose=2)

score = model.evaluate(X_train, y_train)
print(score)
# calculate predictions
predictions = model.predict(X_train)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)
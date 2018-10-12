
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


import pandas as pd
import datetime
import time
import numpy as np
import math


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


url = 'C:\\Users\\brendon.pitcher\\Documents\\Brendon\\Dev\\PlayTime\\Zindi\\TrafficJam\\train_revised.csv'
df = pd.read_csv(url)
columns = ['ride_id', 'travel_date', 'travel_time', 'travel_from', 'car_type', 'max_capacity']
data = df.groupby(columns).size().reset_index(name='counts')
train = processing_pipeline(data)

# create X/Y training and validation sets
X = train.drop('counts', axis=1)
y = train['counts']

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.5, random_state=1)

#X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=1)

# ----------------
# ---- Models ----
# ----------------

rf_params = {
    'n_jobs': -1,
    'n_estimators': 200,
     'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 15,
    'min_samples_leaf': 8,
    'max_features': 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':50,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 50,
    'learning_rate' : 0.75
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 50,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

base_xgb_params = {
    'objective':'reg:linear',
    'learning_rate': .03, #so called `eta` value
    'max_depth': 12,
    'min_child_weight': 8,
    #'gamma': [0.01, 0.1, 0.3],
    'silent': 1,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'n_estimators': 250
}


model1 = RandomForestRegressor(**rf_params)
model1.fit(X_train, y_train)

model2 = AdaBoostRegressor(**ada_params) 
model2.fit(X_train, y_train) 

model3 = GradientBoostingRegressor(**gb_params)
model3.fit(X_train, y_train)

model4 = ExtraTreesRegressor(**et_params)
model4.fit(X_train, y_train)

model5 = XGBRegressor(**base_xgb_params)
model5.fit(X_train, y_train)

# ----------------------------
# ---- Build second layer ----
# ----------------------------

df_val = pd.DataFrame(
    {
        'model1': model1.predict(X_train),
        'model2': model2.predict(X_train),
        'model3': model3.predict(X_train),
        'model4': model4.predict(X_train),
        'model5': model5.predict(X_train),
    })

df_test = pd.DataFrame(
    {
        'model1': model1.predict(X_test),
        'model2': model2.predict(X_test),
        'model3': model3.predict(X_test),
        'model4': model4.predict(X_test),
        'model5': model5.predict(X_test),
    })


xgb1 = XGBRegressor()
xgb_params = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['reg:linear'],
              'learning_rate': [0.01, .03, 0.05], #so called `eta` value
              'max_depth': [5, 7, 12, 15, 20],
              'min_child_weight': [2, 4, 8, 12],
              #'gamma': [0.01, 0.1, 0.3],
              'silent': [1],
              'subsample': [0.5, 0.7, 0.9],
              'colsample_bytree': [0.5, 0.7, 0.9],
              'n_estimators': [200, 250, 300]}
# explained_variance
# neg_mean_squared_error

xgb_grid = RandomizedSearchCV(xgb1,
                        xgb_params,
                        cv = 5,
                        scoring='neg_mean_absolute_error', 
                        n_jobs = 1,
                        verbose=True)


xgb_grid.fit(df_val, y_train)
test_predict = xgb_grid.predict(df_test)


print(mean_absolute_error(y_test, test_predict))

'''
# save results
url = 'C:\\Users\\brendon.pitcher\\Documents\\Brendon\\Dev\\PlayTime\\Zindi\\TrafficJam\\results.csv'

test_predict[test_predict < 0] = 0
X_test['numer_of_ticket'] = test_predict
X_test[['ride_id', 'numer_of_ticket']].to_csv(url, index=False)
'''


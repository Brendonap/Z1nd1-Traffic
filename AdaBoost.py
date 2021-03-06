import pandas as pd
import datetime
import math
import numpy as np

from sklearn import model_selection
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, LinearRegression

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold, cross_val_score

from sklearn.neural_network import MLPRegressor
from pystacknet.pystacknet import StackNetRegressor
import sys


# --------------------------
# ---- Helper functions ----
# --------------------------

url = 'C:\\Users\\brendon.pitcher\\Documents\\Brendon\\Dev\\PlayTime\\Zindi\\TrafficJam\\train_revised.csv'
df = pd.read_csv(url)
columns = ['ride_id', 'travel_date', 'travel_time', 'travel_from', 'car_type', 'max_capacity']
df_train_set = df.groupby(columns).size().reset_index(name='number_of_tickets')
df_train_set = df_train_set.sort_values('travel_date', ascending=False)

df_train_set.drop(['ride_id'], axis=1, inplace=True) #ride_id is unnecessary in training set

df_train_set["travel_date"] = pd.to_datetime(df_train_set["travel_date"],infer_datetime_format=True)
df_train_set["travel_date"] = df_train_set["travel_date"].dt.dayofweek #change the full date to day of week

df_train_set["car_type"] = pd.Categorical(df_train_set["car_type"])
car_type_categories = df_train_set.car_type.cat.categories
df_train_set["car_type"] = df_train_set.car_type.cat.codes

df_train_set = df_train_set.replace(['Kendu Bay', 'Oyugis', 'Keumbu'], 'other')
df_train_set = pd.concat([df_train_set, pd.get_dummies(df_train_set['travel_from'], prefix='travel_from')], axis=1).drop(['travel_from'], axis=1)

#df_train_set = df_train_set.replace(['Kendu Bay', 'Oyugis', 'Keumbu'], 'other')
#df_train_set["travel_from"] = pd.Categorical(df_train_set["travel_from"])
#travel_from_categories = df_train_set.travel_from.cat.categories
#df_train_set["travel_from"] = df_train_set.travel_from.cat.codes

#express travel time in minutes
df_train_set["travel_time"] = df_train_set["travel_time"].str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))

# ------ model
X = df_train_set.drop(["number_of_tickets"], axis=1)
y = df_train_set.number_of_tickets

#X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, shuffle=True)

from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
#scaler.fit(X)
#X = scaler.transform(X)

br = AdaBoostRegressor(ExtraTreesRegressor(),n_estimators=8, learning_rate= 0.05)
params = {
    'base_estimator': [ExtraTreesRegressor(), Ridge(), KernelRidge()],
    'n_estimators': [2, 5, 8, 10, 20],
    'learning_rate': [0.05, 0.2, 0.5],
}

model = RandomizedSearchCV(br, params, scoring='neg_mean_absolute_error', cv=5, verbose=1)

model.fit(X, y)

print(model.best_score_)
print(model.best_params_)

#{'n_estimators': 5, 'learning_rate': 0.05}

#scores = cross_val_score(br, X, y, scoring='neg_mean_absolute_error', cv=5)
#print(sum(scores) / len(scores))


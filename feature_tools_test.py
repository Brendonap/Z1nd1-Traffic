import pandas as pd
import datetime
import math
import numpy as np

from sklearn import model_selection
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, LinearRegression, BayesianRidge, SGDRegressor

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold, cross_val_score

from sklearn.neural_network import MLPRegressor
from pystacknet.pystacknet import StackNetRegressor
from rgf.sklearn import RGFRegressor
import sys

import featuretools as ft


# --------------------------
# ---- Helper functions ----
# --------------------------

url = 'C:\\Users\\brendon.pitcher\\Documents\\Brendon\\Dev\\PlayTime\\Zindi\\TrafficJam\\train_revised.csv'
df = pd.read_csv(url)
columns = ['ride_id', 'travel_date', 'travel_time', 'travel_from', 'car_type']
df_train_set = df.groupby(columns).size().reset_index(name='number_of_tickets')
df_train_set = df_train_set.sort_values('travel_date', ascending=False)

# df_train_set.drop(['ride_id'], axis=1, inplace=True) #ride_id is unnecessary in training set

# df_train_set = df_train_set.replace(['Kendu Bay', 'Oyugis', 'Keumbu'], 'other')
# df_train_set["travel_date"] = pd.to_datetime(df_train_set["travel_date"],infer_datetime_format=True)
# df_train_set["travel_date"] = df_train_set["travel_date"].dt.dayofweek #change the full date to day of week

# df_train_set["car_type"] = pd.Categorical(df_train_set["car_type"])
# car_type_categories = df_train_set.car_type.cat.categories
# df_train_set["car_type"] = df_train_set.car_type.cat.codes

# df_train_set["travel_from"] = pd.Categorical(df_train_set["travel_from"])
# travel_from_categories = df_train_set.travel_from.cat.categories
# df_train_set["travel_from"] = df_train_set.travel_from.cat.codes

# #express travel time in minutes
# df_train_set["travel_time"] = df_train_set["travel_time"].str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))
# df_train_set['is_weekend'] = np.where(df_train_set['travel_date'] >= 5, 1, 0)

# df_train_set = df_train_set[df_train_set['number_of_tickets'] < 47]

# # ------ model
# X = df_train_set.drop(["number_of_tickets"], axis=1)
# y = df_train_set.number_of_tickets  

# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, shuffle=True, random_state=1)

es = ft.EntitySet(id="test")
es = es.entity_from_dataframe('train', df_train_set, make_index = True, index = 'train_index')

feature_matrix_customers, feature_defs = ft.dfs(entityset=es, target_entity = 'train')


def encoding(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = pd.Categorical(df[column])
            df[column] = df[column].cat.codes

    return df


df_train_set = encoding(feature_matrix_customers)



X = df_train_set.drop(["number_of_tickets"], axis=1)
y = df_train_set.number_of_tickets  

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, shuffle=True, random_state=1)

model = RandomForestRegressor(
        n_estimators=100, 
        criterion="mse", 
        max_depth=10, 
        min_samples_split=9, 
        min_samples_leaf=1, 
        min_weight_fraction_leaf=0, 
        max_leaf_nodes=None, 
        min_impurity_decrease=0.0005, 
        oob_score=True,
        random_state=1)

bags = 10
seed = 1

bagged_prediction = np.zeros(X_test.shape[0])

for n in range(0, bags):
    model.set_params(random_state=seed + n)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    bagged_prediction += preds

bagged_prediction /= bags

print(mean_absolute_error(y_test, np.round(bagged_prediction)))


sys.exit()


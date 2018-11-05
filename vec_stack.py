import pandas as pd
import datetime
import math
import numpy as np

from sklearn import model_selection
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor, XGBClassifier
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


# --------------------------
# ---- Helper functions ----
# --------------------------

url = 'C:\\Users\\brendon.pitcher\\Documents\\Brendon\\Dev\\PlayTime\\Zindi\\TrafficJam\\train_revised.csv'
df = pd.read_csv(url)
columns = ['ride_id', 'travel_date', 'travel_time', 'travel_from', 'car_type']
df_train_set = df.groupby(columns).size().reset_index(name='number_of_tickets')
df_train_set = df_train_set.sort_values('travel_date', ascending=False)

df_train_set.drop(['ride_id'], axis=1, inplace=True) #ride_id is unnecessary in training set

df_train_set["travel_date"] = pd.to_datetime(df_train_set["travel_date"],infer_datetime_format=True)
df_train_set["travel_date"] = df_train_set["travel_date"].dt.dayofweek #change the full date to day of week

df_train_set["car_type"] = pd.Categorical(df_train_set["car_type"])
car_type_categories = df_train_set.car_type.cat.categories
df_train_set["car_type"] = df_train_set.car_type.cat.codes

df_train_set = df_train_set.replace(['Kendu Bay', 'Oyugis', 'Keumbu'], 'other')
df_train_set["travel_from"] = pd.Categorical(df_train_set["travel_from"])
travel_from_categories = df_train_set.travel_from.cat.categories
df_train_set["travel_from"] = df_train_set.travel_from.cat.codes

#express travel time in minutes
df_train_set["travel_time"] = df_train_set["travel_time"].str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))
df_train_set['is_weekend'] = np.where(df_train_set['travel_date'] >= 5, 1, 0)

# ------ model
X = df_train_set.drop(["number_of_tickets"], axis=1)
y = df_train_set.number_of_tickets  

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, shuffle=True)



from vecstack import StackingTransformer

estimators_L1 = [
    ('et', ExtraTreesRegressor(n_estimators=100, criterion="mae", random_state=1)),
    ('rf', RandomForestRegressor(n_estimators=100, criterion="mse",max_depth=10,min_samples_split=9,min_samples_leaf=1,min_weight_fraction_leaf=0,max_leaf_nodes=None,min_impurity_decrease=0.0005)),
    ('xgb', XGBRegressor(n_estimators=100, criterion="mae", max_depth=12, subsample=0.5, learning_rate=0.05, colsample_bytree=0.9))
    ]
# Stacking
stack = StackingTransformer(estimators=estimators_L1,regression=True,shuffle=True,random_state=0,verbose=2, stratified=True, n_folds=5)
stack = stack.fit(X_train, y_train)

S_train = stack.transform(X_train)
S_test = stack.transform(X_test)

# Use 2nd level estimator to get final prediction
estimator_L2 = XGBRegressor(random_state=0,n_jobs=-1,learning_rate=0.1,n_estimators=100,max_depth=3)
estimator_L2 = estimator_L2.fit(S_train, y_train)
y_pred = estimator_L2.predict(S_test)

# Final prediction score
print('Final score: [%.8f]' % mean_absolute_error(y_test, y_pred))
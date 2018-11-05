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


import matplotlib.pyplot as plt
import seaborn as sns

# peak 400-460  and 1120-1170


# am - 7 to 10
# mid 10 to 4
# pm 4 to 7
# eve 7 to 12 
# morn 12 to 7


# --------------------------
# ---- Helper functions ----
# --------------------------

def time_of_day(x):
    time = x["hour"]
    if time >= 7 and time < 10:
        return 1
    elif time >= 10 and time < 16:
        return 2
    elif time >= 16 and time < 19:
        return 3
    elif time >= 19 and time <= 24:
        return 4
    elif time < 7:
        return 5
    else:
        return 0

def get_data(cut_off=None, direction=None):
    url = 'C:\\Users\\brendon.pitcher\\Documents\\Brendon\\Dev\\PlayTime\\Zindi\\TrafficJam\\train_revised.csv'
    df = pd.read_csv(url)
    columns = ['ride_id', 'travel_date', 'travel_time', 'travel_from', 'car_type']
    df_train_set = df.groupby(columns).size().reset_index(name='number_of_tickets')
    df_train_set = df_train_set.sort_values('travel_date', ascending=False)

    if (cut_off):
        if direction == 'Bus':
            df_train_set = df_train_set[df_train_set['car_type'] == 'Bus']
        else:
            df_train_set = df_train_set[df_train_set['car_type'] == 'shuttle']

    df_train_set.drop(['ride_id'], axis=1, inplace=True) #ride_id is unnecessary in training set

    df_train_set["travel_date"] = pd.to_datetime(df_train_set["travel_date"],infer_datetime_format=True)
    df_train_set["month"] = df_train_set["travel_date"].dt.month 
    df_train_set["travel_date"] = df_train_set["travel_date"].dt.dayofweek #change the full date to day of week

    df_train_set["car_type"] = pd.Categorical(df_train_set["car_type"])
    car_type_categories = df_train_set.car_type.cat.categories
    df_train_set["car_type"] = df_train_set.car_type.cat.codes

    df_train_set["travel_from"] = pd.Categorical(df_train_set["travel_from"])
    travel_from_categories = df_train_set.travel_from.cat.categories
    df_train_set["travel_from"] = df_train_set.travel_from.cat.codes

    #express travel time in minutes
    df_train_set["hour"] = df_train_set["travel_time"].str.split(':').apply(lambda x: int(x[0]))
    df_train_set['time_cat'] = df_train_set.apply(time_of_day, axis=1)
    df_train_set["travel_time"] = df_train_set["travel_time"].str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))
    df_train_set['is_weekend'] = np.where(df_train_set['travel_date'] >= 5, 1, 0)


    # ------ model
    X = df_train_set.drop(["number_of_tickets", 'hour', 'time_cat'], axis=1)
    y = df_train_set.number_of_tickets  

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, shuffle=True, random_state=1)
    return X_train, X_test, y_train, y_test

# ## bagging example
# df_test_set = pd.read_csv('test_questions.csv', low_memory=False)

# df_test_set.drop(['travel_to'], axis=1, inplace=True)
# df_test_set = df_test_set.sort_values('travel_date', ascending=False)

# df_test_set["travel_date"] = pd.to_datetime(df_test_set["travel_date"],infer_datetime_format=True)
# df_test_set["travel_date"] = df_test_set["travel_date"].dt.dayofweek

# df_test_set["car_type"] = pd.Categorical(df_test_set["car_type"], categories=car_type_categories)
# df_test_set["car_type"] = df_test_set.car_type.cat.codes

# df_test_set["travel_from"] = pd.Categorical(df_test_set["travel_from"], categories=travel_from_categories)
# df_test_set["travel_from"] = df_test_set.travel_from.cat.codes

# df_test_set["travel_time"] = df_test_set["travel_time"].str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))
# df_test_set['is_weekend'] = np.where(df_test_set['travel_date'] >= 5, 1, 0)

# X_test = df_test_set.drop(['ride_id', 'max_capacity'], axis=1)

# stacking test
# def Stacking(model,train,y,n_fold):
#     folds=StratifiedKFold(n_splits=n_fold,random_state=1)
#     train_pred=np.empty((0,1),float)
#     for train_indices,val_indices in folds.split(train,y.values):
#        x_train, x_val = train.iloc[train_indices], train.iloc[val_indices]
#        y_train = y.iloc[train_indices]

#        model.fit(X=x_train,y=y_train)
#        train_pred=np.append(train_pred,model.predict(x_val))
#     return pd.DataFrame(train_pred, columns=['class'])

# model = RandomForestRegressor(
#         n_estimators=100, 
#         criterion="mse", 
#         max_depth=10, 
#         min_samples_split=9, 
#         min_samples_leaf=1, 
#         min_weight_fraction_leaf=0, 
#         max_leaf_nodes=None, 
#         min_impurity_decrease=0.0005, 
#         oob_score=True,
#         random_state=1)


# preds = Stacking(model, X, y, 5)

# X['preds'] = preds
# X['number_of_tickets'] = y

def get_bagged_predictions(X_train, X_test, y_train, y_test):

    model = RandomForestRegressor(
            n_estimators=100, 
            criterion="mae", 
            max_depth=10, 
            min_samples_split=9, 
            min_samples_leaf=1, 
            min_weight_fraction_leaf=0, 
            max_leaf_nodes=None, 
            min_impurity_decrease=0.0005, 
            oob_score=True,
            random_state=1)
    

    bags = 5
    seed = 1

    bagged_prediction = np.zeros(X_test.shape[0])

    for n in range(0, bags):
        model.set_params(random_state=seed + n)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        bagged_prediction += preds

    bagged_prediction /= bags
    return bagged_prediction, model

# bagged_prediction[bagged_prediction < 3] = 1

def plot_corr(dataset):
    pd.scatter_matrix(dataset, alpha = 0.3, figsize = (14,8), diagonal = 'hist')

    # sns.pairplot(dataset)
    # f, ax = plt.subplots(figsize=(10, 8))
    # corr = dataset.corr()
    # sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
                # square=True, ax=ax)

    plt.show()


X_train_bus, X_test_bus, y_train_bus, y_test_bus = get_data()

bagged_prediction, model_bus = get_bagged_predictions(X_train_bus, X_test_bus, y_train_bus, y_test_bus)
print(mean_absolute_error(y_test_bus, np.floor(bagged_prediction)))

X_test_bus['preds'] = bagged_prediction
X_test_bus['real'] = y_test_bus
X_test_bus['diff'] = abs(X_test_bus['real'] - X_test_bus['preds'])

# X_test_bus.sort_values('diff', ascending=False, inplace=True)
# print(X_test_bus.head(50))

plot_corr(X_test_bus)

sys.exit()


X_train_bus, X_test_bus, y_train_bus, y_test_bus = get_data(True, 'Bus')

bagged_prediction, model_bus = get_bagged_predictions(X_train_bus, X_test_bus, y_train_bus, y_test_bus)
print(mean_absolute_error(y_test_bus, np.floor(bagged_prediction)))
print(len(y_test_bus))

X_test_bus['pred'] = y_test_bus
X_test_bus['bus_pred'] = bagged_prediction
plot_corr(X_test_bus)


# X_test_small['preds'] = np.floor(bagged_prediction)
# X_test_small['number_of_tickets'] = y_test_small
# plot_corr(X_test_small)


X_train_shut, X_test_shut, y_train_shut, y_test_shut = get_data(True)
bagged_prediction_shut, model_shut = get_bagged_predictions(X_train_shut, X_test_shut, y_train_shut, y_test_shut)
print(mean_absolute_error(y_test_shut, np.round(bagged_prediction_shut)))
print(len(y_test_shut))


X_test_shut['pred'] = y_test_shut
X_test_shut['shut_pred'] = bagged_prediction_shut
plot_corr(len(X_test_shut))


# 0.4896 0.5104


# X_test_big['preds'] = np.floor(bagged_prediction_big)
# X_test_big['number_of_tickets'] = y_test_big
# plot_corr(X_test_big)

X_train, X_test, y_train, y_test = get_data()
bagged_prediction_full, model_full = get_bagged_predictions(X_train, X_test, y_train, y_test)
print(mean_absolute_error(y_test, np.round(bagged_prediction_full)))

pred_small = model_small.predict(X_test)
pred_big = model_big.predict(X_test)
pred_full = model_full.predict(X_test)

X_test['pred_small'] = pred_small
X_test['pred_big'] = pred_big
X_test['pred_full'] = pred_full
X_test['preds'] = y_test


print('small: ', mean_absolute_error(X_test['preds'], X_test['pred_small']))
print('big: ', mean_absolute_error(X_test['preds'], X_test['pred_big']))
print('full: ', mean_absolute_error(X_test['preds'], X_test['pred_full']))


# X_test['weight'] = np.where(X_test['pred_small'] >= 8, X_test['pred_small'], X_test['pred_big'])
# X_test['averge'] = (X_test['pred_small'] + X_test['pred_full'] + X_test['pred_big']) / 3
# X_test['median'] = X_test.apply(median_value, axis=1)

plot_corr(X_test)

best_weight = 0
best_score = 10000000

# for i in range(5, 100, 5):
#     i = i/100
#     for u in range(5, 100, 5):
#         pred = X_test['pred_small'] * i + X_test['pred_big'] * (1-i) + X_test['pred_full']
#         print(mean_absolute_error(X_test['preds'], pred))
#         error = mean_absolute_error(X_test['preds'], pred)
#         if error < best_score:
#             best_weight = [i, (1-i), u]
#             best_score = error

# print(best_weight, best_score)


# for i in range(5, 100, 5):
#     i = i/100
#     pred = X_test['pred_small'] * i + X_test['pred_full'] * (1-i)
#     print(mean_absolute_error(X_test['preds'], pred))
#     error = mean_absolute_error(X_test['preds'], pred)
#     if error < best_score:
#         best_weight = [i, (1-i)]
#         best_score = error

# print(best_weight, best_score)


# X_test['preds'] = bagged_prediction
# X_test['number_of_tickets'] = y_test




# check prediction by car type some may go over the shuttle
# travel time distibution is heavily skewered to rush hour times
#   - bucket travel time in different intervals


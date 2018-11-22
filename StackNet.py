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



url = 'C:\\Users\\brendon.pitcher\\Documents\\Brendon\\Dev\\PlayTime\\Zindi\\TrafficJam\\train_revised.csv'
df = pd.read_csv(url)
columns = ['ride_id', 'travel_date', 'travel_time', 'travel_from', 'car_type']
df_train_set = df.groupby(columns).size().reset_index(name='number_of_tickets')
#df_train_set = df_train_set.sort_values('travel_date', ascending=False)

#ride_id is unnecessary in training set
df_train_set.drop(['ride_id'], axis=1, inplace=True) 

# convert travel date to datetime
# df_train_set = df_train_set.replace(['Kendu Bay', 'Oyugis', 'Keumbu'], 'other')
df_train_set["travel_date"] = pd.to_datetime(df_train_set["travel_date"],infer_datetime_format=True)

# remove month 2 and 3 2017
mask = (df_train_set["travel_date"] >= '2017-04-01')
df_train_set = df_train_set.loc[mask]

#change the full date to day of week
df_train_set["month"] = df_train_set["travel_date"].dt.month 
df_train_set["week"] = df_train_set["travel_date"].dt.week

df_train_set["travel_date"] = df_train_set["travel_date"].dt.dayofweek 

# encode car type
df_train_set["car_type"] = pd.Categorical(df_train_set["car_type"])
car_type_categories = df_train_set.car_type.cat.categories
df_train_set["car_type"] = df_train_set.car_type.cat.codes

# encode travel_from
df_train_set["travel_from"] = pd.Categorical(df_train_set["travel_from"])
travel_from_categories = df_train_set.travel_from.cat.categories
df_train_set["travel_from"] = df_train_set.travel_from.cat.codes

# df_train_set = pd.concat([df_train_set, pd.get_dummies(df_train_set['travel_from'], prefix='travel_from')], axis=1).drop(['travel_from'], axis=1)


#express travel time in minutes
#df_train_set["travel_time"] = df_train_set["travel_time"].str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))
# df_train_set["is_peak_time"] = np.where((df_train_set['travel_time'] >= 400) & (df_train_set['travel_time'] <= 460) & (df_train_set['travel_time'] >= 1120) & (df_train_set['travel_time'] <= 1170), 1, 0)

df_train_set["travel_time"] = pd.to_datetime(df_train_set["travel_time"],infer_datetime_format=True)
df_train_set["hour_booked"] = df_train_set["travel_time"].dt.hour
df_train_set["minute_booked"] = df_train_set["travel_time"].dt.minute
df_train_set.drop('travel_time', axis=1, inplace=True)


# create is_weekend feature
df_train_set['is_weekend'] = np.where(df_train_set['travel_date'] >= 5, 1, 0)

# ------ model
X = df_train_set.drop(["number_of_tickets"], axis=1)
y = df_train_set.number_of_tickets  

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, shuffle=True, random_state=1)


if __name__ == '__main__':
    models=[ 
            ######## First level ########
            [
                #AdaBoostRegressor(RandomForestRegressor(n_estimators=100,criterion='mse'),n_estimators=8, learning_rate= 0.05),
                #LinearRegression(fit_intercept=True, normalize= True),
                #Ridge(alpha=0.1, fit_intercept=True, normalize=False),
                #BaggingRegressor(RandomForestRegressor(n_estimators=100, criterion="mae"), oob_score= False, n_estimators= 20, max_samples= 0.5, max_features= 0.7),
                #SVR(max_iter=-1, degree=5, kernel='rbf'),
                #KNeighborsRegressor(n_neighbors=5),
                #MLPRegressor(early_stopping=True, hidden_layer_sizes=(2), learning_rate_init= 0.01, max_iter= 1000),
                XGBRegressor(n_estimators=100, criterion="mae", max_depth=12, subsample=0.5, learning_rate=0.05, colsample_bytree=0.9),
                #ExtraTreesRegressor(n_estimators=120, criterion="mae", max_depth=10, max_features=0.5, random_state=1),
                GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=50, max_features=0.5, random_state=1),
                RandomForestRegressor(n_estimators=100, criterion="mae", random_state=1),
                RGFRegressor(max_leaf=4500, algorithm="RGF_Sib", test_interval=50, loss="LS", verbose=False),
            ],
            ######## Second level ########
            [
            #    RandomForestRegressor(n_estimators=100, criterion="mae", random_state=1),
                RandomForestRegressor(n_estimators=100, criterion="mse",max_depth=10,min_samples_split=9,min_samples_leaf=1,min_weight_fraction_leaf=0,max_leaf_nodes=None,min_impurity_decrease=0.0005,oob_score=True)
                # RGFRegressor(max_leaf=4500, algorithm="RGF_Sib", test_interval=50, loss="LS", verbose=False),
            #    ExtraTreesRegressor(n_estimators=100, criterion="mae", random_state=1)
            ],
            #[
            #    RandomForestRegressor(n_estimators=100, criterion="mse",max_depth=10,min_samples_split=9,min_samples_leaf=1,min_weight_fraction_leaf=0,max_leaf_nodes=None,min_impurity_decrease=0.0005,oob_score=True),
            #]
        ]

    model = StackNetRegressor(models, metric="mae", folds=5, restacking=True, use_retraining=False, random_state=12345, verbose=1)

    #score = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=5)
    #print(sum(score) / len(score))

    model.fit(X_train, y_train)
#print(model.score(X_train.values, y_train))

# predictions = model.predict(X_test.as_matrix())
# print(mean_absolute_error(y_test, predictions))


#----- results

df_test_set = pd.read_csv('test_questions.csv', low_memory=False)

df_test_set.drop(['travel_to'], axis=1, inplace=True)
df_test_set = df_test_set.sort_values('travel_date', ascending=False)

# df_test_set = df_test_set.replace(['Kendu Bay', 'Oyugis', 'Keumbu'], 'other')
df_test_set["travel_date"] = pd.to_datetime(df_test_set["travel_date"],infer_datetime_format=True)
df_test_set["month"] = df_test_set["travel_date"].dt.month 
df_test_set["week"] = df_test_set["travel_date"].dt.week 

df_test_set["travel_date"] = df_test_set["travel_date"].dt.dayofweek

df_test_set["car_type"] = pd.Categorical(df_test_set["car_type"], categories=car_type_categories)
df_test_set["car_type"] = df_test_set.car_type.cat.codes

df_test_set["travel_from"] = pd.Categorical(df_test_set["travel_from"], categories=travel_from_categories)
df_test_set["travel_from"] = df_test_set.travel_from.cat.codes

# df_test_set["travel_time"] = df_test_set["travel_time"].str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))
df_test_set["travel_time"] = pd.to_datetime(df_test_set["travel_time"],infer_datetime_format=True)
df_test_set["hour_booked"] = df_test_set["travel_time"].dt.hour
df_test_set["minute_booked"] = df_test_set["travel_time"].dt.minute
df_test_set.drop('travel_time', axis=1, inplace=True)

df_test_set['is_weekend'] = np.where(df_test_set['travel_date'] >= 5, 1, 0)

X_test = df_test_set.drop(['ride_id', 'max_capacity'], axis=1)

test_set_predictions = model.predict(X_test.values)

d = {'ride_id': df_test_set["ride_id"], 'number_of_ticket': test_set_predictions[:,0]}
df_predictions = pd.DataFrame(data=d)
df_predictions = df_predictions[['ride_id','number_of_ticket']]

df_predictions.to_csv('results.csv', index=False)


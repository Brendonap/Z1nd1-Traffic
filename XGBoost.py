import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import sys
from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV, GridSearchCV
from xgboost import XGBRegressor


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

#df_train_set.to_csv('C:\\Users\\brendon.pitcher\\Documents\\Brendon\\Dev\\PlayTime\\Zindi\\TrafficJam\\prediction_prode.csv')

#express travel time in minutes
df_train_set["travel_time"] = df_train_set["travel_time"].str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))
df_train_set['is_weekend'] = np.where(df_train_set['travel_date'] >= 5, 1, 0)


#print(df_train_set.head(5))

# ------ model
X = df_train_set.drop(["number_of_tickets"], axis=1)
y = df_train_set.number_of_tickets  

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, shuffle=True)

if __name__ == '__main__':
    xgb1 = XGBRegressor()
    parameters = {'nthread':[6], #when use hyperthread, xgboost may become slower
                'objective':['reg:linear'],
                'learning_rate': [0.02, 0.04], #so called `eta` value
                'max_depth': [12],
                'min_child_weight': [3.5, 4, 4.5, 6],
                'subsample': [0.6, 0.7, 0.8, 1],
                'colsample_bytree': [0.6, 0.7, 0.8, 1],
                'n_estimators': [200, 300, 500, 1000]
                }

    xgb_grid = GridSearchCV(xgb1,
                            parameters,
                            cv = 5,
                            scoring='neg_mean_absolute_error', 
                            n_jobs = 6,
                            verbose=True)

    xgb_grid.fit(X, y)
    #predictions = xgb_grid.predict(X)
    #test_predict = xgb_grid.predict(X_test)

    print(xgb_grid.best_estimator_)
    print(xgb_grid.best_score_)
    
#print(mean_absolute_error(y_test, predictions))

#scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=5)
#print(sum(scores) / len(scores))

#print(X.head(5))
#model.fit(X, y)
#preds_train_set = model.predict(X_test)

#print(mean_absolute_error(preds_train_set, y_test))
'''
XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.02, max_delta_step=0,
       max_depth=12, min_child_weight=6, missing=None, n_estimators=200,
       n_jobs=1, nthread=6, objective='reg:linear', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)
'''


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import sys
from pystacknet.pystacknet import StackNetRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn import model_selection
from xgboost import XGBRegressor

from sklearn.model_selection import cross_val_score, RandomizedSearchCV

from sklearn.linear_model import Ridge, LinearRegression

def validation_process(model, X, y, parameters, n_folds):

    gb_grid = RandomizedSearchCV(model,
                            parameters,
                            cv = n_folds,
                            scoring='neg_mean_absolute_error', 
                            n_jobs = 1,
                            iid = True,
                            verbose=True)

    gb_grid.fit(X_train,y_train)
    print(gb_grid.best_params_)
    print(gb_grid.best_score_)
    return gb_grid.best_params_ 


url = 'C:\\Users\\brendon.pitcher\\Documents\\Brendon\\Dev\\PlayTime\\Zindi\\TrafficJam\\train_revised.csv'
df = pd.read_csv(url)
columns = ['ride_id', 'travel_date', 'travel_time', 'travel_from', 'car_type', 'max_capacity']
df_train_set = df.groupby(columns).size().reset_index(name='number_of_tickets')

df_train_set.drop(['ride_id'], axis=1, inplace=True) #ride_id is unnecessary in training set

df_train_set["travel_date"] = pd.to_datetime(df_train_set["travel_date"],infer_datetime_format=True)
df_train_set["travel_date"] = df_train_set["travel_date"].dt.dayofweek

df_train_set["day"] = pd.to_datetime(df_train_set["travel_date"],infer_datetime_format=True)
df_train_set["day"] = df_train_set["day"].dt.day

df_train_set['max_capacity'] = np.where(df_train_set['max_capacity'] == 49, 1, 0)

df_train_set["car_type"] = pd.Categorical(df_train_set["car_type"])
car_type_categories = df_train_set.car_type.cat.categories
df_train_set["car_type"] = df_train_set.car_type.cat.codes

df_train_set = pd.concat([df_train_set, pd.get_dummies(df_train_set['travel_from'], prefix='travel_from')], axis=1).drop(['travel_from'], axis=1)

#df_train_set["travel_from"] = pd.Categorical(df_train_set["travel_from"])
#travel_from_categories = df_train_set.travel_from.cat.categories
#df_train_set["travel_from"] = df_train_set.travel_from.cat.codes

df_train_set["travel_time"] = df_train_set["travel_time"].str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))

#print(df_train_set.head(5))

# ------ model

X = df_train_set.drop(["number_of_tickets"], axis=1)
y = df_train_set.number_of_tickets  

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, shuffle=False)

model = RandomForestRegressor(n_estimators=50, criterion="mae", max_features=0.7, random_state=1, oob_score=True)
#model = ExtraTreesRegressor(n_estimators=120, criterion="mae", max_features=0.5, random_state=1)

parameters = {
    'bootstrap': [True, False],
    'max_depth': [10, 20, 30, 50, 60, 80, 90, 100, None],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [50, 75, 100, 150, 200, 250],
    'criterion': ["mae"]
    }

best_params = {'oob_score':True, 'n_estimators': 50, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_features': 'auto', 'max_depth': 80, 'criterion': 'mae', 'bootstrap': True}

#best_params = validation_process(model, X, y, parameters, 5)

model = RandomForestRegressor(**best_params)

model.fit(X_train, y_train)
preds_train_set = model.predict(X_test)

print(mean_absolute_error(preds_train_set,y_test))



sys.exit()

# ----------------------
# ---- Test dataset ----
# ----------------------

df_test_set = pd.read_csv('test_questions.csv', low_memory=False)

df_test_set.drop(['travel_to'], axis=1, inplace=True)

df_test_set["travel_date"] = pd.to_datetime(df_test_set["travel_date"],infer_datetime_format=True)
df_test_set["travel_date"] = df_test_set["travel_date"].dt.dayofweek

df_test_set["day"] = pd.to_datetime(df_test_set["travel_date"],infer_datetime_format=True)
df_test_set["day"] = df_test_set["day"].dt.day

df_test_set["car_type"] = pd.Categorical(df_test_set["car_type"], categories=car_type_categories)
df_test_set["car_type"] = df_test_set.car_type.cat.codes

#df_test_set["travel_from"] = pd.Categorical(df_test_set["travel_from"], categories=travel_from_categories)
#df_test_set["travel_from"] = df_test_set.travel_from.cat.codes

df_test_set = pd.concat([df_test_set, pd.get_dummies(df_test_set['travel_from'], prefix='travel_from')], axis=1).drop(['travel_from'], axis=1)


df_test_set["travel_time"] = df_test_set["travel_time"].str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))

X_test = df_test_set.drop(['ride_id'], axis=1)

# add columns to test set if they dont exist after feature encoding
if ((len(X.columns)) > (len(X_test.columns) + 1)):
    missing_columns = list(set(X.columns) - set(X_test.columns) - set(['number_of_tickets']))

    for miss in missing_columns:
        X_test[miss] = 0

test_set_predictions = model.predict(X_test)

d = {'ride_id': df_test_set["ride_id"], 'number_of_ticket': test_set_predictions}
df_predictions = pd.DataFrame(data=d)
df_predictions = df_predictions[['ride_id','number_of_ticket']]

df_predictions.to_csv('results.csv', index=False)


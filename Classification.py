import pandas as pd
import numpy as np
import datetime
import sys

from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, VotingClassifier

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, log_loss
from pystacknet.pystacknet import StackNetRegressor

from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn import model_selection

from sklearn.linear_model import Ridge, LinearRegression


url = 'C:\\Users\\brendon.pitcher\\Documents\\Brendon\\Dev\\PlayTime\\Zindi\\TrafficJam\\train_revised.csv'
df = pd.read_csv(url)
columns = ['ride_id', 'travel_date', 'travel_time', 'travel_from', 'car_type']
df_train_set = df.groupby(columns).size().reset_index(name='number_of_tickets')

df_train_set.drop(['ride_id'], axis=1, inplace=True) #ride_id is unnecessary in training set

df_train_set["travel_date"] = pd.to_datetime(df_train_set["travel_date"],infer_datetime_format=True)
df_train_set["travel_date"] = df_train_set["travel_date"].dt.dayofweek

# df_train_set['max_capacity'] = np.where(df_train_set['max_capacity'] == 49, 1, 0)

df_train_set["car_type"] = pd.Categorical(df_train_set["car_type"])
car_type_categories = df_train_set.car_type.cat.categories
df_train_set["car_type"] = df_train_set.car_type.cat.codes

df_train_set = df_train_set.replace(['Kendu Bay', 'Oyugis', 'Keumbu'], 'other')
#df_train_set = pd.concat([df_train_set, pd.get_dummies(df_train_set['travel_from'], prefix='travel_from')], axis=1).drop(['travel_from'], axis=1)

df_train_set["travel_from"] = pd.Categorical(df_train_set["travel_from"])
travel_from_categories = df_train_set.travel_from.cat.categories
df_train_set["travel_from"] = df_train_set.travel_from.cat.codes

df_train_set["travel_time"] = df_train_set["travel_time"].str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))

X = df_train_set.drop(["number_of_tickets"], axis=1)
y = df_train_set.number_of_tickets

def Stacking(model,train,y,n_fold):
    folds=StratifiedKFold(n_splits=n_fold,random_state=1)
    train_pred=np.empty((0,1),float)
    for train_indices,val_indices in folds.split(train,y.values):
       x_train, x_val = train.iloc[train_indices], train.iloc[val_indices]
       y_train = y.iloc[train_indices]

       model.fit(X=x_train,y=y_train)
       train_pred=np.append(train_pred,model.predict(x_val))
    return pd.DataFrame(train_pred, columns=['class'])

def target(y):
    if y < 12:
        return 1 
    else:
        return 0

y_class = y.apply(target)

#X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y_class, test_size=0.25, shuffle=True)

#multi:softprob


# if __name__ == '__main__':
#     xgb1 = XGBClassifier()
#     parameters = {
#                 #'nthread':[6], #when use hyperthread, xgboost may become slower
#                 'learning_rate': [0.03, 0.04, 0.05], #so called `eta` value
#                 'max_depth': [4, 5, 7],
#                 'n_estimators': [50, 60, 70],
#                 'colsample_bytree': [0.1, 0.25, 0.5, 0.75, 1]
#                 }

#     xgb_grid = RandomizedSearchCV(xgb1,
#                             parameters,
#                             cv = 5,
#                             scoring='accuracy', 
#                             n_jobs = 1,
#                             verbose=True)

#     xgb_grid.fit(X, y_class)
#     #predictions = xgb_grid.predict(X)
#     #test_predict = xgb_grid.predict(X_test)

#     print(xgb_grid.best_estimator_)
#     print(xgb_grid.best_score_)
    
# sys.exit()

xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.04, max_delta_step=0,
       max_depth=5, min_child_weight=1, missing=None, n_estimators=70,
       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)

# xgb.fit(X_train, y_train)
# pred = xgb.predict(X_test)
# print(accuracy_score(y_test, pred))

# scores = cross_val_score(model, X, y_class, scoring='accuracy', cv=5)
# print(sum(scores)/len(scores))

# class_pred = Stacking(xgb, X, y_class, 5)
# X = pd.concat([X, class_pred.reset_index()], axis=1)


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, shuffle=False)

model = RandomForestRegressor(n_estimators=100, criterion="mse",max_depth=10,min_samples_split=9,min_samples_leaf=1,min_weight_fraction_leaf=0,max_leaf_nodes=None,min_impurity_decrease=0.0005)
model.fit(X_train, y_train)

preds = model.predict(X_test)

print(mean_absolute_error(np.round(preds), y_test))

# ------------------------
# ---- Classification ----
# ------------------------

'''
# hyper param tuning classifier
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }

xgb = XGBClassifier(learning_rate=0.02, n_estimators=250, objective='binary:logistic',
                    silent=True, nthread=1)

random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=5, scoring='neg_log_loss', 
                    n_jobs=1, cv=5, verbose=1, random_state=1001)
random_search = XGBRegressor(learning_rate=0.02, n_estimators=250, objective='binary:logistic')
random_search.fit(X_train, y_train)

pred = random_search.predict_proba(X_test)

print(log_loss(y_test, pred))
#print(accuracy_score(y_test, pred))

#tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel() 
#print(tn, tp, fp, fn)
'''


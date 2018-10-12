import pandas as pd
import numpy as np
import datetime
import sys

from sklearn import model_selection

from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor, BaggingRegressor

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, LeaveOneOut
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, log_loss



def convert_to_date(time_string):
    try:
        format_str = '%d-%m-%y'
        return datetime.datetime.strptime(time_string, format_str)
    except ValueError:
        format_str = '%Y-%m-%d'
        return datetime.datetime.strptime(time_string, format_str)

def convert_timestamp_to_int(df, column_name='travel_date'):
    #df['year'] = df['travel_date'].apply(lambda x: (convert_to_date(x)).year)
    df['month'] = df['travel_date'].apply(lambda x: (convert_to_date(x)).month)
    df['day'] = df[column_name].apply(lambda x: (convert_to_date(x)).day)
    df['day_of_week'] = df[column_name].apply(lambda x: (convert_to_date(x)).weekday())
    df.drop([column_name], axis=1, inplace=True)
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
    data.drop(['travel_time'], axis=1, inplace=True)

    data = data.replace(['Kendu Bay', 'Oyugis', 'Keumbu', 'Sori'], 'other')
    data['car_type_is_bus'] = np.where(data['car_type'] == 'Bus', 1, 0)
    data.drop('car_type', axis=1, inplace=True)

    #encoding = data.groupby('travel_from').size()
    #encoding = encoding/len(data)
    #data['travel_encoding'] = data.travel_from.map(encoding)
    #data.drop('travel_from', axis=1, inplace=True)

    #data["travel_from"] = pd.Categorical(data["travel_from"])
    #data["travel_from"] = data.travel_from.cat.codes
    data = pd.concat([data, pd.get_dummies(data['travel_from'], prefix='travel_from')], axis=1).drop(['travel_from'], axis=1)
    return data

def process_uber_data(df):
    df['day'] = pd.DatetimeIndex(df['Date']).day

    df.drop(
        ['Date', 'Origin Movement ID', 'Destination Movement ID', 'Origin Display Name', 
        'Destination Display Name'], axis=1, inplace=True)

    df = df.groupby('day').mean()
    return df


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
test = pd.read_csv(url)
X_test = processing_pipeline(test)
X_test.drop(['travel_to'], axis=1, inplace=True)

# add columns to test set if they dont exist after feature encoding
if ((len(train.columns)) > (len(X_test.columns) + 1)):
    missing_columns = list(set(train.columns) - set(X_test.columns) - set(['counts']))

    for miss in missing_columns:
        X_test[miss] = 0

# reorder columns
X_test = X_test[train.drop('counts', axis=1).columns]

train['class'] = np.where(train['counts'] < 10, 1, 0)


# create X/Y training and validation sets
X_train_class = train.drop(['counts', 'class'], axis=1)
y_train_class = train[['class', 'counts']]

#X_train_class, X_test, y_train_class, y_test = model_selection.train_test_split(X_train_class, y_train_class, test_size=0.25, shuffle=False)

y_train = y_train_class.drop('class', axis=1)
y_train_class = y_train_class.drop('counts', axis=1)

#y_test_class = y_test.drop('counts', axis=1)
y_test.drop('class', axis=1, inplace=True)


# ------------------------
# ---- Classification ----
# ------------------------


def Stacking(model, train , y, test,n_fold):
    folds=StratifiedKFold(n_splits=n_fold,random_state=1)
    test_pred = []
    train_pred = []

    for train_indices, val_indices in folds.split(train, y.values):
        x_train, x_val = train.iloc[train_indices], train.iloc[val_indices]
        y_train = y.iloc[train_indices]
        model.fit(X=x_train, y=y_train)
        p = model.predict(x_val)
        train_pred = np.append(train_pred, p)

    test_pred=np.append(test_pred, model.predict(test))
    return pd.DataFrame(test_pred, columns=['class']), pd.DataFrame(train_pred, columns=['class'])

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
'''
random_search = XGBClassifier(learning_rate=0.02, n_estimators=250, objective='binary:logistic',
                    silent=True, nthread=1)

test, train = Stacking(random_search, X_train_class, y_train_class, X_test, 10)

X_train = pd.concat([X_train_class.reset_index(), train.reset_index()], axis=1)

X_test = pd.concat([X_test.reset_index(), test.reset_index()], axis=1)
X_test.drop('index', axis=1, inplace=True)
X_train.drop('index', axis=1, inplace=True)

print(log_loss(y_test_class, test))
tn, fp, fn, tp = confusion_matrix(y_test_class, test).ravel() 
print(tn, fp, fn, tp)

# --------------------
# ---- Regression ----
# --------------------

models=[ 
        ######## First level ########
        [
            XGBRegressor(n_estimators=150, criterion="mae", max_depth=10, subsample=0.75, learning_rate=0.1, colsample_bytree=0.9),
            #ExtraTreesRegressor(n_estimators=120, criterion="mae", max_depth=12, max_features=0.5, random_state=1),
            #GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, max_features=0.5, random_state=1),
            #RandomForestRegressor(n_estimators=150, criterion="mae", max_depth=10, max_features=0.5, random_state=1, oob_score=True),
            #AdaBoostRegressor(ExtraTreesRegressor(n_estimators=50, criterion="mae", max_depth=3, max_features=0.5, random_state=1), learning_rate=0.1, loss='linear', n_estimators=1000)
        ],
        ######## Second level ########
        #[
            #ExtraTreesRegressor(n_estimators=100, criterion="mae", max_depth=20, max_features=0.5, random_state=1),
        #]
        # [
        #     XGBRegressor(n_estimators=100, criterion="mae", max_depth=3, max_features=0.5, random_state=1),
        # ]
    ]

from pystacknet.pystacknet import StackNetRegressor

model = StackNetRegressor(models, metric="mae", folds=5, restacking=True, use_retraining=False, random_state=12345, verbose=1)

model.fit(X_train, y_train)
print(model.score(X_train.values, y_train))

predictions = model.predict(X_test.as_matrix())
predictions[predictions < 2] = 1

print(mean_absolute_error(y_test, predictions))

'''
# save results
url = 'C:\\Users\\brendon.pitcher\\Documents\\Brendon\\Dev\\PlayTime\\Zindi\\TrafficJam\\results.csv'

X_test['numer_of_ticket'] = predictions
X_test[['ride_id', 'numer_of_ticket']].to_csv(url, index=False)
'''


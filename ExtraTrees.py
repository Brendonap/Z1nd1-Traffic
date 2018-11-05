import pandas as pd
import datetime
import math
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import model_selection
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold

from sklearn.neural_network import MLPRegressor

# --------------------------
# ---- Helper functions ----
# --------------------------

def plot_feature_importance(X_test, model):
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_],
                axis=0)
    indices = np.argsort(importances)[::-1]
    feature_list = [X_train.columns[indices[f]] for f in range(X_train.shape[1])]  #names of features.
    ff = np.array(feature_list)

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X_train.shape[1]):
        print("%d. feature %d (%f) name: %s" % (f + 1, indices[f], importances[indices[f]], ff[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.rcParams['figure.figsize'] = [16, 6]
    plt.title("Feature importances")
    plt.bar(range(X_train.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
    plt.xticks(range(X_train.shape[1]), ff[indices], rotation=90)
    plt.xlim([-1, X_train.shape[1]])
    plt.show()


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

    return pd.concat([data, pd.get_dummies(data['travel_from'], prefix='travel_from')], axis=1).drop(['travel_from'], axis=1)


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
data = data[data['counts'] < 15]
train = processing_pipeline(data)

# import and handle test data
url = 'C:\\Users\\brendon.pitcher\\Documents\\Brendon\\Dev\\PlayTime\\Zindi\\TrafficJam\\test_questions.csv'
test = pd.read_csv(url)
X_test = processing_pipeline(test)
X_test.drop(['travel_to'],axis=1, inplace=True)

# add columns to test set if they dont exist after feature encoding
if ((len(train.columns)) > (len(X_test.columns) + 1)):
    missing_columns = list(set(train.columns) - set(X_test.columns) - set(['counts']))

    for miss in missing_columns:
        X_test[miss] = 0

# reorder columns
X_test = X_test[train.drop('counts', axis=1).columns]

# create X/Y training and validation sets
X_train = train.drop(['counts'], axis=1)
y_train = train['counts']

X_train, X_test, y_train, y_test = model_selection.train_test_split(X_train, y_train, test_size=0.25, shuffle=False)


# -----------------------
# --- training model ----
# -----------------------

models=[ 
        ######## First level ########
        [
            #BaggingRegressor(RandomForestRegressor(n_estimators=150, criterion="mae", max_depth=50, max_features=0.7), max_samples=0.5, max_features=0.5),
            #BaggingRegressor(RandomForestRegressor(n_estimators=50, criterion="mae", max_depth=100, max_features=0.7), max_samples=0.5, max_features=0.5),
            #SVR(max_iter=-1, kernel='rbf'),
            #KNeighborsRegressor(n_neighbors=2),
            #KNeighborsRegressor(n_neighbors=5),
            #MLPRegressor(hidden_layer_sizes=(100, 100), learning_rate='adaptive', learning_rate_init=0.01, alpha=0.000001),
            #MLPRegressor(hidden_layer_sizes=(50, 50), learning_rate_init=0.01, alpha=0.00001),
            #MLPRegressor(hidden_layer_sizes=(25, 25, 25), learning_rate_init=0.01, alpha=0.00001),
            #XGBRegressor(n_estimators=100, criterion="mae", max_depth=25, subsample=0.5, learning_rate=0.02, colsample_bytree=0.9),
            ExtraTreesRegressor(n_estimators=120, criterion="mae", max_depth=10, max_features=0.5, random_state=1),
            #GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=50, max_features=0.5, random_state=1),
            #RandomForestRegressor(n_estimators=150, criterion="mae", max_depth=50, max_features=0.7, random_state=1, oob_score=True),
        ],
        ######## Second level ########
        #[
            #XGBRegressor(n_estimators=100, criterion="mae", max_depth=5, max_features=0.5, random_state=1),
        #]
        # [
        #     XGBRegressor(n_estimators=100, criterion="mae", max_depth=3, max_features=0.5, random_state=1),
        # ]
    ]


from pystacknet.pystacknet import StackNetRegressor

#model = ExtraTreesRegressor(n_estimators=120, criterion="mae", max_depth=10, max_features=0.5, random_state=1)
model = StackNetRegressor(models, metric="mae", folds=2, restacking=False, use_retraining=False, random_state=12345, verbose=1)

model.fit(X_train, y_train)


#plot_feature_importance(X_train, model)

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

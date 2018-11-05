import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import sys
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold, cross_val_score
from rgf.sklearn import RGFRegressor


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

model = RGFRegressor(max_leaf=4500, algorithm="RGF_Sib", test_interval=50, loss="LS", verbose=False)

scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=5)
print(sum(scores) / len(scores))

#print(X.head(5))
#model.fit(X, y)
#preds_train_set = model.predict(X_test)

#print(mean_absolute_error(preds_train_set, y_test))

sys.exit()

# ----------------------
# ---- Test dataset ----
# ----------------------

df_test_set = pd.read_csv('test_questions.csv', low_memory=False)

df_test_set.drop(['travel_to'], axis=1, inplace=True)
df_test_set = df_test_set.sort_values('travel_date', ascending=False)

df_test_set["travel_date"] = pd.to_datetime(df_test_set["travel_date"],infer_datetime_format=True)
df_test_set["travel_date"] = df_test_set["travel_date"].dt.dayofweek

df_test_set["car_type"] = pd.Categorical(df_test_set["car_type"], categories=car_type_categories)
df_test_set["car_type"] = df_test_set.car_type.cat.codes

df_test_set["travel_from"] = pd.Categorical(df_test_set["travel_from"], categories=travel_from_categories)
df_test_set["travel_from"] = df_test_set.travel_from.cat.codes

df_test_set["travel_time"] = df_test_set["travel_time"].str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))
df_test_set['is_weekend'] = np.where(df_test_set['travel_date'] >= 5, 1, 0)

X_test = df_test_set.drop(['ride_id', 'max_capacity'], axis=1)

print(X_test.head(5))

test_set_predictions = model.predict(X_test)

d = {'ride_id': df_test_set["ride_id"], 'number_of_ticket': test_set_predictions}
df_predictions = pd.DataFrame(data=d)
df_predictions = df_predictions[['ride_id','number_of_ticket']]

df_predictions.to_csv('results.csv', index=False)



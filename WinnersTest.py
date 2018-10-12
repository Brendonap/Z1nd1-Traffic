import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import sys
from pystacknet.pystacknet import StackNetRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import model_selection
from xgboost import XGBRegressor

url = 'C:\\Users\\brendon.pitcher\\Documents\\Brendon\\Dev\\PlayTime\\Zindi\\TrafficJam\\train_revised.csv'
df = pd.read_csv(url)
columns = ['ride_id', 'travel_date', 'travel_time', 'travel_from', 'car_type', 'max_capacity']
df_train_set = df.groupby(columns).size().reset_index(name='number_of_tickets')
df_train_set = df_train_set.sort_values('travel_date', ascending=False)

df_train_set.drop(['ride_id'], axis=1, inplace=True) #ride_id is unnecessary in training set

df_train_set["travel_date"] = pd.to_datetime(df_train_set["travel_date"],infer_datetime_format=True)
df_train_set["travel_date"] = df_train_set["travel_date"].dt.dayofweek #change the full date to day of week

df_train_set["car_type"] = pd.Categorical(df_train_set["car_type"])
car_type_categories = df_train_set.car_type.cat.categories
df_train_set["car_type"] = df_train_set.car_type.cat.codes

df_train_set["travel_from"] = pd.Categorical(df_train_set["travel_from"])
travel_from_categories = df_train_set.travel_from.cat.categories
df_train_set["travel_from"] = df_train_set.travel_from.cat.codes

#express travel time in minutes
df_train_set["travel_time"] = df_train_set["travel_time"].str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))

# ------ model

X = df_train_set.drop(["number_of_tickets"], axis=1)
y = df_train_set.number_of_tickets  

#X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, shuffle=False)

model = RandomForestRegressor(n_estimators=100, criterion="mae", n_jobs=-1)

from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error')
print(sum(scores) / len(scores))

model.fit(X,y)

preds_train_set = model.predict(X)

print(mean_absolute_error(preds_train_set,y))



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

X_test = df_test_set.drop(['ride_id'], axis=1)
test_set_predictions = model.predict(X_test)

d = {'ride_id': df_test_set["ride_id"], 'number_of_ticket': test_set_predictions}
df_predictions = pd.DataFrame(data=d)
df_predictions = df_predictions[['ride_id','number_of_ticket']]

df_predictions.to_csv('results.csv', index=False)



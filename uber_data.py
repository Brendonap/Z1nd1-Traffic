import pandas as pd
import datetime
import math
import numpy as np
import sys
import sqlite3


# --------------------------
# ---- Helper functions ----
# --------------------------

url = 'C:\\Users\\brendon.pitcher\\Documents\\Brendon\\Dev\\PlayTime\\Zindi\\TrafficJam\\train_revised.csv'
df = pd.read_csv(url)
columns = ['ride_id', 'travel_date', 'travel_time', 'travel_from', 'car_type']
df_train_set = df.groupby(columns).size().reset_index(name='number_of_tickets')
df_train_set = df_train_set.sort_values('travel_date', ascending=False)

df_train_set["travel_date"] = pd.to_datetime(df_train_set["travel_date"], infer_datetime_format=True)
df_train_set.drop(['ride_id'], axis=1, inplace=True) #ride_id is unnecessary in training set

# uber imports
base = 'C:\\Users\\brendon.pitcher\\Documents\\Brendon\\Dev\\PlayTime\\Zindi\\TrafficJam\\'
urls = [
    'Data\\WestLands_2017-4-6_daily.csv',
    # 'Data\\WestLands_2017-7-9_week_avg.csv',
    # 'Data\\WestLands_2017-10-12_week_avg.csv',
    # 'Data\\WestLands_2018-1-3_week_avg.csv',
    # 'Data\\WestLands_2018-4-6_week_avg.csv',
    # 'Data\\WestLands_2017-1-3_week_avg.csv'
]

uber = pd.DataFrame()
for url in urls:
    df_url = base + url
    data = pd.read_csv(df_url)
    uber = pd.concat([uber, data])

uber = uber[[
       'Date',
       'Daily Mean Travel Time (Seconds)',
]]

# print(uber['Date Range'].head(50))

uber.columns = ['Date', 'uber_travel_time']
uber["Date"] = pd.to_datetime(uber["Date"],infer_datetime_format=True)

# def cat_day_of_week(x):
#     day = x['day_of_week']
#     if day == 'Monday':
#         return 0
#     elif day == 'Tuesday':
#         return 1
#     elif day == 'Wednesday':
#         return 2
#     elif day == 'Thursday':
#         return 3
#     elif day == 'Friday':
#         return 4
#     elif day == 'Saturday':
#         return 5
#     elif day == 'Sunday':
#         return 6


# uber['day_of_week'] = uber.apply(cat_day_of_week, axis=1)

ub_future = uber[uber['date_start'] > '2017-06-29']
ub_future = ub_future[ub_future['date_end'] < '2018-01-01']
ub_future['date_start'] = ub_future['date_start'] - pd.DateOffset(years=-1)
ub_future['date_end'] = ub_future['date_end'] - pd.DateOffset(years=-1)
uber = pd.concat([uber, ub_future])


df_train_set["travel_date"] = pd.to_datetime(df_train_set["travel_date"], infer_datetime_format=True)

df_train_set["day_of_week"] = df_train_set["travel_date"].dt.dayofweek #change the full date to day of week
df_train_set['is_weekend'] = np.where(df_train_set['day_of_week'] >= 5, 1, 0)

#Make the db in memory
conn = sqlite3.connect(':memory:')
#write the tables
df_train_set.to_sql('train', conn, index=False)
uber.to_sql('uber', conn, index=False)

qry = '''
    select  
        train.*,
        uber.mean_travel_day,
        uber.lower_travel_day,
        uber.upper_travel_day
    from
        train
    inner join uber on train.day_of_week = uber.day_of_week 
      AND train.travel_date between uber.date_start and uber.date_end
    '''

df_train_set = pd.read_sql_query(qry, conn)

df_train_set["car_type"] = pd.Categorical(df_train_set["car_type"])
car_type_categories = df_train_set['car_type'].cat.categories
df_train_set["car_type"] = df_train_set['car_type'].cat.codes

df_train_set["travel_from"] = pd.Categorical(df_train_set["travel_from"])
travel_from_categories = df_train_set['travel_from'].cat.categories
df_train_set["travel_from"] = df_train_set['travel_from'].cat.codes

#express travel time in minutes
df_train_set["travel_time"] = df_train_set["travel_time"].str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))

# def concat_travel_columns(x):
#     time = x['travel_time'] / 60

#     if time >= 7 and time < 10:
#         return 1, x['am']
#     elif time >= 10 and time < 14:
#         return 2, x['mid']
#     elif time >= 14 and time < 19:
#         return 3, x['pm']
#     elif time >= 19 and time <= 24:
#         return 4, x['eve']
#     elif time >= 1 and time < 7:
#         return 5, x['morn']
#     else:
#         return 6, 6


# df_train_set['day_part'], df_train_set['travel_dense'] = zip(*df_train_set.apply(concat_travel_columns, axis=1))

# df_train_set.drop(['am', 'pm', 'mid', 'eve', 'morn', 'daily', 'day_part'], axis=1, inplace=True)

from sklearn.preprocessing import StandardScaler 

df_train_set['range'] = df_train_set['upper_travel_day'] - df_train_set['lower_travel_day']  

scaled_features = df_train_set.copy()
col_names = ['mean_travel_day', 'range']

features = scaled_features[col_names]
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)

df_train_set[col_names] = features


# ------ model
X = df_train_set.drop(["number_of_tickets", 'travel_date', 'upper_travel_day', 'lower_travel_day'], axis=1)
y = df_train_set['number_of_tickets']


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, shuffle=True)


# ------------------------------------------------------------------------------ MODEL

## bagging example
df_test_set = pd.read_csv('test_questions.csv', low_memory=False)

df_test_set.drop(['travel_to'], axis=1, inplace=True)
df_test_set = df_test_set.sort_values('travel_date', ascending=False)

df_test_set["travel_date"] = pd.to_datetime(df_test_set["travel_date"],infer_datetime_format=True)
df_test_set["day_of_week"] = df_test_set["travel_date"].dt.dayofweek

df_test_set.to_sql('test', conn, index=False)

qry = '''
    select  
        test.*,
        uber.mean_travel_day,
        uber.lower_travel_day,
        uber.upper_travel_day
    from
        test
    inner join uber on test.day_of_week = uber.day_of_week 
      AND test.travel_date between uber.date_start and uber.date_end
    '''

# df_test_set = pd.read_sql_query(qry, conn)

# df_test_set["car_type"] = pd.Categorical(df_test_set["car_type"], categories=car_type_categories)
# df_test_set["car_type"] = df_test_set['car_type'].cat.codes

# df_test_set["travel_from"] = pd.Categorical(df_test_set["travel_from"], categories=travel_from_categories)
# df_test_set["travel_from"] = df_test_set['travel_from'].cat.codes

# df_test_set["travel_time"] = df_test_set["travel_time"].str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))
# df_test_set['is_weekend'] = np.where(df_test_set['day_of_week'] >= 5, 1, 0)

# df_test_set['range'] = df_test_set['upper_travel_day'] - df_test_set['lower_travel_day']  


# X_test = df_test_set.drop(['ride_id', 'max_capacity', 'travel_date', 'lower_travel_day', 'upper_travel_day'], axis=1)
# X_test = X_test[X_train.columns]


import pandas as pd
import datetime
import math
import numpy as np
import sys
import sqlite3
from sklearn.preprocessing import MinMaxScaler  

def get_uber_data():
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
        'Data\\WestLands_2017-7-9_daily.csv',
        'Data\\WestLands_2017-10-12_daily.csv',
        'Data\\WestLands_2018-1-3_daily.csv',
        'Data\\WestLands_2018-4-6_daily.csv',
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


    uber.columns = ['Date', 'uber_travel_time']
    uber["Date"] = pd.to_datetime(uber["Date"],infer_datetime_format=True)

    ub_future = uber[(uber['Date'] > '2017-06-30') & (uber['Date'] < '2018-01-01')]
    ub_future['Date'] = ub_future['Date'] - pd.DateOffset(years=-1)

    uber = pd.concat([uber, ub_future])

    scaler = MinMaxScaler()
    uber['uber_travel_time_scaled'] = scaler.fit_transform(uber[['uber_travel_time']])
    uber.drop('uber_travel_time', axis=1, inplace=True)
    return uber



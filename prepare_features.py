import pandas as pd
import datetime
import math
import numpy as np

from sklearn import model_selection
from sklearn.metrics import mean_squared_error, mean_absolute_error
# from xgboost import XGBRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, LinearRegression, BayesianRidge, SGDRegressor

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold, cross_val_score

from sklearn.neural_network import MLPRegressor
# from pystacknet.pystacknet import StackNetRegressor
# from rgf.sklearn import RGFRegressor
import sys


# --------------------------
# ---- Helper functions ----
# --------------------------

class get_data:

    def get_training_data(self):
        url = 'C:\\Users\\brendon.pitcher\\Documents\\Brendon\\Dev\\PlayTime\\Zindi\\TrafficJam\\train_revised.csv'
        df = pd.read_csv(url)
        columns = ['ride_id', 'travel_date', 'travel_time', 'travel_from', 'car_type']
        df_train_set = df.groupby(columns).size().reset_index(name='number_of_tickets')
        #df_train_set = df_train_set.sort_values('travel_date', ascending=False)

        #ride_id is unnecessary in training set
        # df_train_set.drop(['ride_id'], axis=1, inplace=True) 

        # convert travel date to datetime
        df_train_set = df_train_set.replace(['Kendu Bay', 'Oyugis', 'Keumbu'], 'other')
        df_train_set["travel_date"] = pd.to_datetime(df_train_set["travel_date"],infer_datetime_format=True)

        # remove month 2 and 3 2017
        # mask = (df_train_set["travel_date"] >= '2017-04-01')
        # df_train_set = df_train_set.loc[mask]

        #change the full date to day of week
        df_train_set["month"] = df_train_set["travel_date"].dt.month 
        df_train_set["week"] = df_train_set["travel_date"].dt.week
        df_train_set["travel_date"] = df_train_set["travel_date"].dt.dayofweek 

        # encode car type
        df_train_set["car_type"] = pd.Categorical(df_train_set["car_type"])
        self.car_type_categories = df_train_set.car_type.cat.categories
        df_train_set["car_type"] = df_train_set.car_type.cat.codes

        # encode travel_from
        df_train_set["travel_from"] = pd.Categorical(df_train_set["travel_from"])
        self.travel_from_categories = df_train_set.travel_from.cat.categories
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
        return X, y


    def get_test_data(self):
        # --------------------------------------------------------------------------------Test Set
        df_test_set = pd.read_csv('test_questions.csv', low_memory=False)

        df_test_set.drop(['travel_to'], axis=1, inplace=True)
        df_test_set = df_test_set.sort_values('travel_date', ascending=False)

        df_test_set = df_test_set.replace(['Kendu Bay', 'Oyugis', 'Keumbu'], 'other')
        df_test_set["travel_date"] = pd.to_datetime(df_test_set["travel_date"],infer_datetime_format=True)
        df_test_set["month"] = df_test_set["travel_date"].dt.month 
        df_test_set["week"] = df_test_set["travel_date"].dt.week 

        df_test_set["travel_date"] = df_test_set["travel_date"].dt.dayofweek

        df_test_set["car_type"] = pd.Categorical(df_test_set["car_type"], categories=self.car_type_categories)
        df_test_set["car_type"] = df_test_set.car_type.cat.codes

        df_test_set["travel_from"] = pd.Categorical(df_test_set["travel_from"], categories=self.travel_from_categories)
        df_test_set["travel_from"] = df_test_set.travel_from.cat.codes

        # df_test_set["travel_time"] = df_test_set["travel_time"].str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))
        df_test_set["travel_time"] = pd.to_datetime(df_test_set["travel_time"],infer_datetime_format=True)
        df_test_set["hour_booked"] = df_test_set["travel_time"].dt.hour
        df_test_set["minute_booked"] = df_test_set["travel_time"].dt.minute
        df_test_set.drop('travel_time', axis=1, inplace=True)

        df_test_set['is_weekend'] = np.where(df_test_set['travel_date'] >= 5, 1, 0)

        X_test = df_test_set.drop(['max_capacity'], axis=1)
        return X_test
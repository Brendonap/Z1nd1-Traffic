from prepare_features import get_data
from sklearn import model_selection
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from catboost import CatBoostRegressor

import pandas as pd
import numpy as np

# training data
data = get_data()
X, y = data.get_training_data()
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, shuffle=True, random_state=1)

model = CatBoostRegressor(loss_function='MAE', iterations=200, depth=15, learning_rate=0.5)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(mean_absolute_error(y_test, np.round(predictions)))

# test data
# X_test = data.get_test_data()


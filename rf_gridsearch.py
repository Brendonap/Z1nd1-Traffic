from prepare_features import get_data
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

import pandas as pd

# training data
data = get_data()
X, y = data.get_training_data()
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, shuffle=True, random_state=1)


if __name__ == '__main__':
    model = RandomForestRegressor(criterion='mae')
    parameters = {
        'n_estimators': [95, 97, 100, 105, 110],
        'max_depth': [8, 9, 10, 11, 12],
        'min_samples_split': [7, 8, 9, 10, 11],
        'min_samples_leaf': [1],
        'min_impurity_decrease': [0.0005],
        'oob_score': [True],
    }

    grid = GridSearchCV(model,parameters,cv=5,scoring='neg_mean_absolute_error', n_jobs = 6,verbose=True)

    grid.fit(X, y)
    #predictions = xgb_grid.predict(X)
    #test_predict = xgb_grid.predict(X_test)

    print(grid.best_estimator_)
    print(grid.best_score_)


# test data
# X_test = data.get_test_data()


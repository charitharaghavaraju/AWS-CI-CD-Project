import os
import sys

import pandas as pd
import numpy as np
import dill

from src.exception import CustomException

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file:
            dill.dump(obj, file)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(x_train, y_train, x_test, y_test, models, params):
    try:

        model_report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model_params = params[list(models.keys())[i]]

            model_gs = GridSearchCV(model, model_params, cv=3, n_jobs=-1, verbose=1)
            model_gs.fit(x_train, y_train)
            
            model.set_params(**model_gs.best_params_)
            model.fit(x_train, y_train)


            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            model_report[list(models.keys())[i]] = test_model_score

        return model_report
    
    except Exception as e:
        raise CustomException(e, sys)
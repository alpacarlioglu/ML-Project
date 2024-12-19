import os
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score
# from src.components import model_trainer 

def save_object(file_path, obj):
    '''
    open(file_path, 'wb'): This opens the file at the given file_path for writing in binary mode ('wb'). 
    The file object is assigned to file_obj.
    dill.dump(obj, file_obj): This serializes the obj (the Python object) and writes it to file_obj. 
    dill.dump is used to save the object to the file, allowing it to be later restored using dill.load. 
    This is especially useful when you want to save complex Python objects 
    (such as trained models, functions, etc.) to disk.
        '''
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for model_name, model in models.items(): 
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            report[model_name] = test_model_score

        return report
    
    except Exception as e:
        raise CustomException(e, sys)

import pandas as pd
import numpy as np
import sys
import os
from src.exception import CustomeException
import dill

def save_objects(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok= True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        

    except Exception as e:
        raise CustomeException(e, sys)

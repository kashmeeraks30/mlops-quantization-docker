import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import os
from joblib import dump,load
from sklearn.preprocessing import StandardScaler

def train_model(X,y):
    model=LinearRegression()
    #model.fit(X,y)
    scaler = StandardScaler()
    X_normalize_train = scaler.fit_transform(X)
    #X_normalize_test = scaler.transform(X)
    model.fit(X_normalize_train,y)
    print(f"Coefficient size:{model.coef_.shape}")
    print(f"Intercept {model.intercept_}")
    return model

def save_model(model):
    os.makedirs('Models',exist_ok=True)
    joblib.dump(model,'Models/California_Housing_Model.joblib')

def load_model(modelpath="Models/California_Housing_Model.joblib"):
    model=joblib.load(modelpath)
    return(model)

def extract_parameters(model):
    coef=model.coef_
    intercept=model.intercept_
    return coef,intercept

def store_save_unquantized_param(coef,intercept):
    unquantized_param={'coef':coef,'intercept':intercept}

    joblib.dump(unquantized_param,'Models/unquant_params.joblib')
    print("unquantized param dictionary stored as joblib!")
    print(os.path.getsize('Models/unquant_params.joblib'))
    return unquantized_param
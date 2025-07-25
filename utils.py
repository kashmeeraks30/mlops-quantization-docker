import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import os
from joblib import dump,load

def train_model(X,y):
    model=LinearRegression()
    model.fit(X,y)
    print(f"Coefficient size:{model.coef_.shape}")
    print(f"Intercept {model.intercept_}")
    return model

def save_model(model):
    os.makedirs('Models',exist_ok=True)
    joblib.dump(model,'Models/California_Housing_Model.joblib')

def load_model(modelpath="/app/Models/California_Housing_Model.joblib"):
    model=joblib.load(modelpath)
    return(model)
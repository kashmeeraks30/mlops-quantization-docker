import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
from joblib import dump,load

def train_model(X,y):
    model=LinearRegression()
    model.fit(X,y)
    print(f"Coefficient size:{model.coef_.shape}")
    print(f"Intercept {model.intercept_}")
    return model

def save_model(model):
    joblib.dump(model,'California_Housing_Model')
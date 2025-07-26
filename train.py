from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from utils import train_model,save_model
import joblib
from sklearn.preprocessing import StandardScaler

def main():
    housingdata=fetch_california_housing()
    X,y=housingdata.data,housingdata.target
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    model=train_model(X_train,y_train)
    save_model(model)

if __name__=="__main__":
    main()
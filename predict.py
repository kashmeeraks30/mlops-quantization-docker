from utils import load_model
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score,mean_squared_error


def main():
    housingdata=fetch_california_housing()
    X,y=housingdata.data,housingdata.target
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    model=load_model()

    y_predicted=model.predict(X_test)
    
    r_squared=r2_score(y_test,y_predicted)
    mse=mean_squared_error(y_test,y_predicted)


    print(f"R squared value:{r_squared}")
    print(f"Mean Squared Error:{mse}")

if __name__=="__main__":
    main()
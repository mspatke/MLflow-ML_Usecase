import os 
import argparse
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

def get_data():

    URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    try:
        dataframe = pd.read_csv(URL, sep= ";")
        return dataframe
    except Exception as e:
        raise e



def evaluate(y_test, y_pred):
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return rmse, mae, r2



def main(alpha, l1_ratio):

    df = get_data()

    X = df.drop(["quality"],axis=1)
    y = df[["quality"]]

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=123)
    
    with mlflow.start_run():

        lr = ElasticNet(alpha= alpha,l1_ratio= l1_ratio, random_state=42)
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        rmse , mae, r2 = evaluate(y_test,y_pred )

        print(f"Elastic net parameters : alpha: {alpha} ,l1_ratio:{l1_ratio}")
        print(f"Elastic net: rmse:{rmse}, mae:{mae}, R2{r2}")

        mlflow.log_param("alpha",alpha)
        mlflow.log_param("l1_ratio",l1_ratio)

        mlflow.log_metric("MSE",rmse)
        mlflow.log_metric("MAE",mae)
        mlflow.log_metric("R2 Score",r2)

        mlflow.sklearn.log_model(lr,"ML_Model")



if __name__=="__main__":
    
    args= argparse.ArgumentParser()
    args.add_argument("--alpha","-a", type= float,default= 0.5)
    args.add_argument("--l1_ratio","-l1",type=float,default=0.5)
    parsed_args =args.parse_args()
    
    try:
        main(alpha=parsed_args.alpha, l1_ratio = parsed_args.l1_ratio)
    except Exception as e:
        raise e
    
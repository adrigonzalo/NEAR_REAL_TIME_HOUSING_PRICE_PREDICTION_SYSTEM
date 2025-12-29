"""
Here we are going to develop the code to train the model
"""

import pandas as pd
import numpy as np  
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import os
import sys

from models import RandomForestModel

# We need to add 'src' to the path se we can import the local modules
sys.path.append(os.path.join(os.getcwd(), 'src'))


# Function to train the model
def train_model():

    # 1. Load processed data
    print(" --- STARTING TRAINING (OOP) --- ")
    train_path = os.path.join('data', 'processed', 'train_processed.csv')
    df = pd.read_csv(train_path)

    # Select only the numeric columns
    df_numeric = df.select_dtypes(include=[np.number])

    # Calculate the median filling the nan columns
    df_numeric = df_numeric.fillna(df_numeric.median())

    X = df_numeric.drop(['SalePrice'], axis=1)
    y = df_numeric['SalePrice']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. MLflow Configuration: Remote connection
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("Housing_Price_Prediction_OOP")

    # 3. Start the experiment in MLflow.
    with mlflow.start_run(run_name="BaseLine_Linear_Regression_OOP"):

        # INSTANCE MODELS CLASS
        # 1. Initializing the model object
        trainer = RandomForestModel(n_estimators=150, max_depth=15)


        # 2. Use their methos
        trainer.train(X_train, y_train)
        predictions = trainer.predict(X_test)

        # 3. Predict and evaluate
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        print(f" Trained model. RMSE: {rmse:.2f}, R2: {r2:.2f}")

        # 4. Register parameters and metrics in MLflow
        mlflow.log_params(trainer.get_params())
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)


        mlflow.sklearn.log_model(

            sk_model = trainer.model,
            name = "model",
            registered_model_name = "RealEstate_RF_OOP"
        )


if __name__ == "__main__":
    train_model()
"""
Here we are going to develop the code to train the model
"""

import pandas as pd
import numpy as np  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import os

# Function to train the model
def train_model():

    # Remote connection
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("Housing_Price_Prediction")


    # 1. Load and read processed data
    train_path = os.path.join('data', 'processed', 'train_processed.csv')
    df = pd.read_csv(train_path)

    # 2. Simple selection of mumeric variable for the baseline.
    X = df.select_dtypes(include=[np.number]).drop(['SalePrice'], axis=1)
    y = df['SalePrice']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Start the experiment in MLflow.
    with mlflow.start_run(run_name="BaseLine_Linear_Regression"):

        # Train
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict and evaluate
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        print(f" Trained model. RMSE: {rmse:.2f}, R2: {r2:.2f}")

        # 4. Register parameters and metrics in MLflow
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # Save model
        mlflow.sklearn.log_model(model, name = "model")


if __name__ == "__main__":
    train_model()
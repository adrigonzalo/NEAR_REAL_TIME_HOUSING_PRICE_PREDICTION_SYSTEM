
"""
Here we have all the models that we will use in the project
"""

# LIBRARIES
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import RegressorMixin
import joblib
from model_interface import ModelInterface


# This class inheritance. RandomForestModel is a class type of ModelInterface
class RandomForestModel(ModelInterface):

    def __init__(self, n_estimators: int = 100, max_depth: int = 10):

        self.model_name = 'Random_Forest_Regressor'

        self.model = RandomForestRegressor(

            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state = 42
        )

    # Train models
    def train(self, X: pd.DataFrame, y: pd.Series):

        print(f"Training {self.model_name}...")
        self.model.fit(X, y)


    # Generate predictions
    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)
    

    # Save the model object using joblib
    def save(self, path: str):

        joblib.dump(self.model, path)
        print(f" Model saved in {path}")

    
    # Return the parameters to register in MLflow.
    def get_params(self):

        return self.model.get_params()


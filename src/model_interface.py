from abc import ABC, abstractmethod
import pandas as pd


# DEFINE METHOD WHICH MUST HAVE ANY MODEL
class ModelInterface(ABC):

    @abstractmethod
    
    # Method responsible for train the model.
    def train(self, X: pd.DataFrame, y: pd.Series):
        pass


    @abstractmethod
    
    # Method responsible for generating predictions.
    def predict(self, X: pd.DataFrame):
        pass


    @abstractmethod
    
    # Method responsible for save model.
    def save(self, path: str):
        pass




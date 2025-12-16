
"""
In the data ingestion script we would have a database connection or a Data Late. 
In this code, we will simulated it just reading the train and test CSVs.

"""

# LIBRARIES
import pandas as pd
import os

# Root define.
DATA_PATH = os.path.join(os.getcwd(), 'data', 'raw')
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'

# FUNCTION TO LOAD A CSV FIELD FROM THE data/raw FOLDER
def load_data(file_name: str) -> pd.DataFrame:

    try:

        file_path = os.path.join(DATA_PATH, file_name)
        df = pd.read_csv(file_path)
        print(f"Field {file_name} loaded successfully. Dimensions: {df.shape}")
        return df

    except FileNotFoundError:
        print(f"Error: {file_name} field not found in {DATA_PATH}.")
        raise

    except Exception as e:
        print(f"Error loading the field {file_name}: {e}")
        raise


if __name__ == "__main__":
    print("--- RUNNING DATA INGESTION ---")
    train_df = load_data(TRAIN_FILE)
    test_df = load_data(TEST_FILE)
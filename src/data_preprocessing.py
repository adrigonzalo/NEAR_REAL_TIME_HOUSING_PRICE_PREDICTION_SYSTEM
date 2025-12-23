
"""
In this script we will have the preprocessing and data cleaning code.
It is neccessary to have this code encapsulated in classes or functions
"""

# LIBRARIES
import pandas as pd
import numpy as np
import os

from data_ingestion import load_data, TRAIN_FILE, TEST_FILE

PROCESSED_DATA_PATH = os.path.join(os.getcwd(), 'data', 'processed')


# FUNCTION TO PERFORM AN INITIAL CLEANING: NaNs common values and deleting columns with too much NaNs.
"""
We used the Type Hinting syntax defining the function:
- df It is the variable name (argument) that the function receives.
- : pd.DataFrame: This is the hint. It means that we are waiting that the df will be a Pandas Dataframe object.
- -> pd.DataFrame: The arrow indicates what the function returns
"""
def initial_cleaning(df: pd.DataFrame) -> pd.DataFrame:

    print("--- STARTING CLEANING AND PREPROCESSING ---")

    # 1. Treating NaNs as cathegoric variables which means "None".
    for col in ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 
                'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
                'BsmtFinType2', 'MasVnrType', 'MSZoning', 'Utilities',
                'Exterior1st', 'Exterior2nd', 'Cultura', 'Functional',
                'KitchenQual', 'SaleType']:
        if col in df.columns:
            df[col] = df[col].fillna('None')

    # 2. Impute NaNs of numerical variables with the median
    for col in ['LotFrontage', 'GarageYrBlt', 'MasVnrArea']:

        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # 3. Impute NaNs of cathegoric variable with the mode.
    for col in ['Electrical']:

        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # 4. Delete columns with too much NaNs ( > 50% missing )
    if 'Id' in df.columns:
        df = df.drop(['Id'], axis=1)

    
    print(f" Initial cleaning completed. Remaining NaNs: {df.isnull().sum().sum()}")
    return df

# FUNCTION TO SAVE THE PROCESSED DATAFRAME IN THE data/processed
def save_processed_data(df: pd.DataFrame, file_name: str):

    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    file_path = os.path.join(PROCESSED_DATA_PATH, file_name)
    df.to_csv(file_path, index=False)
    print(f"Processed data save in: {file_path}")


# FUNCTION TO CREATE NEW FEATURES
def create_features(df):

    # Age of the house
    df['HouseAge'] = 2025 - df['YearBuilt']

    # Total area of the house
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

    # Total baths (completed baths + 0.5 of the toilets)
    df['Total Bathrooms'] = df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath']


    # Delete the original columns which we resume before
    cols_to_drop = ['YearBuilt', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'FullBath', 'HalfBath', 'BsmtFullBath']
    df = df.drop(columns=cols_to_drop)

    return df




if __name__ == "__main__":

    # 1. Load data
    train_df_raw = load_data(TRAIN_FILE)
    test_df_raw = load_data(TEST_FILE)

    # 2. Preprocess.
    train_df_processed = initial_cleaning(train_df_raw.copy())
    test_df_processed = initial_cleaning(test_df_raw.copy())

    # 3. Create new features
    create_features(train_df_processed)

    # Save
    save_processed_data(train_df_processed, 'train_processed.csv')
    save_processed_data(test_df_processed, 'test_processed.csv')
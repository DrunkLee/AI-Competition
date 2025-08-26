import pandas as pd
import numpy as np
import joblib

GENDER_MAP = {'M': 0, 'F': 1}
SUB_TYPE_MAP = {"member": 0, "plus": 1, "vip": 2}

class DataPreprocessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.file_path)
    
    def make_pickle(self, df: pd.DataFrame, pickle_path: str):
        joblib.dump(df, pickle_path)
        
    def load_pickle(self, pickle_path: str) -> pd.DataFrame:
        return joblib.load(pickle_path)
    
    def common_preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df["gender"] = df['gender'].map(GENDER_MAP)
        df["subscription_type"] = df["subscription_type"].map(SUB_TYPE_MAP)
        return df
    
    def run(self):
        df = self.load_data()
        df = self.common_preprocess(df)
        return df
    
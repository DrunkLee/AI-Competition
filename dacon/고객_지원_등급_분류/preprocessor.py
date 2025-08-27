import pandas as pd
import numpy as np
import joblib

GENDER_MAP = {'M': 0, 'F': 1}
SUB_TYPE_MAP = {"member": 0, "plus": 1, "vip": 2}

class DataPreprocessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        
        self.age_bins = [0, 19, 20, 39, 49, 59, 69, 200]
        self.age_labels = ['10s', '20s', '30s', '40s', '50s', '60s', "60+"]
    
        self.contract_bins = [0, 60, 180, 360, 2000]
        self.contract_labels = ["short", "medium", "long", 'long+']
        
        self.tenure_bins = [0, 5, 25, 60, 2000]
        self.tenure_labels = ["short", "medium", "long", 'long+']
        
    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.file_path)
    
    def make_pickle(self, df: pd.DataFrame, pickle_path: str):
        joblib.dump(df, pickle_path)
    
    def load_pickle(self, pickle_path: str) -> pd.DataFrame:
        return joblib.load(pickle_path)
    
    def __common_preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df["gender"] = df['gender'].map(GENDER_MAP)
        df["subscription_type"] = df["subscription_type"].map(SUB_TYPE_MAP)
        
        df['age_group'] = pd.cut(df['age'], bins=self.age_bins, labels=self.age_labels, right=False)
        df = pd.get_dummies(df, columns=["age_group"], prefix='age')
        
        df["contract_period"] = pd.cut(df["contract_length"], bins=self.contract_bins, labels=self.contract_labels, right=True)
        df = pd.get_dummies(df, columns=["contract_period"], prefix='contract')
        
        df["tenure_group"] = pd.cut(df["tenure"], bins=self.tenure_bins, labels=self.tenure_labels, right=True)
        df = pd.get_dummies(df, columns=["tenure_group"], prefix='tenure')
        
        df["total_usage_days"] = df["tenure"] * df["frequent"]
        df["frequent_per_tenure"] = df['frequent'] / df['tenure']
        df["tenure_countract_ratio"] = df['tenure'] / df['contract_length']
        df["age_tenure_interaction"] = df['age'] * df['tenure']
        df["inactivay_score"] = df["after_interaction"] / (df['frequent'] + 1)
        
        return df
    
    def run(self):
        df = self.load_data()
        df = self.__common_preprocess(df)
        return df
    
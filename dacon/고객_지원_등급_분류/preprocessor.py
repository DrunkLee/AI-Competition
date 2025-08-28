import pandas as pd
import joblib

GENDER_MAP = {'M': 1.0, 'F': 2.0}
SUB_TYPE_MAP = {"member": 1.0, "plus": 2.0, "vip": 3.0}

class DataPreprocessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.eps = 1e-8
        
        self.age_bins = [0, 19, 20, 39, 49, 59, 69]
        self.age_labels = ['10s', '20s', '30s', '40s', '50s', '60s']
        
        self.tenure_bins = [0, 5, 25, 60]
        self.tenure_labels = ["short", "medium", "long"]
        
    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.file_path)
    
    def make_pickle(self, df: pd.DataFrame, pickle_path: str):
        joblib.dump(df, pickle_path)
    
    def load_pickle(self, pickle_path: str) -> pd.DataFrame:
        return joblib.load(pickle_path)
    
    def __common_preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        age: 나이
        gender: 성별
        tenure: 가입 기간
        frequent: 사용 빈도
        payment_interval: 결제 주기
        subscription_type: 구독 유형
        contract_length: 계약 기간
        after_interaction: 마지막 활동 후 경과 시간
        """
        
        df['age_group'] = pd.cut(df['age'], bins=self.age_bins, labels=self.age_labels, right=False)
        df = pd.get_dummies(df, columns=["age_group"], prefix='age', dtype=int)
        
        df["tenure_group"] = pd.cut(df["tenure"], bins=self.tenure_bins, labels=self.tenure_labels, right=True)
        df = pd.get_dummies(df, columns=["tenure_group"], prefix='tenure', dtype=int)
        
        df = pd.get_dummies(df, columns=['gender', 'subscription_type', 'contract_length'], 
                            prefix=['gender', 'sub_type', 'contract'], dtype=int)
        
        """ interaction"""
        df["age_F"] = df["age"] * df["gender_F"]
        df["age_M"] = df["age"] * df["gender_M"]
        
        df["tunere_vip"] = df["tenure"] * df["sub_type_vip"]
        df["tenure_plus"] = df["tenure"] * df["sub_type_plus"]
        df["tenure_member"] = df["tenure"] * df["sub_type_member"]
        df = df.drop(columns=['age', 'tenure'])
        print(df)
        return df
    
    def run(self):
        df = self.load_data()
        df = self.__common_preprocess(df)
        return df
    
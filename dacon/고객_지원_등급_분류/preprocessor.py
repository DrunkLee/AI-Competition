import pandas as pd
import joblib

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
        # one-hot
        df['age_group'] = pd.cut(df['age'], bins=self.age_bins, labels=self.age_labels, right=False)
        df = pd.get_dummies(df, columns=["age_group"], prefix='age', dtype=int)
        
        df["tenure_group"] = pd.cut(df["tenure"], bins=self.tenure_bins, labels=self.tenure_labels, right=True)
        df = pd.get_dummies(df, columns=["tenure_group"], prefix='tenure', dtype=int)
        
        df = pd.get_dummies(df, columns=['gender', 'subscription_type'], 
                            prefix=['gender', 'sub_type'], dtype=int)
        
        """기본 피쳐"""
        # 나이 x 성별 x 사용 빈도
        df["age_F"] = df["age"] * df["gender_F"]
        df["age_M"] = df["age"] * df["gender_M"]
        df["age_F_frequent"] = df["age_F"] * df["frequent"]
        df["age_M_frequent"] = df["age_M"] * df["frequent"]
        
        # 가입 기간 x 구독 유형
        df["tunere_vip"] = df["tenure"] * df["sub_type_vip"]
        df["tenure_plus"] = df["tenure"] * df["sub_type_plus"]
        df["tenure_member"] = df["tenure"] * df["sub_type_member"]
        
        df["after_contract"] = df["after_interaction"] * df["contract_length"]
        df["age_contract"] = df["age"] * df["contract_length"]
        df["activity_per_tenure"] = df["frequent"] / (df["tenure"] + self.eps)
        df["contract_per_freq"] = df["contract_length"] / (df["frequent"] + self.eps)
        
        df = df.drop(columns=['age', 'tenure'])
        print(df.head(3))
        return df
    
    def run(self):
        df = self.load_data()
        df = self.__common_preprocess(df)
        return df
    
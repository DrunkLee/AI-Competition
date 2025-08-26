from xgboost import XGBClassifier
from metric import MacroF1Score
from sklearn.model_selection import train_test_split as tts
import pandas as pd

class XGBC:
    def __init__(self, seed: int = 42, params: dict = None):
        self.seed = seed
        if params is None:
            params = {
                'n_estimators': 1000,
                'learning_rate': 0.02,
                'max_depth': 10,
                'random_state': self.seed,
                'early_stopping_rounds': 30
                }
        self.model = XGBClassifier(**params)
        self.metric = MacroF1Score()
        
    def fit_all(self, df: pd.DataFrame, feature_cols: list, target_col: str):
        print("[INFO] 검증없이 학습을 시작합니다.")
        X = df[feature_cols]
        y = df[target_col]
        self.model.fit(X,y)
        
    def fit_tts(self, df: pd.DataFrame, feature_cols: list, target_col: str, threshold = 0.2):
        print("[INFO] TRAIN TEST SPLIT으로 검증을 시작합니다.")
        X = df[feature_cols]
        y = df[target_col]
        X_train, X_val, y_train, y_val = tts(X, y, test_size=threshold, random_state=self.seed, stratify=y)
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = self.model.predict(X_val)
        self.metric.update(y_val.to_list(), y_pred)
        score = self.metric.result()
        print(f"[INFO] Macro F1 Score: {score:.4f}")
        self.metric.reset()
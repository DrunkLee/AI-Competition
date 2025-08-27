from xgboost import XGBClassifier
from metric import MacroF1Score
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import os
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import optuna

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class XGBC:
    def __init__(self, seed: int = 42, params: dict = None, save_dir: str = None):
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
        self.scaler = None
        
        if save_dir:
            run_time = datetime.now().strftime("%Y%m%d_%H%M%S") + "_ensemble"
            self.save_path = os.path.join(save_dir, run_time)
            os.makedirs(self.save_path, exist_ok=True)
            print(f"[INFO] 모델 저장 경로: {self.save_path}")
        else:
            self.save_path = None
            print(f"[INFO] SAVE_DIR이 설정되지 않아 모델을 저장하지 않습니다.")
    
    def get_feature_importance(self, feature_cols: list, verbose: bool = True):
        if not hasattr(self.model, 'feature_importances_'):
            print("[ERROR] 모델이 학습되지 않았습니다.")
            return
        importance = self.model.feature_importances_
        importance_df = pd.DataFrame({'feature': feature_cols, 'importance': importance})
        importance_df = importance_df.sort_values(by='importance', ascending=False).reset_index(drop=True)
        print("\n[INFO] 피쳐 중요도:")
        for i, row in importance_df.iterrows():
            print(f"{i+1:2d}위: {row['feature']} {row['importance']:.4f}")
        print()
        return importance_df
    
    def plot_feature_importance(self, feature_cols: list, save_path: str = None):
        importance_df = self.get_feature_importance(feature_cols)
        if importance_df is None:
            raise ValueError("[ERROR] 모델이 학습되지 않아 종료합니다.")
        plt.figure(figsize=(15,10))
        sns.barplot(data=importance_df, x="importance", y='feature')
        plt.title("Feature_Importance")
        plt.xlabel("Importance")
        plt.ylabel('feature')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'feature_importance.png'))
        print("[INFO] 모델 피쳐 중요도 시각화가 저장되었습니다.")
    
    def fit_all(self, df: pd.DataFrame, feature_cols: list, target_col: str):
        print("[INFO] 검증없이 학습을 시작합니다.")
        X = df[feature_cols]
        y = df[target_col]
        self.model.fit(X,y)
    
    def fit_tts(self, df: pd.DataFrame, feature_cols: list, target_col: str, threshold = 0.2):
        print("[INFO] TRAIN TEST SPLIT으로 검증을 시작합니다.")
        print(f"[INFO] 학습에 사용할 feature 개수: {len(feature_cols)}")
        X = df[feature_cols]
        y = df[target_col]
        
        X_train, X_val, y_train, y_val = tts(X, y, test_size=threshold, random_state=self.seed, stratify=y)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
        X_val_scaled = scaler.transform(X_val)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_cols)
        
        self.model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
        y_pred = self.model.predict(X_val_scaled, iteration_range = (0, self.model.best_iteration + 1))
        
        self.metric.update(y_val.to_list(), y_pred)
        score = self.metric.result()
        print(f"[INFO] Macro F1 Score: {score:.4f}")
        
        self.metric.reset()
        
        if self.save_path:
            self.plot_feature_importance(feature_cols, self.save_path)
            model_path = os.path.join(self.save_path, 'xgbc_model.pkl')
            joblib.dump(self.model, model_path)
            print(f"[INFO] 모델이 저장되었습니다: {model_path}")
        else:
            self.get_feature_importance(feature_cols)
        print("[INFO] 학습이 완료되었습니다.")
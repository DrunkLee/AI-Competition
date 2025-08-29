from xgboost import XGBClassifier
from metric import MacroF1Score
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
import os
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
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
        model_path = os.path.join(self.save_path, 'xgbc_model.pkl')
        print(f"[INFO] 학습 데이터를 저장합니다. {model_path}")
        joblib.dump(self.model, model_path)
        self.plot_feature_importance(feature_cols=feature_cols, save_path=self.save_path)
    
    def fit_kfold(self, df: pd.DataFrame, feature_cols: list, target_col: str, n_splits = 3):
        print("[INFO] K-FOLD로 검증을 시작합니다.")
        X = df[feature_cols]
        y = df[target_col]
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        
        fold_scores = []
        feature_importances = []
        
        for fold,(train_idx, val_idx) in enumerate(skf.split(X,y)):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            fold_model = XGBClassifier(**self.model.get_params())
            
            fold_model.fit(X_train, y_train,
                           eval_set=[(X_val,y_val)],
                           verbose=False)
            
            fold_pred = fold_model.predict(X_val)
            
            self.metric.update(y_val.to_list(), fold_pred)
            score = self.metric.result()
            self.metric.reset()
            
            fold_scores.append(score)
            
            print(f"[INFO] Fold {fold} Macro F1 Score: {score:.4f}")
            
            fold_importance_df = pd.DataFrame({
                'feature': X_train.columns,
                'importance': fold_model.feature_importances_
            })
            feature_importances.append(fold_importance_df)
        mean_score = np.mean(fold_scores)
        print("="*50)
        print(f"[INFO] {n_splits}-Fold 교차 검증 평균 Macro F1 Score: {mean_score:.4f}")
        all_importances = pd.concat(feature_importances, axis=0)
        final_importance = all_importances.groupby('feature')['importance'].mean().sort_values(ascending=False).reset_index()
        
        print("\n[INFO] 최종 피처 중요도 (교차 검증 평균):")
        for i,row in final_importance.iterrows():
            print(f"{i+1:2d}위: {row['feature']} (평균 중요도: {row['importance']:.4f})")
    
    def find_best_params(self, df, feature_cols: list, target_col: str, n_trials : int = 50, n_splits: int = 3):
        print(f"[INFO] Optuna를 사용한 하이퍼파라미터 튜닝을 시작합니다. (n_trials={n_trials})")
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 500, 2500, step=100),
                'max_depth': trial.suggest_int('max_depth', 5, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.1),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0, step=0.1),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'lambda': trial.suggest_float('lambda', 0.1, 10, log=True),
                'alpha': trial.suggest_float('alpha', 0.1, 10, log=True),
                'random_state': self.seed,
                'early_stopping_rounds': 30,
                'n_jobs': -1
            }
            X = df[feature_cols]
            y = df[target_col]
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
            f1_scores = []

            for train_idx, val_idx in skf.split(X, y):
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
                
                model = XGBClassifier(**params)
                model.fit(X_train, y_train,
                          eval_set=[(X_val, y_val)],
                          verbose=False)
                
                pred = model.predict(X_val)
                self.metric.update(y_val.to_list(), pred)
                score = self.metric.result()
                self.metric.reset()
                f1_scores.append(score)
            return np.mean(f1_scores)
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        
        print("\n[INFO] 하이퍼파라미터 튜닝 완료!")
        print(f"Best trial Macro F1 Score: {study.best_value:.4f}")
        print("Best params: ")
        for key, value in study.best_params.items():
            print(f"{key}: {value}")
            
        self.model = XGBClassifier(**study.best_params)
        return study.best_params
    
    def predict(self, test_df: pd.DataFrame, feature_cols: list) -> list:
        print("\n[INFO] 테스트 데이터에 대한 예측을 시작합니다.")
        X_test = test_df[feature_cols]
        preds = self.model.predict(X_test)
        print(f"[INFO] 예측이 완료되었습니다. 예측된 샘플 개수: {len(preds)}")
        return preds
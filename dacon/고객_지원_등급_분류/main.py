from preprocessor import DataPreprocessor
from models import XGBC

train_path = "./data/train.csv"

pre = DataPreprocessor(train_path)
df = pre.run()

feature_cols = [col for col in df.columns if col not in ["ID", "support_needs"]]

print("="*50)
print(f"[INFO] 학습에 사용할 Feature 갯수: {len(feature_cols)}")
print("="*50)
target_col = "support_needs"

model = XGBC(seed=42)
model.fit_kfold(df,
                feature_cols = feature_cols,
                target_col = target_col,
                n_splits=3)
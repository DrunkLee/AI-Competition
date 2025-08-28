from preprocessor import DataPreprocessor
from models import XGBC

train_path = "./data/train.csv"

pre = DataPreprocessor(train_path)
df = pre.run()

feature_cols = [col for col in df.columns if col not in ["ID", "support_needs"]]
target_col = "support_needs"

model = XGBC(seed=42)
model.fit_tts(df,
                feature_cols = feature_cols,
                target_col = target_col,
                threshold=0.35)
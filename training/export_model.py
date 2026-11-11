from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "breast_cancer.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "breast_cancer_classifier.joblib"

data = pd.read_csv(DATA_PATH)
features = data.drop(columns=["target", "dataset_version"])
target = data["target"]

model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=1.0, max_iter=2000, random_state=42)),
    ]
)

model.fit(features, target)
MODEL_PATH.parent.mkdir(exist_ok=True)
joblib.dump(model, MODEL_PATH)

print(f"Exported trained model to {MODEL_PATH}")
print(f"Model expects {features.shape[1]} features")

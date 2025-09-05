# src/calibrate.py
import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from src.data import load_data
from src.preprocess import clean
from src.model import build_baseline, ModelConfig

# Load data again
df = load_data()
df["text"] = (df["title"].fillna("") + " " + df["description"].fillna("")).apply(clean)
X, y = df["text"], df["label"]

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Load trained pipeline
pipe = joblib.load("artifacts/models/model.joblib")
vec = pipe.named_steps["tfidf"]
clf = pipe.named_steps["clf"]

# Calibrate
cal = CalibratedClassifierCV(clf, cv=3, method="sigmoid")
cal.fit(vec.transform(X_train), y_train)

joblib.dump(cal, "artifacts/models/calibrated_clf.joblib")
print("Saved calibrated classifier to artifacts/models/calibrated_clf.joblib")

# Example: get probabilities for one text
text = "NASA launches new space telescope"
probs = cal.predict_proba(vec.transform([clean(text)]))[0]
print("Probabilities:", probs)

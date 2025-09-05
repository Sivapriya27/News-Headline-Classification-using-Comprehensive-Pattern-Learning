# src/predict_all.py
import joblib, pandas as pd
from src.preprocess import clean

pipe = joblib.load("artifacts/models/model.joblib")
df = pd.read_csv("data/ag_news.csv")

# preprocess
df['text'] = (df['title'].fillna("") + " " + df['description'].fillna("")).apply(clean)

# predictions
df['pred'] = pipe.predict(df['text'])

# save
df.to_csv("artifacts/predictions_all.csv", index=False)
print("Saved predictions to artifacts/predictions_all.csv")

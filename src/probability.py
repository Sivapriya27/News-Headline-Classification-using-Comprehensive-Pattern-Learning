import joblib
import pandas as pd
import matplotlib.pyplot as plt
from src.preprocess import clean

vec = joblib.load("artifacts/models/vectorizer.joblib")
cal = joblib.load("artifacts/models/calibrated_clf.joblib")

samples = [
    "Stock markets rally after inflation report",
    "Local team wins championship",
    "NASA announces new Mars mission",
    "UN condemns conflict in region"
]

for text in samples:
    probs = cal.predict_proba(vec.transform([clean(text)]))[0]
    plt.bar(["World","Sports","Business","Sci/Tech"], probs)
    plt.title(f"Probabilities: {text[:40]}...")
    plt.ylim(0,1)
    plt.savefig(f"artifacts/plots/probs_{text.split()[0]}.png")
    plt.close()
   
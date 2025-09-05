import json
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from .config import ARTIFACTS, MODELS_DIR, METRICS_DIR, PLOTS_DIR, RANDOM_STATE, TEST_SIZE
from .utils import ensure_dirs, save_json
from .data import load_data
from .preprocess import clean

def main():
    ensure_dirs(ARTIFACTS, MODELS_DIR, METRICS_DIR, PLOTS_DIR)
    model_path = MODELS_DIR / "model.joblib"
    pipe = joblib.load(model_path)

    df = load_data()
    df["text"] = (df["title"].fillna("") + " " + df["description"].fillna("")).apply(clean)
    X, y = df["text"].values, df["label"].values

    _, X_test, _, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred, digits=3)

    save_json({"accuracy": acc, "precision_weighted": pr, "recall_weighted": rc, "f1_weighted": f1},
              METRICS_DIR / "metrics.json")
    with open(METRICS_DIR / "classification_report.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix.png")
    plt.close()

    print("Saved metrics and confusion matrix.")

if __name__ == "__main__":
    main()

import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import ARTIFACTS, MODELS_DIR, METRICS_DIR, PLOTS_DIR, RANDOM_STATE, TEST_SIZE, VAL_SIZE
from .utils import ensure_dirs, save_json, timestamp
from .data import load_data
from .preprocess import clean
from .model import build_baseline, ModelConfig

def main():
    ensure_dirs(ARTIFACTS, MODELS_DIR, METRICS_DIR, PLOTS_DIR)
    df = load_data()
    df["text"] = (df["title"].fillna("") + " " + df["description"].fillna("")).apply(clean)

    X = df["text"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipe = build_baseline(ModelConfig())
    pipe.fit(X_train, y_train)

    # Save model and meta
    model_path = MODELS_DIR / "model.joblib"
    vec_path = MODELS_DIR / "vectorizer.joblib"  # vectorizer embedded in pipeline; kept for clarity
    meta_path = MODELS_DIR / "meta.json"

    joblib.dump(pipe, model_path)
    # Extract vectorizer if needed
    try:
        joblib.dump(pipe.named_steps["tfidf"], vec_path)
    except Exception:
        pass

    save_json({
        "timestamp": timestamp(),
        "model": "LinearSVC + TFIDF",
        "sklearn_pipeline": True,
        "labels": "AG News: 0=World,1=Sports,2=Business,3=Sci/Tech",
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
    }, meta_path)

    # quick holdout metrics saved for convenience; full eval in evaluate.py
    from sklearn.metrics import accuracy_score
    preds = pipe.predict(X_test)
    acc = float(accuracy_score(y_test, preds))
    save_json({"holdout_accuracy": acc}, METRICS_DIR / "train_holdout.json")

    print(f"Saved model to {model_path}")

if __name__ == "__main__":
    main()

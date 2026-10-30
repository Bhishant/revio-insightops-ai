from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "real_yelp_reviews.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "sentiment_pipeline.joblib"
METRICS_PATH = MODEL_DIR / "evaluation_metrics.json"
BACKUP_PATH = MODEL_DIR / "sentiment_pipeline_before_real_yelp.joblib"

def normalize_label(value):
    if pd.isna(value):
        return None
    s = str(value).strip().lower()
    mapping = {
        "positive": "positive", "pos": "positive", "1": "positive",
        "neutral": "neutral", "neu": "neutral", "0": "neutral",
        "negative": "negative", "neg": "negative", "-1": "negative",
    }
    return mapping.get(s)

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Put real_yelp_reviews.csv inside the project's data folder."
        )

    df = pd.read_csv(DATA_PATH)
    required = {"text", "rating_review"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {missing}. Found: {df.columns.tolist()}")

    df = df[["text", "rating_review"]].copy()
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["label"] = df["rating_review"].map(normalize_label)
    df = df[(df["text"].str.len() > 0) & df["label"].notna()]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    if df["label"].nunique() != 3:
        raise ValueError(f"Expected positive, neutral, negative labels; found: {df['label'].value_counts().to_dict()}")

    print("Usable reviews:", len(df))
    print("Class distribution:")
    print(df["label"].value_counts().to_string())

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.20, random_state=42, stratify=df["label"]
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and not BACKUP_PATH.exists():
        joblib.copy if False else None
        import shutil
        shutil.copy2(MODEL_PATH, BACKUP_PATH)
        print(f"Backed up previous model to: {BACKUP_PATH}")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.98,
            max_features=100000,
            sublinear_tf=True,
            stop_words="english",
        )),
        ("classifier", LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            C=2.0,
            solver="lbfgs",
            random_state=42,
        )),
    ])

    print("\nTraining TF-IDF + Logistic Regression...")
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")
    report = classification_report(
        y_test, predictions, labels=["positive", "neutral", "negative"],
        output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_test, predictions, labels=["positive", "neutral", "negative"])

    print("\n=== HELD-OUT REAL YELP EVALUATION ===")
    print(f"Train rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(
        y_test, predictions, labels=["positive", "neutral", "negative"], zero_division=0
    ))
    print("Confusion matrix (rows=true, columns=predicted; positive, neutral, negative):")
    print(matrix)

    joblib.dump(pipeline, MODEL_PATH)
    metrics = {
        "model": "TF-IDF + Logistic Regression",
        "dataset": "Yelp restaurant reviews (rating_review labels)",
        "split": {"train_rows": int(len(X_train)), "test_rows": int(len(X_test)), "test_size": 0.2, "random_state": 42},
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "class_order": ["positive", "neutral", "negative"],
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "note": "Held-out split from Yelp restaurant review dataset; results may not generalize to other industries."
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    if BACKUP_PATH.exists():
        print(f"Previous model backup retained: {BACKUP_PATH}")

if __name__ == "__main__":
    main()

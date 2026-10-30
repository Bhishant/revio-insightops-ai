"""
Evaluate Revio InsightOps AI's existing sentiment model on the uploaded Yelp dataset.
This script evaluates only; it does not train or modify the model.
"""
from pathlib import Path
import sys
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "sentiment_pipeline.joblib"
DATA_PATH = ROOT / "data" / "real_yelp_reviews.csv"
LABELS = ["positive", "neutral", "negative"]

def find_exact_or_contains(columns, exact_names, contains_names=()):
    mapping = {str(c).strip().lower(): c for c in columns}
    for name in exact_names:
        if name in mapping:
            return mapping[name]
    for c in columns:
        lowered = str(c).strip().lower()
        if any(term in lowered for term in contains_names):
            return c
    return None

def normalize_label(value):
    if pd.isna(value):
        return None
    value = str(value).strip().lower()
    return {
        "positive": "positive", "pos": "positive",
        "neutral": "neutral", "neu": "neutral",
        "negative": "negative", "neg": "negative",
    }.get(value)

def label_from_rating(value):
    try:
        rating = float(value)
    except (TypeError, ValueError):
        return None
    if rating <= 2:
        return "negative"
    if rating == 3:
        return "neutral"
    if rating >= 4:
        return "positive"
    return None

def main():
    if not MODEL_PATH.exists():
        sys.exit(f"Model not found: {MODEL_PATH}\nRun `python train_model.py` from your project folder first.")
    if not DATA_PATH.exists():
        sys.exit(f"Dataset not found: {DATA_PATH}\nPlace real_yelp_reviews.csv inside the project's data folder.")

    df = pd.read_csv(DATA_PATH)
    print("Dataset columns:", df.columns.tolist())

    # For the supplied Yelp file, `text` is review text and `rating_review`
    # already contains Positive / Neutral / Negative labels.
    text_col = find_exact_or_contains(
        df.columns,
        ["text", "review_text", "review", "content", "comment", "feedback"],
        ("review_text", "review", "comment", "feedback", "text")
    )
    label_col = find_exact_or_contains(
        df.columns,
        ["rating_review", "sentiment", "sentiment_label", "label", "polarity", "class"],
        ("sentiment", "polarity", "label", "class")
    )
    rating_col = find_exact_or_contains(
        df.columns,
        ["stars_review", "review_stars", "star_rating", "review_rating", "stars", "rating"],
        ("stars_review", "review_stars", "star_rating", "review_rating", "rating", "stars")
    )

    if text_col is None:
        sys.exit("Could not identify the review text column. Expected a column such as `text`.")
    if label_col is None and rating_col is None:
        sys.exit("Could not find sentiment labels or star ratings.")

    work = pd.DataFrame({"text": df[text_col].fillna("").astype(str).str.strip()})
    if label_col is not None:
        work["true_sentiment"] = df[label_col].map(normalize_label)
        # If a label column exists but is not a recognized sentiment field, use star ratings.
        if work["true_sentiment"].notna().sum() == 0 and rating_col is not None:
            work["true_sentiment"] = df[rating_col].map(label_from_rating)
            label_description = f"derived from {rating_col}"
        else:
            label_description = label_col
    else:
        work["true_sentiment"] = df[rating_col].map(label_from_rating)
        label_description = f"derived from {rating_col}"

    work = work[work["text"].ne("") & work["true_sentiment"].isin(LABELS)]
    work = work.drop_duplicates(subset=["text"]).copy()

    if len(work) < 20:
        sys.exit(
            f"Only {len(work)} usable labelled reviews found.\n"
            f"Detected text column: {text_col}; label column: {label_col}; rating column: {rating_col}.\n"
            "Please share this output so the column mapping can be corrected."
        )

    if len(work) > 3000:
        work = work.sample(n=3000, random_state=42)

    model = joblib.load(MODEL_PATH)
    y_true = work["true_sentiment"]
    y_pred = model.predict(work["text"])

    print("\n=== EXTERNAL YELP REVIEW EVALUATION ===")
    print(f"Rows evaluated: {len(work)}")
    print(f"Text column: {text_col}")
    print(f"Ground-truth labels: {label_description}")
    print("\nClass distribution:")
    print(y_true.value_counts().reindex(LABELS, fill_value=0).to_string())
    print(f"\nAccuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro-F1: {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))
    print("Confusion matrix (rows=true, columns=predicted; order positive, neutral, negative):")
    print(confusion_matrix(y_true, y_pred, labels=LABELS))
    print("\nInterpretation note: Yelp restaurant reviews are an external cross-domain test, not a representative sample of all businesses.")
    if label_description.startswith("derived from"):
        print("Caution: labels were derived from star ratings, which are imperfect proxies for sentiment.")

if __name__ == "__main__":
    main()


import re

ISSUE_RULES = {
    "Delivery": [
        "delivery", "late delivery", "delayed delivery",
        "shipping", "courier", "package", "arrived late",
        "takeout", "take out", "pickup", "pick up",
        "waiting for order", "order never arrived"
    ],
    "Product quality": [
        "product quality", "broken product", "defective",
        "damaged item", "faulty", "poor quality",
        "food quality", "food was cold", "undercooked",
        "overcooked", "raw food", "food poisoning",
        "plastic in food", "portion size", "tasteless",
        "bland food", "bad food"
    ],
    "Customer support": [
        "customer support", "customer service", "rude staff",
        "unhelpful staff", "ignored my complaint",
        "no response", "manager", "staff was rude",
        "poor service", "terrible service", "waiter",
        "waitress", "server", "hostess", "host"
    ],
    "Billing & payment": [
        "charged", "refund", "billing", "payment",
        "overpriced", "overcharged", "charged twice",
        "wrong bill", "incorrect bill", "hidden fee",
        "extra charge", "tip", "gratuity", "expensive"
    ],
    "App & website": [
        "app", "website", "crashes", "crashed",
        "login", "loading", "checkout error",
        "page error", "online ordering", "booking website"
    ],
}


def normalize_text(text):
    """Normalize text before matching issue-related phrases."""
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def detect_issue(text):
    """
    Return the category with the strongest phrase match.
    Return Other when no relevant phrase is found.
    """
    lowered = normalize_text(text)
    scores = {}

    for category, phrases in ISSUE_RULES.items():
        scores[category] = sum(
            1 for phrase in phrases
            if re.search(
                r"(?<!\w)" + re.escape(phrase) + r"(?!\w)",
                lowered
            )
        )

    best_category = max(scores, key=scores.get)

    return best_category if scores[best_category] > 0 else "Other"


def priority_score(sentiment, text):
    """
    Transparent rule-based priority estimate, capped at 55.
    This is not a trained risk-prediction model.
    """
    sentiment = str(sentiment).lower().strip()
    lowered = normalize_text(text)

    score = {
        "negative": 15,
        "neutral": 5,
        "positive": 1,
    }.get(sentiment, 5)

    urgent = [
        "refund", "charged twice", "fraud", "unsafe",
        "food poisoning", "allergic reaction",
        "plastic in food", "overcharged", "broken"
    ]

    unresolved = [
        "ignored", "not resolved", "several calls",
        "no response", "still waiting", "no refund"
    ]

    if any(phrase in lowered for phrase in urgent):
        score += 25

    if any(phrase in lowered for phrase in unresolved):
        score += 15

    return min(score, 55)

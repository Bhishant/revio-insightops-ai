from src.core import detect_issue, priority_score

def test_detect_issue_delivery():
    assert detect_issue("My delivery arrived late") == "Delivery"

def test_detect_issue_billing():
    assert detect_issue("I was charged twice and need a refund") == "Billing & payment"

def test_negative_review_gets_higher_base_priority():
    assert priority_score("negative", "The service was poor") > priority_score("positive", "Great service")

def test_urgent_terms_raise_priority():
    assert priority_score("negative", "I was charged twice") > priority_score("negative", "The service was poor")

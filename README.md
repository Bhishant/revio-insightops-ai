# Revio InsightOps AI

**AI powered customer feedback intelligence — from reviews to actionable insights.**

A assessment MVP that classifies review sentiment, groups likely issue themes, prioritizes reviews with transparent heuristics, and surfaces supporting examples in a Streamlit dashboard.

## Features
- CSV upload (`text` required; `date`, `rating`, and `branch` optional)
- TF-IDF + Logistic Regression sentiment baseline
- Rule-based issue categories and transparent priority score
- Interactive Plotly charts, review filters, and CSV export
- Reproducible evaluation on a small held-out synthetic dataset

## Quick start (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

Open the local URL printed by Streamlit.

## Dataset and evaluation
The included training and evaluation data are synthetic examples created for a prototype. They are not Cynbit customer data. Run `python train_model.py` to train the baseline and write `models/evaluation_metrics.json`.

**Important limitation:** the evaluation set is very small and synthetic. Its metrics are only a smoke test of the pipeline and are not evidence of real world accuracy. Before deployment for business decisions, evaluate on a representative, human labelled dataset and review class-level performance.

## Current architecture
1. CSV ingestion and text validation
2. TF-IDF feature extraction
3. Logistic Regression sentiment classification
4. Transparent keyword-based issue categorization
5. Heuristic priority scoring and evidence examples
6. Streamlit dashboard and CSV export

## Responsible use notes
- No customer data is sent to an external service by this MVP.
- Priority is a heuristic, not a measured estimate of financial impact.
- Predictions may be wrong, especially for sarcasm, mixed sentiment, domain-specific language, or languages other than English.
- Human review is required before acting on recommendations.

## Future work
- Compare against a pretrained transformer on representative labelled data
- Add semantic clustering for recurring issues
- Add an optional LLM response-drafting feature with evidence-grounding and human approval
- Track issue trends over time after enough dated reviews are available

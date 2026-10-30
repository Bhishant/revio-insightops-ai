from pathlib import Path
import json
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from src.core import detect_issue, priority_score

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "sentiment_pipeline.joblib"
METRICS_PATH = ROOT / "models" / "evaluation_metrics.json"
SAMPLE_PATH = ROOT / "data" / "sample_reviews.csv"

st.set_page_config(page_title="Revio InsightOps AI", page_icon="🔎", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:1.4rem;padding-bottom:2rem;max-width:1500px}
.hero{padding:1.35rem 1.5rem;border-radius:16px;background:linear-gradient(120deg,#101d35,#234879);color:white;margin-bottom:1rem;border:1px solid #2e507d}
.hero h1{margin:0;font-size:2.15rem;letter-spacing:-.03em}
.hero p{margin:.45rem 0 0;color:#d7e6ff;font-size:1rem}
.eyebrow{text-transform:uppercase;letter-spacing:.12em;font-size:.72rem;color:#a9c7f2;font-weight:700}
[data-testid="stMetric"]{background:rgba(100,120,150,.07);border:1px solid rgba(128,150,180,.22);padding:14px 16px;border-radius:12px}
</style>
<div class="hero"><div class="eyebrow">Customer feedback intelligence</div>
<h1>Revio InsightOps AI</h1><p>Turn customer reviews into evidence-backed actions.</p></div>
""", unsafe_allow_html=True)

def make_recommendation(issue):
    return {
        "Delivery":"Audit delayed orders by carrier and delivery region; compare promised versus actual delivery dates.",
        "Product quality":"Check whether complaints cluster around a product, batch, or supplier; inspect repeat defects.",
        "Customer support":"Review response times and unresolved tickets; identify where hand-offs or follow-ups fail.",
        "Billing & payment":"Reconcile disputed charges and refunds; check for repeated payment or checkout failures.",
        "App & website":"Reproduce the reported workflow and inspect error logs, device/browser patterns, and recent releases.",
        "Other":"Review the feedback and label the issue before selecting a corrective action."
    }.get(issue, "Review the feedback and identify the appropriate next step.")

def make_response(issue):
    action = {
        "Delivery":"check the shipment timeline and provide a clear delivery update",
        "Product quality":"review the product details and offer an appropriate resolution",
        "Customer support":"review the previous interaction and assign an owner for follow-up",
        "Billing & payment":"verify the transaction and refund status before confirming the next step",
        "App & website":"reproduce the reported issue and share a verified workaround if available",
        "Other":"review the feedback and identify the appropriate next step"
    }.get(issue, "review the feedback and identify the appropriate next step")
    return ("Thank you for sharing this feedback. We're sorry this experience did not meet expectations. "
            f"Our team should {action}. We will review the details and follow up with a clear update. "
            "Please contact support with your order or reference number if further details are needed.")

@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
    except Exception:
        return None

model = load_model()
if model is None:
    st.error("Sentiment model missing. Run `python train_model.py`, then refresh.")
    st.stop()

with st.sidebar:
    st.markdown("### Review data")
    uploaded = st.file_uploader("Upload reviews (CSV)", type=["csv"])
    st.caption("Required: `text` · Optional: `date`, `rating`, `branch`")
    use_sample = st.checkbox("Use sample reviews", value=uploaded is None)
    st.divider()
    st.caption("Sentiment is an ML prediction. Issue categories and priority are rule-based estimates; verify important cases manually.")

if uploaded is not None:
    try:
        raw = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()
elif use_sample and SAMPLE_PATH.exists():
    raw = pd.read_csv(SAMPLE_PATH)
else:
    raw = pd.DataFrame({"text":[]})

if "text" not in raw.columns:
    st.error("Your CSV needs a `text` column containing customer reviews.")
    st.stop()
raw = raw.dropna(subset=["text"]).copy()
raw["text"] = raw["text"].astype(str).str.strip()
raw = raw[raw["text"].str.len() > 0].reset_index(drop=True)
if raw.empty:
    st.info("Upload a CSV or select sample reviews to begin.")
    st.stop()

raw["Sentiment"] = model.predict(raw["text"])
raw["Confidence"] = model.predict_proba(raw["text"]).max(axis=1) if hasattr(model, "predict_proba") else None
raw["Issue"] = raw["text"].apply(detect_issue)
raw["Priority"] = raw.apply(lambda r: priority_score(r["Sentiment"], r["text"]), axis=1)
raw["Review"] = raw["text"]
negative_rows = raw[raw["Sentiment"] == "negative"].copy()
total = len(raw)
negative = len(negative_rows)
negative_pct = negative / total * 100 if total else 0
high_priority = int((raw["Priority"] >= 40).sum())
low_conf = int((raw["Confidence"] < .55).sum()) if raw["Confidence"].notna().any() else 0

overview, insights, explorer, evaluation = st.tabs(["📊 Overview","💡 AI Insights","🔎 Review Explorer","🧪 Model Evaluation"])
with overview:
    a,b,c,d = st.columns(4)
    a.metric("Reviews analyzed", f"{total:,}")
    b.metric("Negative reviews", f"{negative:,}", f"{negative_pct:.1f}% of total")
    c.metric("High-priority flags", f"{high_priority:,}")
    d.metric("Low-confidence predictions", f"{low_conf:,}")
    st.subheader("Feedback overview")
    left,right = st.columns(2)
    with left:
        counts = raw["Sentiment"].value_counts().rename_axis("Sentiment").reset_index(name="Reviews")
        fig = px.pie(counts,names="Sentiment",values="Reviews",hole=.58,title="Sentiment distribution",
                     color="Sentiment",color_discrete_map={"positive":"#34d399","neutral":"#fbbf24","negative":"#fb7185"})
        fig.update_layout(legend_title_text="",margin=dict(t=55,b=10,l=10,r=10))
        st.plotly_chart(fig,use_container_width=True)
    with right:
        counts = raw["Issue"].value_counts().rename_axis("Issue").reset_index(name="Reviews").sort_values("Reviews")
        fig = px.bar(counts,x="Reviews",y="Issue",orientation="h",text="Reviews",title="Common feedback themes")
        fig.update_layout(xaxis_title="Review count",yaxis_title="",margin=dict(t=55,b=10,l=10,r=10))
        st.plotly_chart(fig,use_container_width=True)
    st.caption("Counts reflect the loaded dataset. Validate predictions before operational use.")

with insights:
    st.subheader("Evidence-backed action recommendations")
    st.write("Recommendations use detected issue categories and example reviews. They are decision support, not autonomous actions.")
    if negative_rows.empty:
        st.success("No negative reviews were predicted. Verify predictions against labelled examples before drawing conclusions.")
    else:
        summary = negative_rows.groupby("Issue").agg(Negative_reviews=("Review","count"),Average_priority=("Priority","mean")).reset_index()
        summary["Average_priority"] = summary["Average_priority"].round(1)
        summary = summary.sort_values(["Average_priority","Negative_reviews"],ascending=False)
        issue = st.selectbox("Investigate an issue",summary["Issue"].tolist())
        selected = negative_rows[negative_rows["Issue"] == issue].sort_values("Priority",ascending=False)
        c1,c2,c3 = st.columns(3)
        c1.metric("Negative reviews in theme",len(selected))
        c2.metric("Average priority score",f"{selected['Priority'].mean():.1f}/55")
        c3.metric("Highest individual score",f"{int(selected['Priority'].max())}/55")
        st.markdown("#### Supporting review evidence")
        for review in selected["Review"].head(3).tolist():
            st.markdown(f"> {review}")
        st.markdown("#### Recommended investigation")
        st.info(make_recommendation(issue))
        st.markdown("#### Draft customer response")
        st.text_area("Editable draft — review before sending",value=make_response(issue),height=130,key="response_draft")
        st.caption("This is a local response template, not an LLM-generated response. A human should personalize and approve it.")
        st.markdown("#### Issue priority summary")
        st.dataframe(summary.rename(columns={"Issue":"Issue category","Negative_reviews":"Negative reviews","Average_priority":"Average priority"}),
                     use_container_width=True,hide_index=True)

with explorer:
    st.subheader("Search and inspect individual reviews")
    c1,c2 = st.columns(2)
    with c1:
        sentiments = st.multiselect("Sentiment",["positive","neutral","negative"],default=["negative","neutral","positive"])
    with c2:
        issues = st.multiselect("Issue category",sorted(raw["Issue"].unique()),default=sorted(raw["Issue"].unique()))
    view = raw[raw["Sentiment"].isin(sentiments)&raw["Issue"].isin(issues)].sort_values("Priority",ascending=False).copy()
    cols = ["Review","Sentiment","Confidence","Issue","Priority"]+[x for x in ["rating","branch","date"] if x in view.columns]
    st.dataframe(view[cols].rename(columns={"Review":"Customer review","Sentiment":"Predicted sentiment","Confidence":"Model confidence",
        "Issue":"Issue category","Priority":"Priority score","rating":"Rating","branch":"Branch","date":"Date"}),
        use_container_width=True,hide_index=True)
    st.download_button("Download analyzed reviews (CSV)",data=raw.to_csv(index=False).encode("utf-8"),
                       file_name="revio_analyzed_reviews.csv",mime="text/csv")

with evaluation:
    st.subheader("Model evaluation")
    st.write("Metrics are measured on a held-out test split of the Yelp restaurant review dataset.")
    if METRICS_PATH.exists():
        try:
            metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            a,b,c = st.columns(3)
            a.metric("Evaluation examples",metrics.get("split", {}).get("test_rows", "—"))
            b.metric("Accuracy",f"{metrics.get('accuracy',0):.1%}")
            c.metric("Macro-F1",f"{metrics.get('macro_f1',0):.3f}")
            report = metrics.get("classification_report",{})
            rows = [{"Class":label.title(),"Precision":round(report[label].get("precision",0),3),
                     "Recall":round(report[label].get("recall",0),3),"F1-score":round(report[label].get("f1-score",0),3),
                     "Support":int(report[label].get("support",0))}
                    for label in ["positive","neutral","negative"] if label in report]
            if rows:
                st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
            st.warning(metrics.get("note","Small synthetic dataset; interpret metrics cautiously."))
        except Exception:
            st.info("Evaluation metrics could not be read. Run `python train_model.py` again.")
    else:
        st.info("Evaluation file not found. Run `python train_model.py` to generate it.")
    st.markdown("**Next step:** expand the human-labelled test set and review per-class errors before reporting performance claims.")

st.divider()
st.caption("Revio InsightOps AI · Assessment prototype · This version makes no external API calls.")

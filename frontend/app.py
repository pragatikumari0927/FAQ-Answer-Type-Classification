"""
FAQ Answer Type Classification -- Streamlit Dashboard
=======================================================
An interactive dashboard for a supervised NLP pipeline that classifies
FAQ answers by:
  1. Answer Type   (Definition, Symptoms, Treatment, Diagnosis, ...)
  2. Sentiment      (Positive / Neutral / Negative)
  3. Tone           (Reassuring / Cautionary / Neutral-Informative)

Trained on the MedQuAD dataset (public, GitHub: abachaa/MedQuAD).
"""
import json
import re
import string

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

try:
    from wordcloud import WordCloud, STOPWORDS
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

st.set_page_config(
    page_title="FAQ Answer Type Classifier",
    page_icon="🏷️",
    layout="wide",
)

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "faq_answer_types_labeled.csv"
MODELS_DIR = BASE_DIR / "backend" / "models"
ASSETS_DIR = BASE_DIR / "report" / "assets"

PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")
SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = PUNCT_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    vectorizer = joblib.load(f"{MODELS_DIR}/tfidf_vectorizer.pkl")
    type_clf = joblib.load(f"{MODELS_DIR}/answer_type_model.pkl")
    type_enc = joblib.load(f"{MODELS_DIR}/answer_type_encoder.pkl")
    sent_clf = joblib.load(f"{MODELS_DIR}/sentiment_model.pkl")
    sent_enc = joblib.load(f"{MODELS_DIR}/sentiment_encoder.pkl")
    tone_clf = joblib.load(f"{MODELS_DIR}/tone_model.pkl")
    tone_enc = joblib.load(f"{MODELS_DIR}/tone_encoder.pkl")
    with open(f"{MODELS_DIR}/metrics.json") as f:
        metrics = json.load(f)
    return vectorizer, type_clf, type_enc, sent_clf, sent_enc, tone_clf, tone_enc, metrics


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


vectorizer, type_clf, type_enc, sent_clf, sent_enc, tone_clf, tone_enc, metrics = load_artifacts()
df = load_data()

TYPE_COLORS = {
    "Definition/Overview": "#4C72B0", "Symptoms": "#DD8452", "Treatment": "#55A868",
    "Genetics": "#C44E52", "Frequency": "#8172B2", "Causes": "#937860",
    "Diagnosis": "#DA8BC3", "Research": "#8C8C8C", "Prognosis": "#CCB974",
    "Risk Factors": "#64B5CD", "Considerations": "#B07AA1", "Prevention": "#59A14F",
}


def predict_all(question: str, answer: str):
    text = clean_text(f"{question} {answer}")
    X = vectorizer.transform([text])

    type_probs = type_clf.predict_proba(X)[0]
    type_pred = type_enc.classes_[type_probs.argmax()]

    sent_probs = sent_clf.predict_proba(X)[0]
    sent_pred = sent_enc.classes_[sent_probs.argmax()]

    tone_probs = tone_clf.predict_proba(X)[0]
    tone_pred = tone_enc.classes_[tone_probs.argmax()]

    return {
        "answer_type": (type_pred, dict(zip(type_enc.classes_, type_probs))),
        "sentiment": (sent_pred, dict(zip(sent_enc.classes_, sent_probs))),
        "tone": (tone_pred, dict(zip(tone_enc.classes_, tone_probs))),
    }


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🏷️ FAQ Answer Type Classifier")
st.sidebar.caption("NLP dashboard — TF-IDF + Logistic Regression")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview & EDA", "🔮 Try It Live", "📁 Batch Classify", "📈 Model Performance", "ℹ️ About"],
)
st.sidebar.divider()
st.sidebar.metric("Dataset size", f"{len(df):,} FAQ pairs")
st.sidebar.metric("Answer Type accuracy", f"{metrics['answer_type']['accuracy']*100:.1f}%")
st.sidebar.metric("Sentiment accuracy", f"{metrics['sentiment']['accuracy']*100:.1f}%")

# ---------------------------------------------------------------------------
# Page: Overview & EDA
# ---------------------------------------------------------------------------
if page == "📊 Overview & EDA":
    st.title("📊 Dataset Overview & Exploratory Analysis")
    st.markdown(
        "Source: **MedQuAD** — 47K+ medical FAQ pairs from 12 NIH sites, "
        "expert-annotated with question/answer types. "
        f"After cleaning and consolidating into a presentation-friendly taxonomy: **{len(df):,} rows / "
        f"{df['answer_type'].nunique()} answer types**."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total FAQs", f"{len(df):,}")
    c2.metric("Answer Types", df["answer_type"].nunique())
    c3.metric("Avg. answer length (words)", int(df["answer"].str.split().apply(len).mean()))
    c4.metric("Unique topics (focus)", df["focus"].nunique())

    st.subheader("Answer Type distribution")
    type_counts = df["answer_type"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = [TYPE_COLORS.get(t, "#999999") for t in type_counts.index]
    ax.bar(type_counts.index, type_counts.values, color=colors)
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sentiment distribution")
        sent_counts = df["sentiment"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(4.5, 4))
        ax2.pie(sent_counts.values, labels=sent_counts.index, autopct="%1.0f%%",
                colors=["#59A14F", "#BAB0AC", "#E15759"])
        st.pyplot(fig2)
    with col2:
        st.subheader("Tone distribution")
        tone_counts = df["tone"].value_counts()
        fig3, ax3 = plt.subplots(figsize=(4.5, 4))
        ax3.pie(tone_counts.values, labels=tone_counts.index, autopct="%1.0f%%",
                colors=["#4C72B0", "#DD8452", "#8C8C8C"])
        st.pyplot(fig3)

    st.subheader("Word cloud by Answer Type")
    if WORDCLOUD_AVAILABLE:
        selected_type = st.selectbox("Choose an answer type", sorted(df["answer_type"].unique()))
        text_blob = " ".join(df.loc[df["answer_type"] == selected_type, "answer"].astype(str).tolist())
        wc = WordCloud(width=1000, height=350, background_color="white",
                        stopwords=STOPWORDS, colormap="viridis").generate(text_blob[:200000])
        fig4, ax4 = plt.subplots(figsize=(11, 4))
        ax4.imshow(wc, interpolation="bilinear")
        ax4.axis("off")
        st.pyplot(fig4)
    else:
        st.info(
            "Install the `wordcloud` package to enable this chart "
            "(`pip install wordcloud`), then restart the app. "
            "Showing a top-words bar chart instead."
        )
        selected_type = st.selectbox("Choose an answer type", sorted(df["answer_type"].unique()))
        text_blob = " ".join(df.loc[df["answer_type"] == selected_type, "answer"].astype(str).tolist())
        words = [w for w in re.findall(r"[a-z]+", text_blob.lower()) if len(w) > 3]
        top_words = pd.Series(words).value_counts().head(20)
        st.bar_chart(top_words)

    st.subheader("Sample rows")
    st.dataframe(
        df[["question", "answer_type", "sentiment", "tone"]].sample(10, random_state=1),
        use_container_width=True,
    )

# ---------------------------------------------------------------------------
# Page: Try It Live
# ---------------------------------------------------------------------------
elif page == "🔮 Try It Live":
    st.title("🔮 Classify a FAQ Answer")
    st.markdown("Enter a question and its answer to see the predicted **Answer Type**, **Sentiment**, and **Tone**.")

    examples = {
        "-- pick an example --": ("", ""),
        "Definition-style": ("What is asthma?", "Asthma is a chronic disease that affects the airways in the lungs, causing them to become inflamed and narrowed."),
        "Treatment-style": ("How is asthma treated?", "Treatment usually involves inhaled corticosteroids and bronchodilators to reduce inflammation and open the airways. In most cases, the condition is manageable with the right medication plan."),
        "Risk-style": ("What are the risks of untreated asthma?", "Untreated asthma can lead to severe, potentially life-threatening asthma attacks, permanent lung damage, and frequent emergency hospital visits."),
    }
    choice = st.selectbox("Quick examples", list(examples.keys()))
    ex_q, ex_a = examples[choice]

    col1, col2 = st.columns(2)
    with col1:
        question = st.text_input("FAQ Question", value=ex_q, placeholder="e.g. What are the symptoms of ...?")
    with col2:
        pass
    answer = st.text_area("FAQ Answer", value=ex_a, height=160,
                           placeholder="Paste the answer text here...")

    if st.button("Classify", type="primary", use_container_width=True):
        if not answer.strip():
            st.warning("Please enter an answer to classify.")
        else:
            result = predict_all(question, answer)
            c1, c2, c3 = st.columns(3)

            type_pred, type_probs = result["answer_type"]
            sent_pred, sent_probs = result["sentiment"]
            tone_pred, tone_probs = result["tone"]

            c1.metric("Predicted Answer Type", type_pred,
                      f"{type_probs[type_pred]*100:.1f}% confidence")
            c2.metric("Predicted Sentiment", sent_pred,
                      f"{sent_probs[sent_pred]*100:.1f}% confidence")
            c3.metric("Predicted Tone", tone_pred,
                      f"{tone_probs[tone_pred]*100:.1f}% confidence")

            st.subheader("Answer Type — full probability breakdown")
            probs_df = pd.DataFrame(sorted(type_probs.items(), key=lambda x: -x[1]),
                                     columns=["Answer Type", "Probability"])
            st.bar_chart(probs_df.set_index("Answer Type"))

# ---------------------------------------------------------------------------
# Page: Batch Classify
# ---------------------------------------------------------------------------
elif page == "📁 Batch Classify":
    st.title("📁 Batch Classification")
    st.markdown(
        "Upload a CSV with `question` and `answer` columns to classify many FAQs at once."
    )
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        missing = {"question", "answer"} - set(batch_df.columns)
        if missing:
            st.error(f"CSV is missing required column(s): {missing}")
        else:
            with st.spinner("Classifying..."):
                texts = (batch_df["question"].fillna("") + " " + batch_df["answer"].fillna("")).apply(clean_text)
                X = vectorizer.transform(texts)
                batch_df["predicted_answer_type"] = type_enc.inverse_transform(type_clf.predict(X))
                batch_df["predicted_sentiment"] = sent_enc.inverse_transform(sent_clf.predict(X))
                batch_df["predicted_tone"] = tone_enc.inverse_transform(tone_clf.predict(X))

            st.success(f"Classified {len(batch_df)} rows.")
            st.dataframe(batch_df, use_container_width=True)

            st.download_button(
                "Download results as CSV",
                batch_df.to_csv(index=False).encode("utf-8"),
                file_name="faq_classified.csv",
                mime="text/csv",
            )
    else:
        st.info("No file uploaded yet. You can try the sample dataset below instead.")
        if st.button("Use a random 20-row sample from the training data"):
            sample = df[["question", "answer"]].sample(20, random_state=7)
            texts = (sample["question"] + " " + sample["answer"]).apply(clean_text)
            X = vectorizer.transform(texts)
            sample["predicted_answer_type"] = type_enc.inverse_transform(type_clf.predict(X))
            sample["predicted_sentiment"] = sent_enc.inverse_transform(sent_clf.predict(X))
            st.dataframe(sample, use_container_width=True)

# ---------------------------------------------------------------------------
# Page: Model Performance
# ---------------------------------------------------------------------------
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")

    tab1, tab2, tab3 = st.tabs(["Answer Type", "Sentiment", "Tone"])

    with tab1:
        m = metrics["answer_type"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{m['accuracy']*100:.1f}%")
        c2.metric("Macro F1", f"{m['macro_f1']*100:.1f}%")
        c3.metric("Test samples", m["n_test"])
        st.image(f"{ASSETS_DIR}/cm_answer_type.png", use_container_width=True)
        rep = pd.DataFrame(m["class_report"]).T.drop(index=["accuracy"], errors="ignore")
        st.dataframe(rep.round(3), use_container_width=True)

    with tab2:
        m = metrics["sentiment"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{m['accuracy']*100:.1f}%")
        c2.metric("Macro F1", f"{m['macro_f1']*100:.1f}%")
        c3.metric("Test samples", m["n_test"])
        st.image(f"{ASSETS_DIR}/cm_sentiment.png", use_container_width=True)
        rep = pd.DataFrame(m["class_report"]).T.drop(index=["accuracy"], errors="ignore")
        st.dataframe(rep.round(3), use_container_width=True)

    with tab3:
        m = metrics["tone"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{m['accuracy']*100:.1f}%")
        c2.metric("Macro F1", f"{m['macro_f1']*100:.1f}%")
        c3.metric("Test samples", m["n_test"])
        rep = pd.DataFrame(m["class_report"]).T.drop(index=["accuracy"], errors="ignore")
        st.dataframe(rep.round(3), use_container_width=True)

# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------
else:
    st.title("ℹ️ About this project")
    st.markdown("""
### FAQ Answer Type Classification using NLP

**Pipeline**
1. **Data**: [MedQuAD](https://github.com/abachaa/MedQuAD) — public, expert-annotated medical FAQ dataset (47K+ QA pairs, 12 NIH sources).
2. **Cleaning**: consolidated 15 raw question types into 11 clean Answer Type categories, filtered short/noisy rows, deduplicated.
3. **Features**: TF-IDF (unigrams + bigrams, 15,000 features, English stopwords removed).
4. **Models**: Logistic Regression (`class_weight="balanced"`) for:
   - **Answer Type** — Definition, Symptoms, Treatment, Diagnosis, Causes, Genetics, Frequency, Prognosis, Risk Factors, Prevention, Considerations, Research
   - **Sentiment** — Positive / Neutral / Negative (weak-labeled with NLTK VADER, then learned by a supervised classifier so it generalizes to new text)
   - **Tone** — Reassuring / Cautionary / Neutral-Informative (weak-labeled with a keyword lexicon, then learned the same way)
5. **Evaluation**: held-out 20% test split, accuracy + macro-F1 + per-class precision/recall/F1 + confusion matrices.
6. **Dashboard**: this Streamlit app — EDA, live single-example prediction, CSV batch classification, and model performance reporting.

**Why Logistic Regression + TF-IDF?** It's fast to train, highly interpretable (one weight per word/n-gram per class), and a strong, standard baseline for text classification — a good foundation to later swap in embeddings (e.g. Sentence-BERT) or a fine-tuned transformer if higher accuracy is needed.
    """)

"""
train.py
--------
Full NLP training pipeline for the FAQ Answer Type Classification project.

Steps:
1. Load the cleaned FAQ dataset (question, answer, answer_type).
2. Text preprocessing (lowercasing, cleaning, stopword-aware TF-IDF).
3. Feature engineering: TF-IDF over question+answer text (word n-grams).
4. Train a Logistic Regression classifier -> Answer Type (multi-class).
5. Derive Sentiment labels with VADER (lexicon-based) on the answer text,
   then train a second Logistic Regression classifier -> Sentiment, so the
   dashboard can predict sentiment for brand-new text too (not just look
   VADER up), which is the standard supervised-on-top-of-lexicon pattern.
6. Derive a simple rule-based Tone label (Reassuring / Cautionary /
   Neutral-Informative) from keyword lexicons.
7. Evaluate everything, save metrics + confusion matrices, persist all
   artifacts (vectorizer, encoders, models) with joblib for the Streamlit app.
"""
import json
import re
import string

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "faq_answer_types.csv"
MODELS_DIR = BASE_DIR / "backend" / "models"
ASSETS_DIR = BASE_DIR / "report" / "assets"

nltk.download("vader_lexicon", quiet=True)

# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")
SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = PUNCT_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Tone lexicon (simple, transparent, rule-based -> used as weak supervision)
# ---------------------------------------------------------------------------
REASSURING_WORDS = {
    "safe", "effective", "manageable", "treatable", "curable", "improve",
    "recovery", "successful", "helps", "relief", "benefit", "healthy",
    "prevent", "protect", "normal", "mild",
}
CAUTIONARY_WORDS = {
    "risk", "danger", "severe", "fatal", "complication", "serious",
    "warning", "emergency", "death", "chronic", "life-threatening",
    "worsen", "damage", "irreversible", "caution",
}


def rule_based_tone(answer: str) -> str:
    tokens = set(clean_text(answer).split())
    reassure_hits = len(tokens & REASSURING_WORDS)
    caution_hits = len(tokens & CAUTIONARY_WORDS)
    if caution_hits > reassure_hits and caution_hits > 0:
        return "Cautionary"
    if reassure_hits > caution_hits and reassure_hits > 0:
        return "Reassuring"
    return "Neutral/Informative"


def vader_sentiment_label(score: float) -> str:
    if score >= 0.05:
        return "Positive"
    if score <= -0.05:
        return "Negative"
    return "Neutral"


def save_confusion_matrix(cm, labels, path, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=7)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df["text"] = (df["question"].fillna("") + " " + df["answer"].fillna("")).apply(clean_text)

    # -----------------------------------------------------------------
    # Weak-supervised labels for sentiment + tone
    # -----------------------------------------------------------------
    print("Deriving sentiment (VADER) and tone (lexicon) labels...")
    sia = SentimentIntensityAnalyzer()
    df["sentiment_score"] = df["answer"].apply(lambda a: sia.polarity_scores(str(a))["compound"])
    df["sentiment"] = df["sentiment_score"].apply(vader_sentiment_label)
    df["tone"] = df["answer"].apply(rule_based_tone)

    # -----------------------------------------------------------------
    # TF-IDF features (shared vectorizer for answer-type + sentiment)
    # -----------------------------------------------------------------
    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
        min_df=2,
    )
    X = vectorizer.fit_transform(df["text"])

    results = {}

    # -----------------------------------------------------------------
    # 1) Answer Type classifier
    # -----------------------------------------------------------------
    print("Training Answer Type classifier...")
    y_type_enc = LabelEncoder()
    y_type = y_type_enc.fit_transform(df["answer_type"])

    Xtr, Xte, ytr, yte, idx_tr, idx_te = train_test_split(
        X, y_type, df.index, test_size=0.2, random_state=42, stratify=y_type
    )
    type_clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=5)
    type_clf.fit(Xtr, ytr)
    pred = type_clf.predict(Xte)

    acc = accuracy_score(yte, pred)
    f1 = f1_score(yte, pred, average="macro")
    report = classification_report(yte, pred, target_names=y_type_enc.classes_, output_dict=True)
    cm = confusion_matrix(yte, pred)
    save_confusion_matrix(cm, y_type_enc.classes_, f"{ASSETS_DIR}/cm_answer_type.png",
                           "Answer Type - Confusion Matrix")

    results["answer_type"] = {
        "accuracy": acc,
        "macro_f1": f1,
        "classes": list(y_type_enc.classes_),
        "class_report": report,
        "n_train": Xtr.shape[0],
        "n_test": Xte.shape[0],
    }
    print(f"  Answer Type -> accuracy={acc:.3f}, macro-F1={f1:.3f}")

    # -----------------------------------------------------------------
    # 2) Sentiment classifier (learns to reproduce/generalize VADER labels)
    # -----------------------------------------------------------------
    print("Training Sentiment classifier...")
    y_sent_enc = LabelEncoder()
    y_sent = y_sent_enc.fit_transform(df["sentiment"])

    Xtr2, Xte2, ytr2, yte2 = train_test_split(
        X, y_sent, test_size=0.2, random_state=42, stratify=y_sent
    )
    sent_clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=5)
    sent_clf.fit(Xtr2, ytr2)
    pred2 = sent_clf.predict(Xte2)

    acc2 = accuracy_score(yte2, pred2)
    f1_2 = f1_score(yte2, pred2, average="macro")
    report2 = classification_report(yte2, pred2, target_names=y_sent_enc.classes_, output_dict=True)
    cm2 = confusion_matrix(yte2, pred2)
    save_confusion_matrix(cm2, y_sent_enc.classes_, f"{ASSETS_DIR}/cm_sentiment.png",
                           "Sentiment - Confusion Matrix")

    results["sentiment"] = {
        "accuracy": acc2,
        "macro_f1": f1_2,
        "classes": list(y_sent_enc.classes_),
        "class_report": report2,
        "n_train": Xtr2.shape[0],
        "n_test": Xte2.shape[0],
    }
    print(f"  Sentiment -> accuracy={acc2:.3f}, macro-F1={f1_2:.3f}")

    # -----------------------------------------------------------------
    # 3) Tone classifier (learns the rule-based tone labels)
    # -----------------------------------------------------------------
    print("Training Tone classifier...")
    y_tone_enc = LabelEncoder()
    y_tone = y_tone_enc.fit_transform(df["tone"])

    Xtr3, Xte3, ytr3, yte3 = train_test_split(
        X, y_tone, test_size=0.2, random_state=42, stratify=y_tone
    )
    tone_clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=5)
    tone_clf.fit(Xtr3, ytr3)
    pred3 = tone_clf.predict(Xte3)

    acc3 = accuracy_score(yte3, pred3)
    f1_3 = f1_score(yte3, pred3, average="macro")
    report3 = classification_report(yte3, pred3, target_names=y_tone_enc.classes_, output_dict=True)

    results["tone"] = {
        "accuracy": acc3,
        "macro_f1": f1_3,
        "classes": list(y_tone_enc.classes_),
        "class_report": report3,
        "n_train": Xtr3.shape[0],
        "n_test": Xte3.shape[0],
    }
    print(f"  Tone -> accuracy={acc3:.3f}, macro-F1={f1_3:.3f}")

    # -----------------------------------------------------------------
    # Save everything
    # -----------------------------------------------------------------
    print("Saving artifacts...")
    joblib.dump(vectorizer, f"{MODELS_DIR}/tfidf_vectorizer.pkl")
    joblib.dump(type_clf, f"{MODELS_DIR}/answer_type_model.pkl")
    joblib.dump(y_type_enc, f"{MODELS_DIR}/answer_type_encoder.pkl")
    joblib.dump(sent_clf, f"{MODELS_DIR}/sentiment_model.pkl")
    joblib.dump(y_sent_enc, f"{MODELS_DIR}/sentiment_encoder.pkl")
    joblib.dump(tone_clf, f"{MODELS_DIR}/tone_model.pkl")
    joblib.dump(y_tone_enc, f"{MODELS_DIR}/tone_encoder.pkl")

    with open(f"{MODELS_DIR}/metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save the labeled dataset (with sentiment/tone) for the EDA dashboard
    labeled_path = DATA_PATH.with_name(DATA_PATH.stem + "_labeled.csv")
    df.to_csv(labeled_path, index=False)

    print("Done.")


if __name__ == "__main__":
    main()

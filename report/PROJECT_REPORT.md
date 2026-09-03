# Project Report: FAQ Answer Type Classification using NLP

## 1. Problem Statement

Given an FAQ question and its answer, automatically classify:
1. **Answer Type** — what kind of information the answer provides
   (Definition, Symptoms, Treatment, Diagnosis, Causes, Genetics, Frequency,
   Prognosis, Risk Factors, Prevention, Considerations, Research)
2. **Sentiment** — Positive / Neutral / Negative tone of the answer text
3. **Tone** — Reassuring / Cautionary / Neutral-Informative framing

This kind of classification is useful for organizing large FAQ knowledge
bases, powering smarter chatbot retrieval, and auto-tagging new FAQ content
as it's written.

## 2. Dataset

**MedQuAD** (Medical Question Answering Dataset), public on GitHub
(abachaa/MedQuAD). 47,457 medical QA pairs sourced from 12 NIH websites,
each expert-annotated with a question/answer type (`qtype`).

After parsing, cleaning, deduplication, and consolidating 15 raw `qtype`
values into 11 presentation-friendly Answer Type categories:

| Stage | Rows |
|---|---|
| Raw parsed QA pairs | 16,406 |
| After cleaning + consolidation | 15,804 |

## 3. Methodology

1. **Text preprocessing**: lowercase, strip punctuation, collapse whitespace.
2. **Feature extraction**: TF-IDF over `question + answer` text — unigrams
   and bigrams, 15,000 max features, English stopwords removed,
   sublinear TF scaling.
3. **Weak supervision** for Sentiment and Tone (no manual labels exist for
   these in MedQuAD):
   - Sentiment labels come from NLTK's VADER lexicon (compound score
     thresholds: ≥0.05 Positive, ≤-0.05 Negative, else Neutral).
   - Tone labels come from a small hand-built keyword lexicon
     (reassuring words vs. cautionary words).
   - A supervised Logistic Regression classifier is then trained on the
     TF-IDF features to reproduce these labels — this lets the model
     generalize to brand-new text at inference time, rather than only
     being able to look up VADER/keyword scores.
4. **Answer Type classifier**: Logistic Regression, `class_weight="balanced"`
   to handle class imbalance (largest class ~4,500 rows, smallest ~180).
5. **Train/test split**: 80/20, stratified by label.

## 4. Results

| Task | Accuracy | Macro F1 | Test set size |
|---|---|---|---|
| Answer Type (11 classes) | **95.1%** | 91.6% | 3,161 |
| Sentiment (3 classes) | 83.1% | 79.6% | 3,161 |
| Tone (3 classes) | 83.3% | 81.5% | 3,161 |

Confusion matrices for Answer Type and Sentiment are in `report/assets/`.
Full per-class precision/recall/F1 is in `report/metrics.json` and viewable
in the dashboard's "Model Performance" page.

**Why Answer Type performs best**: the 11 categories map to fairly distinct
vocabulary (e.g. "treatment" answers use different words than "symptoms"
answers), which TF-IDF captures well. Sentiment and Tone are harder because
most medical FAQ answers are written in a neutral, clinical register, so the
signal separating "Reassuring" from "Neutral" is subtler.

## 5. Dashboard

A 5-page Streamlit app (`frontend/app.py`):
- **Overview & EDA** — dataset stats, class/sentiment/tone distributions, word clouds
- **Try It Live** — single-example prediction with confidence scores
- **Batch Classify** — CSV upload → bulk predictions → CSV download
- **Model Performance** — accuracy, macro-F1, confusion matrices, classification reports
- **About** — pipeline documentation

## 6. Limitations & Future Work

- Domain is currently medical FAQs (from MedQuAD); the pipeline generalizes
  to other domains given a similarly labeled dataset.
- TF-IDF + Logistic Regression is fast and interpretable but caps out below
  what a fine-tuned transformer (e.g. DistilBERT) could achieve, especially
  for the harder Sentiment/Tone tasks.
- Sentiment/Tone labels are weakly supervised (lexicon-derived), not
  human-annotated — a small manually labeled validation set would give a
  more trustworthy accuracy estimate for those two tasks.

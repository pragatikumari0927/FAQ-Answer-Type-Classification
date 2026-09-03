# FAQ Answer Type Classification using NLP

A complete NLP project that classifies FAQ answers by **Answer Type**
(Definition, Symptoms, Treatment, Diagnosis, ...), **Sentiment**, and **Tone**,
with an interactive Streamlit dashboard.

## Dataset

[MedQuAD](https://github.com/abachaa/MedQuAD) — 47K+ public, expert-annotated
medical FAQ pairs collected from 12 NIH websites (cancer.gov, NIDDK, GARD,
MedlinePlus, etc). Each question already carries an expert-assigned
`qtype` (e.g. *symptoms*, *treatment*, *exams and tests*), which this project
consolidates into a clean 11-category **Answer Type** taxonomy.

## Project structure

```
faq_project/
├── backend/
│   ├── data_ingestion.py     # raw MedQuAD XML -> data/raw/medquad_raw.csv
│   ├── prepare_data.py       # raw CSV -> cleaned data/faq_answer_types.csv
│   ├── train.py              # full training pipeline
│   └── models/                # trained artifacts (TF-IDF + classifiers)
├── data/
│   ├── faq_answer_types.csv           # cleaned dataset
│   └── faq_answer_types_labeled.csv   # + derived sentiment & tone labels
├── frontend/
│   └── app.py                 # Streamlit dashboard
├── report/
│   ├── assets/                 # confusion matrix images
│   ├── metrics.json            # accuracy / F1 / classification reports
│   └── PROJECT_REPORT.md       # write-up: approach, results, learnings
├── Problem Statement and Topic.docx
├── requirements.txt
├── .gitignore
└── README.md
```

## How it works

1. **Text cleaning**: lowercase, strip punctuation, collapse whitespace.
2. **Features**: TF-IDF (uni+bigrams, 15K features, English stopwords removed)
   over `question + answer`.
3. **Answer Type model**: Logistic Regression (`class_weight="balanced"`),
   11 classes after merging small categories.
4. **Sentiment model**: labels derived with NLTK VADER (lexicon-based), then
   a supervised Logistic Regression classifier is trained on top so it
   generalizes to brand-new text at inference time.
5. **Tone model**: same pattern, using a small reassuring/cautionary keyword
   lexicon for weak supervision.
6. **Evaluation**: 80/20 stratified split, accuracy, macro-F1, full
   classification report, confusion matrices.

## Setup

```bash
pip install -r requirements.txt
python -m nltk.downloader vader_lexicon   # only needed once
```

## Run the dashboard

```bash
cd frontend
streamlit run app.py
```

Models are already trained and included in `backend/models/`, so the app
runs immediately — no retraining needed.

The dashboard has five pages:
- **Overview & EDA** — class distributions, sentiment/tone breakdown, word clouds
- **Try It Live** — paste a question/answer, get instant predictions + confidence
- **Batch Classify** — upload a CSV, download predictions
- **Model Performance** — accuracy, macro-F1, confusion matrices, per-class report
- **About** — pipeline write-up

## Re-run the pipeline from scratch (optional)

```bash
# 1. Get the raw MedQuAD dataset
git clone https://github.com/abachaa/MedQuAD.git
python backend/data_ingestion.py --medquad-dir MedQuAD

# 2. Clean + consolidate labels
python backend/prepare_data.py

# 3. Retrain all three classifiers
python backend/train.py
```

## Extending this project

- Swap TF-IDF + Logistic Regression for embeddings (Sentence-BERT) or a
  fine-tuned transformer (DistilBERT) for higher accuracy.
- Add more FAQ domains beyond medical (banking, e-commerce, HR) by repeating
  `backend/prepare_data.py` on a differently-sourced CSV with
  `question, answer, answer_type` columns.
- Add explainability (e.g. LIME/SHAP on the TF-IDF + LogReg model) to show
  which words drove a prediction.

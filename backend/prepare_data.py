"""
prepare_data.py
----------------
Loads the raw parsed MedQuAD FAQ data, consolidates the fine-grained
`qtype` labels into a clean, human-readable Answer Type taxonomy, does
basic text cleaning, and writes the final training CSV.

Source dataset: MedQuAD (Medical Question Answering Dataset)
https://github.com/abachaa/MedQuAD
Ben Abacha & Demner-Fushman, "A Question-Entailment Approach to Question
Answering", BMC Bioinformatics, 2019.

Expects data/raw/medquad_raw.csv to already exist (produced by
`data_ingestion.py`, which parses the MedQuAD XML files into a flat CSV).
"""
import re
from pathlib import Path

import pandas as pd

# Paths are anchored to the project root (one level up from backend/) so this
# script works no matter which directory it's run from.
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "medquad_raw.csv"
OUT_PATH = BASE_DIR / "data" / "faq_answer_types.csv"

# Map the 15 raw qtypes -> 11 clean, presentation-friendly Answer Type labels
LABEL_MAP = {
    "information": "Definition/Overview",
    "symptoms": "Symptoms",
    "complications": "Symptoms",
    "treatment": "Treatment",
    "inheritance": "Genetics",
    "genetic changes": "Genetics",
    "frequency": "Frequency",
    "causes": "Causes",
    "exams and tests": "Diagnosis",
    "stages": "Diagnosis",
    "research": "Research",
    "outlook": "Prognosis",
    "susceptibility": "Risk Factors",
    "considerations": "Considerations",
    "prevention": "Prevention",
}

MIN_CLASS_SIZE = 40


def clean_text(t: str) -> str:
    t = str(t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def main():
    df = pd.read_csv(RAW_PATH)
    df = df.dropna(subset=["question", "answer", "answer_type"])

    df["answer_type"] = df["answer_type"].str.strip().str.lower()
    df["label"] = df["answer_type"].map(LABEL_MAP)
    df = df.dropna(subset=["label"])

    df["question"] = df["question"].apply(clean_text)
    df["answer"] = df["answer"].apply(clean_text)

    # basic length filters to drop noisy rows
    df = df[df["answer"].str.len().between(20, 4000)]
    df = df[df["question"].str.len().between(5, 300)]

    # drop very small classes
    counts = df["label"].value_counts()
    keep_labels = counts[counts >= MIN_CLASS_SIZE].index
    df = df[df["label"].isin(keep_labels)]

    # dedupe identical Q/A pairs
    df = df.drop_duplicates(subset=["question", "answer"])

    df = df[["question", "answer", "label", "focus", "source"]]
    df = df.rename(columns={"label": "answer_type"})

    df.to_csv(OUT_PATH, index=False)
    print(f"Final dataset: {df.shape[0]} rows, {df['answer_type'].nunique()} classes")
    print(df["answer_type"].value_counts())


if __name__ == "__main__":
    main()

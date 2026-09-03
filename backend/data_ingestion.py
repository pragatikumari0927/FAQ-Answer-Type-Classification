"""
data_ingestion.py
------------------
Parses the raw MedQuAD XML dump (https://github.com/abachaa/MedQuAD) into a
single flat CSV of (question, answer, answer_type, focus, source) rows.

Usage:
    1. Download/clone MedQuAD next to this project (or anywhere), e.g.:
         git clone https://github.com/abachaa/MedQuAD.git
    2. Run:
         python data_ingestion.py --medquad-dir /path/to/MedQuAD
    3. This writes data/raw/medquad_raw.csv, which prepare_data.py then
       cleans and consolidates into data/faq_answer_types.csv.

This step only needs to be re-run if you want to regenerate the dataset
from scratch. The repo already ships with the processed CSVs and trained
models, so it is NOT required to run the app.
"""
import argparse
import csv
import glob
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_PATH = BASE_DIR / "data" / "raw" / "medquad_raw.csv"


def parse(medquad_dir: str):
    rows = []
    xml_files = glob.glob(os.path.join(medquad_dir, "*", "*.xml"))
    print(f"Found {len(xml_files)} xml files")

    for fp in xml_files:
        try:
            tree = ET.parse(fp)
        except ET.ParseError:
            continue
        root = tree.getroot()
        focus_el = root.find("Focus")
        focus = focus_el.text.strip() if focus_el is not None and focus_el.text else ""
        source = root.attrib.get("source", "")
        qapairs = root.find("QAPairs")
        if qapairs is None:
            continue
        for qa in qapairs.findall("QAPair"):
            q_el = qa.find("Question")
            a_el = qa.find("Answer")
            if q_el is None or a_el is None:
                continue
            question = (q_el.text or "").strip()
            answer = (a_el.text or "").strip()
            qtype = q_el.attrib.get("qtype", "").strip().lower()
            if not question or not answer or not qtype:
                continue
            answer = re.sub(r"\s+", " ", answer)
            question = re.sub(r"\s+", " ", question)
            if len(answer) < 15:
                continue
            rows.append({
                "question": question, "answer": answer,
                "answer_type": qtype, "focus": focus, "source": source,
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--medquad-dir", required=True,
                         help="Path to a cloned MedQuAD repo (contains folders like 1_CancerGov_QA/)")
    args = parser.parse_args()

    rows = parse(args.medquad_dir)
    print(f"Parsed {len(rows)} QA pairs")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer", "answer_type", "focus", "source"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()

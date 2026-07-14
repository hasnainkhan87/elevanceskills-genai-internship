"""
preprocess_medquad.py
======================
Converts raw MedQuAD XML files into dataset/medquad_dataset.csv, using the
same prompt/response column structure as Task 1's dataset.csv so the
existing CSVLoader → FAISS → RetrievalQA pipeline works unchanged.

MedQuAD XML structure (confirmed against the real repo):
    <Document id="..." source="...">
      <Focus>Disease or Topic Name</Focus>
      <QAPairs>
        <QAPair pid="1">
          <Question qid="..." qtype="symptoms">Full question text</Question>
          <Answer>Full answer text</Answer>
        </QAPair>
        ...
      </QAPairs>
    </Document>

FOLDERS USED (and why)
-----------------------
MedQuAD has 12 top-level folders. Folders 10, 11, 12 (MedlinePlus A.D.A.M.
Encyclopedia, MedlinePlus Drugs, MedlinePlus Herbal Supplements) have their
<Answer> tags emptied out by the dataset's own authors to respect
MedlinePlus copyright — embedding those would just produce near-empty
vectors, so they're excluded entirely.

The 6 folders below all have real, populated answers:

    1_CancerGov_QA            all 116 files
    3_GHR_QA                  sample 200 of 1086 files
    5_NIDDK_QA                all 157 files
    6_NINDS_QA                sample 200 of 277 files
    7_SeniorHealth_QA         all 48 files
    8_NHLBI_QA_XML            all 88 files

To keep total volume within a safe range for free-tier Gemini embedding
quota, each XML file contributes at most 3 QA pairs (the first 3 found),
which keeps single long documents (some MedQuAD answers run past 4000
words) from dominating the sample. Folder-level sampling uses random.seed(42)
for reproducibility.

RUN
---
From inside task3/, with MedQuAD cloned at task3/medquad/:

    python src/preprocess_medquad.py

Output: dataset/medquad_dataset.csv
"""

import csv
import random
import re
from pathlib import Path
import xml.etree.ElementTree as ET

BASE_DIR = Path(__file__).resolve().parent.parent  # task3/
MEDQUAD_ROOT = BASE_DIR / "medquad"
OUTPUT_CSV = BASE_DIR / "dataset" / "medquad_dataset.csv"

RANDOM_SEED = 42
MAX_QA_PER_FILE = 3

# Hard cap on TOTAL rows across all folders combined, applied AFTER
# per-folder collection below. Confirmed via real testing: the full
# folder-level plan produces ~2,366 rows, and even a 300-row cap with
# batched/throttled embedding still took a long time to build in practice
# on a free-tier account (rate limits are tighter and less predictable
# than the published numbers suggest — actual throughput varies by
# account/region). 100 rows keeps a real one-time build noticeably
# faster while still giving a genuine multi-folder sample across all 6
# topic areas to demonstrate retrieval + entity recognition working end
# to end. Raise this if your account has enough quota to comfortably
# absorb a longer build — check
# https://ai.google.dev/gemini-api/docs/rate-limits for your tier first.
MAX_TOTAL_ROWS = 50


# folder_name -> None (use all files) or int (sample this many files)
FOLDER_PLAN = {
    "1_CancerGov_QA": None,
    "3_GHR_QA": 200,
    "5_NIDDK_QA": None,
    "6_NINDS_QA": 200,
    "7_SeniorHealth_QA": None,
    "8_NHLBI_QA_XML": None,
}


def clean_whitespace(text: str) -> str:
    """Collapse repeated whitespace/newlines left over from XML formatting."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def parse_xml_file(xml_path: Path, folder_name: str):
    """Parse one MedQuAD XML file, return up to MAX_QA_PER_FILE row dicts."""
    rows = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        return rows  # skip malformed files silently

    focus_el = root.find("Focus")
    focus = clean_whitespace(focus_el.text) if focus_el is not None else ""
    source = root.attrib.get("source", folder_name)

    count = 0
    for qapair in root.findall(".//QAPair"):
        if count >= MAX_QA_PER_FILE:
            break

        q_el = qapair.find("Question")
        a_el = qapair.find("Answer")
        if q_el is None or a_el is None:
            continue

        question = clean_whitespace(q_el.text)
        answer = clean_whitespace(a_el.text)
        qtype = q_el.attrib.get("qtype", "general").strip()

        if not question or not answer or len(answer) < 20:
            continue

        rows.append({
            "prompt": question,
            "response": answer,
            "focus": focus,
            "source": source,
            "qtype": qtype,
        })
        count += 1

    return rows


def collect_rows():
    if not MEDQUAD_ROOT.exists():
        raise FileNotFoundError(
            f"MedQuAD folder not found at {MEDQUAD_ROOT}.\n"
            "Clone it first (from inside task3/):\n"
            "  git clone https://github.com/abachaa/MedQuAD.git medquad"
        )

    random.seed(RANDOM_SEED)
    all_rows = []

    for folder_name, sample_size in FOLDER_PLAN.items():
        folder_path = MEDQUAD_ROOT / folder_name
        if not folder_path.exists():
            print(f"  WARNING: {folder_name} not found under {MEDQUAD_ROOT}, skipping.")
            continue

        xml_files = sorted(folder_path.glob("*.xml"))
        if sample_size is not None and len(xml_files) > sample_size:
            xml_files = random.sample(xml_files, sample_size)

        folder_rows = []
        for xml_file in xml_files:
            folder_rows.extend(parse_xml_file(xml_file, folder_name))

        print(f"  {folder_name}: {len(xml_files)} files -> {len(folder_rows)} QA pairs")
        all_rows.extend(folder_rows)

    if len(all_rows) > MAX_TOTAL_ROWS:
        random.seed(RANDOM_SEED)  # re-seed so this sampling step is also reproducible
        print(
            f"\n  Total collected: {len(all_rows)} rows, exceeds MAX_TOTAL_ROWS="
            f"{MAX_TOTAL_ROWS}. Sampling down (random, seed={RANDOM_SEED}) rather "
            f"than truncating, so every folder still contributes rows instead of "
            f"only the first ones processed."
        )
        all_rows = random.sample(all_rows, MAX_TOTAL_ROWS)

    return all_rows


def save_csv(rows):
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["prompt", "response", "focus", "source", "qtype"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} QA pairs to {OUTPUT_CSV}")


def main():
    print("Parsing MedQuAD XML files...")
    rows = collect_rows()

    if not rows:
        raise ValueError("No QA pairs were parsed — check MEDQUAD_ROOT and FOLDER_PLAN.")

    save_csv(rows)

    print("\nPreview of first 3 rows:")
    for row in rows[:3]:
        print(f"  Focus:    {row['focus']}")
        print(f"  Type:     {row['qtype']}")
        print(f"  Question: {row['prompt'][:80]}")
        print(f"  Answer:   {row['response'][:80]}")
        print()


if __name__ == "__main__":
    main()

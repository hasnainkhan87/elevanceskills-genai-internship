"""
preprocess_arxiv.py
=====================
Converts the raw arXiv Kaggle metadata snapshot (arxiv-metadata-oai-snapshot.json,
~1.1GB+, ~2.8 million papers, one JSON object per line) into a small, filtered
CSV limited to a chosen subset of categories (default: computer science, cs.*).

WHY STREAMING + RESERVOIR SAMPLING (not load-everything-then-filter)
-----------------------------------------------------------------------
The full file is far too large to load into memory with pandas.read_json()
in one shot on a typical laptop. This script reads the file ONE LINE AT A
TIME (it's JSONL — one JSON object per line, not one giant JSON array), so
memory use stays flat regardless of file size.

Because we don't know in advance how many cs.* papers exist in the stream
until we've read the whole file, plain random.sample() isn't usable (it
needs the full list up front). Instead this uses RESERVOIR SAMPLING —
a classic streaming algorithm that produces a uniform random sample of a
fixed size from a stream of unknown length, in one pass, without ever
holding more than `max_rows` candidates in memory at once. This is the
correct tool for "sample N items from a huge stream," which is exactly
what a multi-GB single-pass file requires.

CATEGORY FILTER
----------------
arXiv's `categories` field is a space-separated string like "cs.CV cs.LG".
A paper is kept if ANY of its categories start with one of the configured
prefixes (default: just "cs." for computer science, per the task brief's
example). Change CATEGORY_PREFIXES to target a different field.

OUTPUT COLUMNS
---------------
prompt, response, categories, arxiv_id, authors, update_date

`prompt` = title, `response` = abstract — same naming convention as
Task 1 and Task 3's CSVs, so the same CSVLoader(source_column="prompt")
pattern works unchanged for the "paper lookup by title" retrieval use case.
(Note: unlike Task 1/3's short FAQ answers, `response` here is a full
paper abstract — usually a paragraph, not a one-line answer. The RAG
prompt in langchain_helper.py is written with that in mind.)

RUN
---
    python src/preprocess_arxiv.py --input path/to/arxiv-metadata-oai-snapshot.json

Output: dataset/arxiv_cs_sample.csv
"""

import argparse
import csv
import json
import random
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = BASE_DIR / "dataset" / "arxiv_cs_sample.csv"

RANDOM_SEED = 42
DEFAULT_MAX_ROWS = 300  # kept small on purpose — see README for why
CATEGORY_PREFIXES = ("cs.",)  # e.g. add "stat.ML" or "eess." to broaden


def clean_whitespace(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def matches_category(categories_field: str) -> bool:
    if not categories_field:
        return False
    cats = categories_field.split()
    return any(cat.startswith(CATEGORY_PREFIXES) for cat in cats)


def format_authors(record: dict) -> str:
    """arXiv's 'authors' field is a free-text string; prefer it directly."""
    return clean_whitespace(record.get("authors", ""))


def reservoir_sample_jsonl(input_path: Path, max_rows: int, seed: int = RANDOM_SEED):
    """
    Single-pass reservoir sampling over a JSONL file, keeping only records
    whose categories match CATEGORY_PREFIXES.

    Returns a list of up to max_rows row-dicts, uniformly sampled from all
    matching records seen in the stream.
    """
    rng = random.Random(seed)
    reservoir = []
    seen_matching = 0
    total_lines = 0
    parse_errors = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue

            categories = record.get("categories", "")
            if not matches_category(categories):
                continue

            title = clean_whitespace(record.get("title", ""))
            abstract = clean_whitespace(record.get("abstract", ""))
            if not title or not abstract or len(abstract) < 40:
                continue  # skip incomplete records

            row = {
                "prompt": title,
                "response": abstract,
                "categories": clean_whitespace(categories),
                "arxiv_id": record.get("id", ""),
                "authors": format_authors(record),
                "update_date": record.get("update_date", ""),
            }

            seen_matching += 1
            if len(reservoir) < max_rows:
                reservoir.append(row)
            else:
                # classic reservoir sampling: replace a random existing
                # element with decreasing probability as the stream grows
                j = rng.randint(0, seen_matching - 1)
                if j < max_rows:
                    reservoir[j] = row

            if total_lines % 200000 == 0:
                print(f"  ...scanned {total_lines:,} lines, "
                      f"{seen_matching:,} matching so far, reservoir size {len(reservoir)}")

    print(f"\nScanned {total_lines:,} total lines ({parse_errors} unparseable).")
    print(f"Found {seen_matching:,} papers matching categories {CATEGORY_PREFIXES}.")
    print(f"Sampled {len(reservoir)} rows (seed={seed}).")
    return reservoir


def save_csv(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["prompt", "response", "categories", "arxiv_id", "authors", "update_date"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess arXiv JSONL into a filtered, sampled CSV")
    parser.add_argument("--input", required=True, help="Path to arxiv-metadata-oai-snapshot.json")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CSV path")
    parser.add_argument("--max_rows", type=int, default=DEFAULT_MAX_ROWS,
                         help=f"Max papers to sample (default: {DEFAULT_MAX_ROWS})")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} not found. Download arxiv-metadata-oai-snapshot.json from "
            "https://www.kaggle.com/datasets/Cornell-University/arxiv first."
        )

    print(f"Streaming {input_path} (this may take a few minutes for a multi-GB file)...")
    rows = reservoir_sample_jsonl(input_path, args.max_rows)

    if not rows:
        raise ValueError("No matching papers found — check CATEGORY_PREFIXES and the input file.")

    save_csv(rows, Path(args.output))

    print("\nPreview of first 3 rows:")
    for row in rows[:3]:
        print(f"  Title:      {row['prompt'][:80]}")
        print(f"  Categories: {row['categories']}")
        print(f"  Abstract:   {row['response'][:100]}...")
        print()


if __name__ == "__main__":
    main()

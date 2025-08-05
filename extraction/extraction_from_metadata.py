#!/usr/bin/env python3

import os
import json
import pdfplumber
import requests
import re
from jsonschema import validate, ValidationError

# ===== Configuration =====

BUCKET = "indian-high-court-judgments"
SRC_ROOT = "sorted_cases/civil/2025/36_29/taphc"
OUTPUT_JSONL = "8/5/outputs/civil_2025_1_12/all_cases.jsonl"
LOCAL_PDF_DIR = "./downloaded_pdfs"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
YEAR = "2025"
BENCH = "taphc"  # manually set bench name

# Minimal JSON schema for validation
SCHEMA = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["instruction", "question", "answer", "metadata"],
  "properties": {
    "instruction": {"type": "string"},
    "question": {"type": "string"},
    "answer": {
      "type": "object",
      "required": ["Judgment Facts", "Reasoning & Background", "Verdict Results", "Relevant Statutes"],
      "properties": {
        "Judgment Facts": {"type": "string"},
        "Reasoning & Background": {"type": "string"},
        "Verdict Results": {"type": "string"},
        "Relevant Statutes": {
          "type": "array",
          "items": {"type": "string"}
        }
      },
      "additionalProperties": False
    },
    "metadata": {
      "type": "object",
      "required": [
        "court_code", "title", "description", "judge",
        "cnr", "date_of_registration", "decision_date",
        "disposal_nature", "court"
      ],
      "properties": {
        "court_code": {"type": "string"},
        "title": {"type": "string"},
        "description": {"type": "string"},
        "judge": {"type": "string"},
        "cnr": {"type": "string"},
        "date_of_registration": {"type": "string", "format": "date"},
        "decision_date": {"type": "string", "format": "date"},
        "disposal_nature": {"type": "string"},
        "court": {"type": "string"}
      },
      "additionalProperties": False
    }
  },
  "additionalProperties": False
}


def download_from_s3(s3_key: str, dest_path: str):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    cmd = f"aws s3 cp s3://{BUCKET}/{s3_key} {dest_path} --no-sign-request"
    if os.system(cmd) != 0:
        raise RuntimeError(f"Failed to download s3://{BUCKET}/{s3_key}")


def extract_pdf_text(pdf_path: str) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def call_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        print("⚠️ GEMINI_API_KEY not set; skipping Gemini call.")
        return ""
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"
    response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
    response.raise_for_status()
    data = response.json()
    candidates = data.get("candidates", [])
    if candidates:
        return candidates[0]["content"]["parts"][0]["text"]
    return ""


def extract_json_from_text(text: str):
    # Return empty structure on failure
    default = {
        "Judgment Facts": "",
        "Reasoning & Background": "",
        "Verdict Results": "",
        "Relevant Statutes": []
    }
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return default



def build_prompt(metadata: dict, full_text: str) -> str:
    return f"""You are a precise legal assistant AI.

Case Metadata:
- Title: {metadata.get('title')}
- Court code: {metadata.get('court_code')}
- CNR: {metadata.get('cnr')}
- Judge: {metadata.get('judge')}
- Decision Date: {metadata.get('decision_date')}

Extract as valid JSON ONLY with these fields:
- Judgment Facts (string)
- Reasoning & Background (string)
- Verdict Results (string)
- Relevant Statutes (array of strings)

Judgment Text:
{full_text}
"""


def enhance_output_for_readability(answer: dict):
    # Simple normalization: ensure four fields exist
    for field in ["Judgment Facts", "Reasoning & Background", "Verdict Results", "Relevant Statutes"]:
        answer.setdefault(field, "" if field != "Relevant Statutes" else [])
    return answer


def prune_metadata(meta):
    allowed_keys = [
        "court_code", "title", "description", "judge",
        "cnr", "date_of_registration", "decision_date",
        "disposal_nature", "court"
    ]
    return {k: meta[k] for k in allowed_keys if k in meta}


def process_metadata_json(json_filepath: str):
    import os
    from jsonschema import validate, ValidationError

    # Load full metadata JSON
    meta = json.load(open(json_filepath, "r", encoding="utf-8"))

    # Extract pdf_link and validate presence before pruning
    pdf_link = meta.get("pdf_link")
    if not pdf_link or not pdf_link.strip():
        print(f"Skipping {json_filepath}: pdf_link is empty or missing")
        return None

    pdf_filename = os.path.basename(pdf_link)  # extract filename only
    court_code = meta.get("court_code", "").replace("~", "_")
    s3_key = f"data/pdf/year={YEAR}/court={court_code}/bench={BENCH}/{pdf_filename}"
    local_pdf_path = os.path.join(LOCAL_PDF_DIR, pdf_filename)

    try:
        download_from_s3(s3_key, local_pdf_path)
        full_text = extract_pdf_text(local_pdf_path)
    except Exception as e:
        print(f"Error processing PDF for {json_filepath}: {e}")
        return None

    prompt = build_prompt(meta, full_text)
    raw_response = call_gemini(prompt)
    answer = extract_json_from_text(raw_response)
    answer = enhance_output_for_readability(answer)

    # Prune metadata only now for output and validation
    meta_pruned = prune_metadata(meta)

    output_entry = {
        "instruction": "Extract the four labeled sections (Judgment Facts, Reasoning & Background, Verdict Results, Relevant Statutes) from this judgment.",
        "question": f"Parse the judgment text of '{meta.get('title')}'.",
        "answer": answer,
        "metadata": meta_pruned
    }

    # Validate output_entry against schema
    try:
        validate(instance=output_entry, schema=SCHEMA)
    except ValidationError as ve:
        print(f"Schema validation failed for {json_filepath}: {ve.message}")
        return None

    return output_entry



def load_processed_titles(path: str):
    titles = set()
    if not os.path.exists(path):
        return titles
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                t = obj.get("metadata", {}).get("title")
                if t:
                    titles.add(t)
            except:
                continue
    return titles


def main():
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    os.makedirs(LOCAL_PDF_DIR, exist_ok=True)

    processed_titles = load_processed_titles(OUTPUT_JSONL)
    written = 0

    with open(OUTPUT_JSONL, "a", encoding="utf-8") as out_f:
        for root, _, files in os.walk(SRC_ROOT):
            for fname in files:
                if not fname.endswith(".json"):
                    continue
                path = os.path.join(root, fname)
                meta = json.load(open(path, "r", encoding="utf-8"))
                title = meta.get("title")
                if title in processed_titles:
                    continue

                print(f"Processing: {title}")
                result = process_metadata_json(path)
                if result:
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_f.flush()
                    processed_titles.add(title)
                    written += 1

    print(f"\nCompleted. {written} new cases written to {OUTPUT_JSONL}.")


if __name__ == "__main__":
    main()

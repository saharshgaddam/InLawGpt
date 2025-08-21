
# import os
# import json
# import time
# import asyncio
# import aiohttp
# import pdfplumber
# import traceback
# from jsonschema import validate, ValidationError

# # Configuration
# BUCKET = "indian-high-court-judgments"
# SRC_ROOT = "sorted_cases/criminal/2025/36_29/taphc"
# OUTPUT_JSONL = "outputs/criminal_2025_final_batched.jsonl"
# LOCAL_PDF_DIR = "./downloaded_pdfs"
# PROGRESS_FILE = "outputs/process_progress.json"

# MAX_CONCURRENCY = 3
# BATCH_MAX_ITEMS = 3
# BATCH_MAX_CHARS = 12000
# PDF_TEXT_CHAR_LIMIT = 5000
# RETRY_CYCLES = 4
# RETRY_BACKOFF_BASE = 2.5
# RETRY_BACKOFF_MAX = 30.0

# GEMINI_API_KEY = "AIzaSyCFSCT2-L4kah02Zv14b6HNCjW7mupfgm0"
# GEMINI_MODEL = "gemini-2.5-flash-lite"
# GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

# SCHEMA = {
#     "$schema": "http://json-schema.org/draft-07/schema#",
#     "type": "object",
#     "required": [
#         "metadata", "Judgment Facts", "Reasoning & Background",
#         "Verdict Results", "Relevant Statutes", "Subcategory"
#     ],
#     "properties": {
#         "metadata": {
#             "type": "object",
#             "required": [
#                 "court_code", "title", "description", "judge",
#                 "cnr", "date_of_registration", "decision_date",
#                 "disposal_nature", "court", "case_type"
#             ],
#             "properties": {
#                 "court_code": {"type": "string"},
#                 "title": {"type": "string"},
#                 "description": {"type": "string"},
#                 "judge": {"type": "string"},
#                 "cnr": {"type": "string"},
#                 "date_of_registration": {"type": "string"},
#                 "decision_date": {"type": "string"},
#                 "disposal_nature": {"type": "string"},
#                 "court": {"type": "string"},
#                 "case_type": {"type": "string"}
#             },
#             "additionalProperties": False
#         },
#         "Judgment Facts": {"type": "array", "items": {"type": "string"}},
#         "Reasoning & Background": {"type": "array", "items": {"type": "string"}},
#         "Verdict Results": {"type": "array", "items": {"type": "string"}},
#         "Relevant Statutes": {"type": "array", "items": {"type": "string"}},
#         "Subcategory": {"type": "string"}
#     },
#     "additionalProperties": False
# }

# ALLOWED_META_KEYS = {
#     "court_code", "title", "description", "judge",
#     "cnr", "date_of_registration", "decision_date",
#     "disposal_nature", "court", "case_type"
# }

# def atomic_write_json(path: str, data):
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     tmp_path = path + ".tmp"
#     with open(tmp_path, "w", encoding="utf-8") as f:
#         json.dump(data, f, ensure_ascii=False, indent=2)
#     os.replace(tmp_path, path)

# def safe_load_json(path: str, default):
#     if not os.path.exists(path) or os.path.getsize(path) == 0:
#         return default
#     try:
#         with open(path, "r", encoding="utf-8") as f:
#             content = f.read().strip()
#             if not content:
#                 return default
#             return json.loads(content)
#     except Exception:
#         try:
#             os.replace(path, path + ".bad")
#         except Exception:
#             pass
#         return default

# def prune_metadata_strict(meta: dict) -> dict:
#     pruned = {k: v for k, v in meta.items() if k in ALLOWED_META_KEYS}
#     if "case_type" not in pruned or not pruned["case_type"].strip():
#         pruned["case_type"] = "criminal"
#     return pruned

# async def download_pdf(s3_key: str, local_path: str):
#     if os.path.exists(local_path):
#         return
#     import subprocess
#     loop = asyncio.get_event_loop()
#     def run_cmd():
#         cmd = [
#             "aws", "s3", "cp",
#             f"s3://{BUCKET}/{s3_key}",
#             local_path,
#             "--no-sign-request"
#         ]
#         return subprocess.run(cmd, capture_output=True)
#     result = await loop.run_in_executor(None, run_cmd)
#     if result.returncode != 0:
#         raise RuntimeError(f"AWS CLI download failed for {s3_key}: {result.stderr.decode()}")

# async def extract_pdf_text(pdf_path: str) -> str:
#     def sync_extract():
#         with pdfplumber.open(pdf_path) as pdf:
#             texts = []
#             for page in pdf.pages:
#                 texts.append(page.extract_text() or "")
#             full_text = "\n\n".join(texts)
#             return full_text[:PDF_TEXT_CHAR_LIMIT]
#     loop = asyncio.get_event_loop()
#     try:
#         text = await asyncio.wait_for(loop.run_in_executor(None, sync_extract), timeout=30)
#         return text
#     except Exception as e:
#         print(f"[ERROR] PDF extraction timeout or error: {e}")
#         return ""

# async def call_gemini(session: aiohttp.ClientSession, prompt: str):
#     payload = {
#         "contents": [{"parts": [{"text": prompt}]}],
#         "generationConfig": {"response_mime_type": "application/json"}
#     }
#     for attempt in range(RETRY_CYCLES):
#         try:
#             async with session.post(GEMINI_URL, json=payload, timeout=120) as resp:
#                 resp.raise_for_status()
#                 data = await resp.json()
#                 # out_text = data["candidates"][0]["content"]["parts"]["text"]
#                 # return out_text
#                 candidates = data.get("candidates", [])
#                 if not candidates:
#                     raise RuntimeError("No output from Gemini API")
#                 # Correct indexing here:
#                 return candidates[0]["content"]["parts"][0]["text"]
#         except (aiohttp.ClientError, asyncio.TimeoutError) as e:
#             wait = min(RETRY_BACKOFF_BASE ** attempt, RETRY_BACKOFF_MAX)
#             print(f"[WARN] Gemini call failed, retry {attempt+1}/{RETRY_CYCLES}, waiting {wait}s: {e}")
#             await asyncio.sleep(wait)
#     raise RuntimeError("Gemini API call failed after retries")

# def extract_json_from_text(text: str):
#     try:
#         start = text.index("[")
#         end = text.rindex("]") + 1
#         obj = json.loads(text[start:end])
#         for rec in obj:
#             for field in ["Judgment Facts", "Reasoning & Background", "Verdict Results"]:
#                 if field in rec and isinstance(rec[field], str):
#                     paras = [p.strip() for p in rec[field].split("\n\n") if p.strip()]
#                     if not paras:
#                         paras = [p.strip() for p in rec[field].split("\n") if p.strip()]
#                     rec[field] = paras
#             if "Subcategory" not in rec or not isinstance(rec.get("Subcategory"), str):
#                 rec["Subcategory"] = ""
#         return obj
#     except Exception as e:
#         print(f"[ERROR] Failed to parse Gemini output JSON: {e}")
#         return []

# def build_batch_prompt(batch_cases):
#     instruction = """
# You are an expert legal assistant extracting detailed information from Indian High Court criminal judgments.
# For each case below, output a JSON object with fields:
# - Judgment Facts (array of paragraphs)
# - Reasoning & Background (array of paragraphs)
# - Verdict Results (array of paragraphs)
# - Relevant Statutes (array of strings)
# - Subcategory (free text; criminal subcategory)
# Output a JSON array in the exact same order as input cases.
# """
#     input_cases = [{
#         "title": case["metadata"].get("title", ""),
#         "court_code": case["metadata"].get("court_code", ""),
#         "cnr": case["metadata"].get("cnr", ""),
#         "judge": case["metadata"].get("judge", ""),
#         "decision_date": case["metadata"].get("decision_date", ""),
#         "text": case["text"]
#     } for case in batch_cases]

#     return instruction + "\nCases:\n" + json.dumps(input_cases, ensure_ascii=False)

# async def process_batch(session, batch_cases, out_file, processed_set):
#     prompt = build_batch_prompt(batch_cases)
#     try:
#         response_text = await call_gemini(session, prompt)
#         results = extract_json_from_text(response_text)
#         for case, res in zip(batch_cases, results):
#             output_record = {
#                 "metadata": prune_metadata_strict(case["metadata"]),
#                 "Judgment Facts": res.get("Judgment Facts", []),
#                 "Reasoning & Background": res.get("Reasoning & Background", []),
#                 "Verdict Results": res.get("Verdict Results", []),
#                 "Relevant Statutes": res.get("Relevant Statutes", []),
#                 "Subcategory": res.get("Subcategory", "")
#             }
#             try:
#                 validate(instance=output_record, schema=SCHEMA)
#             except ValidationError as e:
#                 print(f"[ERROR] Validation failed for case {case['metadata'].get('title', 'N/A')}: {e.message}")
#                 continue
#             out_file.write(json.dumps(output_record, ensure_ascii=False) + "\n")
#             out_file.flush()
#             processed_set.add(case["metadata"].get("title"))
#     except Exception as e:
#         print(f"[ERROR] Batch processing error: {e}")
#         traceback.print_exc()

# async def main_async():
#     os.makedirs(LOCAL_PDF_DIR, exist_ok=True)
#     os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
#     processed_titles = safe_load_json(PROGRESS_FILE, set())
#     if isinstance(processed_titles, list):
#         processed_titles = set(processed_titles)

#     batch_cases = []
#     batch_chars = 0
#     semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

#     async def process_and_wait(session, cases, out_file, processed_set):
#         async with semaphore:
#             await process_batch(session, cases, out_file, processed_set)

#     tasks = []

#     async with aiohttp.ClientSession() as session:
#         with open(OUTPUT_JSONL, "a", encoding="utf-8") as out_file:
#             for root, _, files in os.walk(SRC_ROOT):
#                 for fname in files:
#                     if not fname.lower().endswith(".json"):
#                         continue

#                     path = os.path.join(root, fname)
#                     meta = json.load(open(path, "r", encoding="utf-8"))
#                     title = meta.get("title")
#                     if not title or title in processed_titles:
#                         continue

#                     # Filter criminal cases by heuristic
#                     case_type = meta.get("case_type", "").strip().lower()
#                     if case_type != "criminal":
#                         desc = meta.get("description", "").lower()
#                         title_lc = title.lower()
#                         keywords = ["fir", "ipc", "crpc", "accused", "charge", "bail", "criminal", "offence", "murder", "robbery"]
#                         if not any(k in desc or k in title_lc for k in keywords):
#                             print(f"[SKIP] Non-criminal or unknown case_type: {title}")
#                             continue

#                     pdf_link = meta.get("pdf_link") or ""
#                     if not pdf_link.strip():
#                         print(f"[SKIP] Missing PDF link: {title}")
#                         continue

#                     pdf_filename = os.path.basename(pdf_link)
#                     local_pdf_path = os.path.join(LOCAL_PDF_DIR, pdf_filename)
#                     s3_key = f"data/pdf/year=2025/court={meta.get('court_code','').replace('~','_')}/bench=taphc/{pdf_filename}"

#                     try:
#                         await download_pdf(s3_key, local_pdf_path)
#                     except Exception as e:
#                         print(f"[ERROR] PDF download failed for {title}: {e}")
#                         continue

#                     pdf_text = await extract_pdf_text(local_pdf_path)
#                     if not pdf_text.strip():
#                         print(f"[WARN] Extracted empty text for {title}")
#                         continue

#                     case_record = {"metadata": meta, "text": pdf_text}
#                     serialize_len = len(json.dumps(case_record, ensure_ascii=False)) + 2000  # Approx overhead

#                     if (len(batch_cases) >= BATCH_MAX_ITEMS) or (batch_chars + serialize_len > BATCH_MAX_CHARS):
#                         task = asyncio.create_task(process_and_wait(session, batch_cases, out_file, processed_titles))
#                         tasks.append(task)
#                         batch_cases = []
#                         batch_chars = 0

#                     batch_cases.append(case_record)
#                     batch_chars += serialize_len

#             if batch_cases:
#                 task = asyncio.create_task(process_and_wait(session, batch_cases, out_file, processed_titles))
#                 tasks.append(task)

#             if tasks:
#                 await asyncio.gather(*tasks)

#     atomic_write_json(PROGRESS_FILE, list(processed_titles))
#     print(f"[SUMMARY] Completed processing {len(processed_titles)} cases.")

# def main():
#     if not GEMINI_API_KEY:
#         raise RuntimeError("Please set GEMINI_API_KEY environment variable.")
#     start_time = time.time()
#     try:
#         asyncio.run(main_async())
#     except Exception as e:
#         print(f"[FATAL] Exception: {e}")
#         traceback.print_exc()
#     finally:
#         print(f"[INFO] Elapsed time: {time.time() - start_time:.1f}s")

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3

import os
import json
import time
import asyncio
import aiohttp
import pdfplumber
import traceback
from jsonschema import validate, ValidationError

# ====== Configuration ======
BUCKET = "indian-high-court-judgments"
SRC_ROOT = "sorted_cases/criminal/2025/36_29/taphc"  # Process only criminal cases
OUTPUT_JSONL = "outputs/criminal_2025_final_batched.jsonl"
LOCAL_PDF_DIR = "./downloaded_pdfs"
PROGRESS_FILE = "outputs/process_progress.json"

MAX_CONCURRENCY = 3  # Limit concurrency to avoid rate limiting
BATCH_MAX_ITEMS = 4  # Batch size to reduce API calls
BATCH_MAX_CHARS = 12000  # Rough character token estimate limit for batching
PDF_TEXT_CHAR_LIMIT = 20000
RETRY_CYCLES = 4
RETRY_BACKOFF_BASE = 2.5
RETRY_BACKOFF_MAX = 30.0

GEMINI_API_KEY = "AIzaSyCFSCT2-L4kah02Zv14b6HNCjW7mupfgm0"
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": [
        "metadata", "Judgment Facts", "Reasoning & Background",
        "Verdict Results", "Relevant Statutes", "Subcategory"
    ],
    "properties": {
        "metadata": {
            "type": "object",
            "required": [
                "court_code", "title", "description", "judge",
                "cnr", "date_of_registration", "decision_date",
                "disposal_nature", "court", "case_type"
            ],
            "properties": {
                "court_code": {"type": "string"},
                "title": {"type": "string"},
                "description": {"type": "string"},
                "judge": {"type": "string"},
                "cnr": {"type": "string"},
                "date_of_registration": {"type": "string"},
                "decision_date": {"type": "string"},
                "disposal_nature": {"type": "string"},
                "court": {"type": "string"},
                "case_type": {"type": "string"}
            },
            "additionalProperties": False
        },
        "Judgment Facts": {"type": "array", "items": {"type": "string"}},
        "Reasoning & Background": {"type": "array", "items": {"type": "string"}},
        "Verdict Results": {"type": "array", "items": {"type": "string"}},
        "Relevant Statutes": {"type": "array", "items": {"type": "string"}},
        "Subcategory": {"type": "string"}
    },
    "additionalProperties": False
}

ALLOWED_META_KEYS = {
    "court_code", "title", "description", "judge",
    "cnr", "date_of_registration", "decision_date",
    "disposal_nature", "court", "case_type"
}

# --- Helpers ---

def atomic_write_json(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)

def safe_load_json(path: str, default):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return default
            return json.loads(content)
    except Exception:
        try:
            os.replace(path, path + ".bad")
        except Exception:
            pass
        return default

def prune_metadata_strict(meta: dict) -> dict:
    pruned = {k: v for k, v in meta.items() if k in ALLOWED_META_KEYS}
    # Force all cases to have 'case_type' as 'criminal'
    pruned["case_type"] = "criminal"
    return pruned

async def download_pdf(s3_key: str, local_path: str):
    if os.path.exists(local_path):
        return
    import subprocess
    loop = asyncio.get_event_loop()
    def run_cmd():
        cmd = [
            "aws", "s3", "cp",
            f"s3://{BUCKET}/{s3_key}",
            local_path,
            "--no-sign-request"
        ]
        return subprocess.run(cmd, capture_output=True)
    result = await loop.run_in_executor(None, run_cmd)
    if result.returncode != 0:
        raise RuntimeError(f"AWS CLI download failed for {s3_key}: {result.stderr.decode()}")

async def extract_pdf_text(pdf_path: str) -> str:
    def sync_extract():
        with pdfplumber.open(pdf_path) as pdf:
            texts = []
            for page in pdf.pages:
                texts.append(page.extract_text() or "")
            full_text = "\n\n".join(texts)
            return full_text[:PDF_TEXT_CHAR_LIMIT]
    loop = asyncio.get_event_loop()
    try:
        text = await asyncio.wait_for(loop.run_in_executor(None, sync_extract), timeout=30)
        return text
    except Exception as e:
        print(f"[ERROR] PDF extraction timeout or error: {e}")
        return ""

# async def call_gemini(session: aiohttp.ClientSession, prompt: str):
#     payload = {
#         "contents": [{"parts": [{"text": prompt}]}],
#         "generationConfig": {"response_mime_type": "application/json"}
#     }
#     for attempt in range(RETRY_CYCLES):
#         try:
#             async with session.post(GEMINI_URL, json=payload, timeout=120) as resp:
#                 resp.raise_for_status()
#                 data = await resp.json()
#                 out_text = data["candidates"][0]["content"]["parts"][0]["text"]
#                 return out_text
#         except (aiohttp.ClientError, asyncio.TimeoutError) as e:
#             wait = min(RETRY_BACKOFF_BASE ** attempt, RETRY_BACKOFF_MAX)
#             print(f"[WARN] Gemini call failed, retry {attempt+1}/{RETRY_CYCLES}, waiting {wait}s: {e}")
#             await asyncio.sleep(wait)
#     raise RuntimeError("Gemini API call failed after retries")
async def call_gemini(session: aiohttp.ClientSession, prompt: str):
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"response_mime_type": "application/json"}
    }
    for attempt in range(RETRY_CYCLES):
        try:
            async with session.post(GEMINI_URL, json=payload, timeout=120) as resp:
                resp.raise_for_status()
                data = await resp.json()
                # out_text = data["candidates"][0]["content"]["parts"]["text"]
                # return out_text
                candidates = data.get("candidates", [])
                if not candidates:
                    raise RuntimeError("No output from Gemini API")
                # Correct indexing here:
                return candidates[0]["content"]["parts"][0]["text"]
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            wait = min(RETRY_BACKOFF_BASE ** attempt, RETRY_BACKOFF_MAX)
            print(f"[WARN] Gemini call failed, retry {attempt+1}/{RETRY_CYCLES}, waiting {wait}s: {e}")
            await asyncio.sleep(wait)
    raise RuntimeError("Gemini API call failed after retries")

def extract_json_from_text(text: str):
    try:
        start = text.index("[")
        end = text.rindex("]") + 1
        obj = json.loads(text[start:end])
        for rec in obj:
            for field in ["Judgment Facts", "Reasoning & Background", "Verdict Results"]:
                if field in rec and isinstance(rec[field], str):
                    paras = [p.strip() for p in rec[field].split("\n\n") if p.strip()]
                    if not paras:
                        paras = [p.strip() for p in rec[field].split("\n") if p.strip()]
                    rec[field] = paras
            if "Subcategory" not in rec or not isinstance(rec.get("Subcategory"), str):
                rec["Subcategory"] = ""
        return obj
    except Exception as e:
        print(f"[ERROR] Failed to parse Gemini output JSON: {e}")
        return []
CONTROLLED_SUBCATEGORIES = [
    "Theft",
    "Assault",
    "Murder",
    "Fraud",
    "Dowry Harassment",
    "Sexual Harassment",
    "Bail/Prostitution",
    "Drug Offenses",
    "Quashing of FIR",
    "Corruption",
    "Cyber Crime",
    "Other"
]


def build_batch_prompt(batch_cases):
    instruction = """
You are an expert legal assistant extracting detailed information from Indian High Court criminal judgments.
For each case below, output a JSON object with fields:
- Judgment Facts (array of paragraphs)
- Reasoning & Background (array of paragraphs)
- Verdict Results (array of paragraphs)
- Relevant Statutes (array of strings)
- Subcategory (string): choose exactly ONE subcategory from the following list:

[Theft, Assault, Murder, Robbery, Fraud, Dowry Harassment, Domestic Violence, Sexual Harassment, Rape, Bail/Prostitution, Drug Offenses, Cheating, Criminal Breach of Trust, Quashing of FIR, Investigation Procedures, Corruption, Cyber Crime]

If the case does not perfectly match any category, choose the closest related one from the list above. Do NOT create your own categories or select any category outside this list.
Output a JSON array in the exact same order as input cases.
"""
    input_cases = [{
        "title": case["metadata"].get("title", ""),
        "court_code": case["metadata"].get("court_code", ""),
        "cnr": case["metadata"].get("cnr", ""),
        "judge": case["metadata"].get("judge", ""),
        "decision_date": case["metadata"].get("decision_date", ""),
        "text": case["text"]
    } for case in batch_cases]

    return instruction + "\nCases:\n" + json.dumps(input_cases, ensure_ascii=False)

async def process_batch(session, batch_cases, out_file, processed_set):
    prompt = build_batch_prompt(batch_cases)
    try:
        response_text = await call_gemini(session, prompt)
        results = extract_json_from_text(response_text)
        for case, res in zip(batch_cases, results):
            output_record = {
                "metadata": prune_metadata_strict(case["metadata"]),
                "Judgment Facts": res.get("Judgment Facts", []),
                "Reasoning & Background": res.get("Reasoning & Background", []),
                "Verdict Results": res.get("Verdict Results", []),
                "Relevant Statutes": res.get("Relevant Statutes", []),
                "Subcategory": res.get("Subcategory", "")
            }
            try:
                validate(instance=output_record, schema=SCHEMA)
            except ValidationError as e:
                print(f"[ERROR] Validation failed for case {case['metadata'].get('title', 'N/A')}: {e.message}")
                continue
            out_file.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            out_file.flush()
            processed_set.add(case["metadata"].get("title"))
    except Exception as e:
        print(f"[ERROR] Batch processing error: {e}")
        traceback.print_exc()

async def main_async():
    os.makedirs(LOCAL_PDF_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    processed_titles = safe_load_json(PROGRESS_FILE, set())
    if isinstance(processed_titles, list):
        processed_titles = set(processed_titles)

    batch_cases = []
    batch_chars = 0
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def process_and_wait(session, cases, out_file, processed_set):
        async with semaphore:
            await process_batch(session, cases, out_file, processed_set)

    tasks = []

    async with aiohttp.ClientSession() as session:
        with open(OUTPUT_JSONL, "a", encoding="utf-8") as out_file:
            for root, _, files in os.walk(SRC_ROOT):
                for fname in files:
                    if not fname.lower().endswith(".json"):
                        continue

                    path = os.path.join(root, fname)
                    meta = json.load(open(path, "r", encoding="utf-8"))
                    title = meta.get("title")
                    if not title or title in processed_titles:
                        continue

                    # Since all cases are criminal by default, no filtering here

                    pdf_link = meta.get("pdf_link") or ""
                    if not pdf_link.strip():
                        print(f"[SKIP] Missing PDF link: {title}")
                        continue

                    pdf_filename = os.path.basename(pdf_link)
                    local_pdf_path = os.path.join(LOCAL_PDF_DIR, pdf_filename)
                    s3_key = f"data/pdf/year=2025/court={meta.get('court_code','').replace('~','_')}/bench=taphc/{pdf_filename}"

                    try:
                        await download_pdf(s3_key, local_pdf_path)
                    except Exception as e:
                        print(f"[ERROR] PDF download failed for {title}: {e}")
                        continue

                    pdf_text = await extract_pdf_text(local_pdf_path)
                    if not pdf_text.strip():
                        print(f"[WARN] Extracted empty text for {title}")
                        continue

                    case_record = {"metadata": meta, "text": pdf_text}
                    serialize_len = len(json.dumps(case_record, ensure_ascii=False)) + 2000

                    if (len(batch_cases) >= BATCH_MAX_ITEMS) or (batch_chars + serialize_len > BATCH_MAX_CHARS):
                        task = asyncio.create_task(process_and_wait(session, batch_cases, out_file, processed_titles))
                        tasks.append(task)
                        batch_cases = []
                        batch_chars = 0

                    batch_cases.append(case_record)
                    batch_chars += serialize_len

            if batch_cases:
                task = asyncio.create_task(process_and_wait(session, batch_cases, out_file, processed_titles))
                tasks.append(task)

            if tasks:
                await asyncio.gather(*tasks)

    atomic_write_json(PROGRESS_FILE, list(processed_titles))
    print(f"[SUMMARY] Completed processing {len(processed_titles)} cases.")

def main():
    if not GEMINI_API_KEY:
        raise RuntimeError("Please set GEMINI_API_KEY environment variable.")
    start_time = time.time()
    try:
        asyncio.run(main_async())
    except Exception as e:
        print(f"[FATAL] Exception: {e}")
        traceback.print_exc()
    finally:
        print(f"[INFO] Elapsed time: {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    main()

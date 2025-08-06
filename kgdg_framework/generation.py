import json
import numpy as np
import argparse
import os
import time
import traceback
import multiprocessing
from google.generativeai import GenerativeModel
import google.generativeai as genai


# ========== ARGUMENTS ==========
parser = argparse.ArgumentParser()
parser.add_argument('--cores', default=1, type=int, help="Number of parallel workers")
parser.add_argument('--nums', default=10, type=int, help="Number of seed problems to process/generate each run")
parser.add_argument('--suffix', default=None, type=str, help="Suffix for output file name")
parser.add_argument('--cases_dir', default="extraction/all_cases.jsonl", type=str, help="File path of legal cases JSONL")
parser.add_argument('--seed_file', default="seed.json", type=str, help="Seed problems JSON file path")
parser.add_argument('--output_dir', default="./generation", type=str, help="Output directory for generation results")
parser.add_argument('--api_key', required=False, type=str, help="Google Gemini API key")


args = parser.parse_args()


# ========== SETUP GEMINI ==========
API_KEY = "AIzaSyA5h1sYXwxvZrl4RUPNISLMx1WLt2B0wyE"  # Paste your real API key here
genai.configure(api_key=API_KEY)
model = GenerativeModel("gemini-1.5-flash")  # Change model as per availability


# ========== CONSTANTS ==========


MAX_PROMPT_TOKENS = 8000  # Adjust according to Gemini limits
RETRY_LIMIT = 3
RETRY_BACKOFF_BASE = 2  # Exponential backoff base (seconds)


# ========== PROMPTS ==========


CONTROL_PROMPT = """
You are given a JSON-formatted legal question and answer pair in English from the Indian civil legal domain.
{JSON}

Please respond with the type of legal document you want to use to generate a similar question-answer.  
Since your database consists solely of Indian Civil Legal Documents, please respond with:
{{"type": "Civil Legal Document"}}
"""


GENERATE_PROMPT = """
You are given the following JSON legal Q&A pair from Indian civil law domain:

{JSON}

Based on the following legal case content:

{DOCS}

Generate a new question and answer pair in JSON format with these constraints:
- Keep the instruction field unchanged.
- Create a new, relevant question and detailed answer consistent with the case content.
- Add a "reasoning" field: a string explaining the logical steps leading to the answer.
- Add a "references" field: a dictionary with keys as legal statutes/rules mentioned, and values as short descriptions or relevant sections.
- Modify names, places, and sensitive info to anonymize parties.
- Do NOT copy the case content verbatim; paraphrase and adapt.
- The answer must fully address the instruction and question.

Return the JSON of the new Q&A pair only.
"""


# ========== UTILS ==========


def count_tokens(text):
    # Simple proxy: count words separated by whitespace (adjust to Gemini tokenizer later if needed)
    return len(text.split())


def truncate_text(text, max_words):
    words = text.split()
    return " ".join(words[:max_words])


def build_prompt(seed_json, case_text, prompt_template):
    return prompt_template.format(JSON=json.dumps(seed_json, ensure_ascii=False), DOCS=case_text)


def save_results_atomic(results, out_path):
    temp_path = out_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as fw:
        json.dump(results, fw, ensure_ascii=False, indent=2)
    os.replace(temp_path, out_path)


def clean_json_response(reply):
    """
    Remove markdown code-fencing from a Gemini/LLM JSON reply if present,
    and parse it as a JSON object.
    """
    reply = reply.strip()
    if reply.startswith("```json"):
        reply = reply[len("```json"):].strip()
    elif reply.startswith("```"):
        reply = reply[len("```"):].strip()
    if reply.endswith("```"):
        reply = reply[:-3].strip()
    return json.loads(reply)


# ========== LOAD KNOWLEDGE BASE ==========


def load_knowledge_base_jsonl(file_path):
    cases = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                case_json = json.loads(line)
                cases.append(case_json)
            except Exception as e:
                print(f"Failed to parse line: {e}")
    print(f"Loaded {len(cases)} legal cases from '{file_path}'")
    return cases


knowledge_base = load_knowledge_base_jsonl(args.cases_dir)


# ========== SAMPLE CASE FUNCTION ==========


# def sample_civil_case():
#     if not knowledge_base:
#         return ""
#     case = knowledge_base[np.random.randint(len(knowledge_base))]
#     facts = case.get("answer", {}).get("Judgment Facts", "")
#     reasoning = case.get("answer", {}).get("Reasoning & Background", "")
#     verdict = case.get("answer", {}).get("Verdict Results", "")
#     doc_text = f"Facts: {facts}\nReasoning & Background: {reasoning}\nVerdict: {verdict}"
#     # Truncate if too long to help token limit
#     if count_tokens(doc_text) > 3000:
#         doc_text = truncate_text(doc_text, 3000)
#     return doc_text
def sample_civil_case():
    if not knowledge_base:
        return None, ""
    idx = np.random.randint(len(knowledge_base))
    case = knowledge_base[idx]
    facts = case.get("answer", {}).get("Judgment Facts", "")
    reasoning = case.get("answer", {}).get("Reasoning & Background", "")
    verdict = case.get("answer", {}).get("Verdict Results", "")
    doc_text = f"Facts: {facts}\nReasoning & Background: {reasoning}\nVerdict: {verdict}"
    if count_tokens(doc_text) > 3000:
        doc_text = truncate_text(doc_text, 3000)
    return idx, doc_text



# ========== GEMINI CALL WITH RETRY ==========


def gemini_generate(prompt: str):
    for attempt in range(RETRY_LIMIT):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Gemini API error (attempt {attempt + 1}/{RETRY_LIMIT}): {e}")
            time.sleep(RETRY_BACKOFF_BASE ** attempt)
    print("Gemini API failed after retries.")
    return None


# ========== GENERATE FUNCTION PER SEED ==========


def solve(seed_question):
    try:
        # Step 1: Determine doc type (always Civil here, but keeping API call for extensibility)
        control_prompt = CONTROL_PROMPT.format(JSON=json.dumps(seed_question, ensure_ascii=False))
        control_resp = gemini_generate(control_prompt)
        if control_resp is None:
            print("No response from Gemini on control prompt.")
            return None
        try:
            # **Use clean_json_response here to handle markdown triple backticks**
            type_json = clean_json_response(control_resp)
        except Exception:
            print(f"Failed to parse control response JSON:\n{control_resp}")
            return None
        doc_type = type_json.get("type", "")
        if doc_type != "Civil Legal Document":
            print(f"Unexpected doc_type from Gemini: '{doc_type}', skipping seed.")
            return None

        # Step 2: Sample a civil legal case doc
        idx, sampled_doc = sample_civil_case()

        # Step 3: Build generation prompt, check token limits
        gen_prompt = build_prompt(seed_question, sampled_doc, GENERATE_PROMPT)
        if count_tokens(gen_prompt) > MAX_PROMPT_TOKENS:
            # Conservative truncation of sampled_doc to reduce prompt size
            allowed_tokens = MAX_PROMPT_TOKENS - count_tokens(json.dumps(seed_question, ensure_ascii=False)) - 500
            sampled_doc_short = truncate_text(sampled_doc, max(allowed_tokens, 1000))
            gen_prompt = build_prompt(seed_question, sampled_doc_short, GENERATE_PROMPT)

        # Step 4: Generate new Q&A pair
        gen_resp = gemini_generate(gen_prompt)
        if gen_resp is None:
            print("No response from Gemini on generation prompt.")
            return None

        result = clean_json_response(gen_resp)

        # Add seed info metadata for traceability
        result["source_seed_question"] = seed_question.get("question", "")[:100]
        result["doc_type"] = doc_type
        result["reference_doc_index"] = idx

        return result

    except Exception:
        print("Exception in solve function:\n" + traceback.format_exc())
        return None


# ========== MAIN ==================


def main():
    # Load seeds
    with open(args.seed_file, "r", encoding="utf-8") as f:
        seed_questions = json.load(f)

    total_seeds = len(seed_questions)
    print(f"Loaded {total_seeds} seed questions.")

    # Limit seeds to requested number
    seeds_to_process = seed_questions[: args.nums]

    # Prepare for multiprocessing
    with multiprocessing.Pool(processes=args.cores) as pool:
        start_time = time.time()
        results = pool.map(solve, seeds_to_process)
        duration = time.time() - start_time

    # Filter out None results (failed generations)
    results = [res for res in results if res is not None]

    print(f"Generation completed in {duration:.2f} seconds. Success: {len(results)}/{len(seeds_to_process)}")

    # Prepare output file path
    os.makedirs(args.output_dir, exist_ok=True)
    suffix = f"-{args.suffix}" if args.suffix else ""
    output_file = os.path.join(args.output_dir, f"GEN{suffix}.json")

    # Handle resuming: append to existing file if present
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as fr:
                existing_results = json.load(fr)
            combined_results = existing_results + results
            print(f"Appending to existing output file with {len(existing_results)} previous records.")
        except Exception:
            print(f"Failed to load existing output file {output_file}. Overwriting.")
            combined_results = results
    else:
        combined_results = results

    # Save atomically
    save_results_atomic(combined_results, output_file)
    print(f"Saved generated data to: {output_file}")


if __name__ == "__main__":
    main()

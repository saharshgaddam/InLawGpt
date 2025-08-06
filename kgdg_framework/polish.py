import json
import argparse
import multiprocessing
import time
import traceback
import google.generativeai as genai
from google.generativeai import GenerativeModel

# === Command line args ===
parser = argparse.ArgumentParser()
parser.add_argument('--cores', default=3, type=int, help="Number of parallel workers")
parser.add_argument('--nums', default=10, type=int, help="Number of data items to process")
parser.add_argument('--suffix', default=None, type=str, help="Suffix for input/output file names")
parser.add_argument('--api_key', required=False, type=str, help="Google Gemini API key")
args = parser.parse_args()

# === Gemini API setup ===
API_KEY = "AIzaSyA5h1sYXwxvZrl4RUPNISLMx1WLt2B0wyE" 
genai.configure(api_key=API_KEY)
model = GenerativeModel("gemini-1.5-flash")  # You can use latest model as available

# === Polishing prompts for the Indian legal domain ===
REFINE_PROMPT = """You are an expert in Indian civil law.

You are given a JSON dictionary of legal statutes and references cited in a legal reasoning. Each key is the name/title of a statute or rule, and each value is its content as used in explanation.

{JSON}

Some statute contents may be incorrect or slightly paraphrased. Please correct each value to match the official, authoritative text for each statute as per Indian law. Return the corrected dictionary as JSON only, with no extra explanation.
"""

CORRECT_PROMPT = """You are an expert in Indian civil law.

You are given a legal Q&A in JSON format, containing fields:
- instruction: how to answer
- question: legal question
- answer: generated answer
- references: legal statutes and sections (as dictionary)
- reasoning: step-by-step legal logic

{JSON}

Based on the question, the references, and their official content, improve or correct the reasoning and answer if necessary for legal soundness and clarity. If the existing reasoning and answer are already fully accurate, return the original JSON unchanged. Always return only valid JSON, without extra comment.
"""

# === Helper functions ===

def gemini_generate(prompt: str) -> str:
    """Call Gemini LLM synchronously with prompt and return text."""
    for attempt in range(3):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Gemini API error (attempt {attempt + 1}/3): {e}")
            time.sleep(2 ** attempt)
    print("Gemini API failed after retries.")
    return None

def clean_json_response(reply: str):
    """Remove code-fencing and parse JSON."""
    reply = reply.strip()
    if reply.startswith("```json"):
        reply = reply[len("```json"):].strip()
    elif reply.startswith("```"):
        reply = reply[len("```"):].strip()
    if reply.endswith("```"):
        reply = reply[:-3].strip()
    return json.loads(reply)

def extract_qna_fields(item):
    return {
        "instruction": item.get("instruction", ""),
        "question": item.get("question", ""),
        "answer": item.get("answer", ""),
        "reasoning": item.get("reasoning", ""),
        "references": item.get("references", {}),
    }

def polish_item(item):
    """Polish references and Q&A for one item."""
    try:
        # Step 1: Refine references
        ref_prompt = REFINE_PROMPT.format(JSON=json.dumps(item.get("references", {}), ensure_ascii=False))
        refined_refs_raw = gemini_generate(ref_prompt)
        if refined_refs_raw is None:
            return None
        refined_refs = clean_json_response(refined_refs_raw)

        if refined_refs != item.get("references", {}):
            item["references_old"] = item.get("references", {}).copy()
            item["references"] = refined_refs

            # Step 2: Correct the answer and reasoning
            correction_input = extract_qna_fields(item)
            corr_prompt = CORRECT_PROMPT.format(JSON=json.dumps(correction_input, ensure_ascii=False))
            corrected_output_raw = gemini_generate(corr_prompt)
            if corrected_output_raw is None:
                return None
            corrected_output = clean_json_response(corrected_output_raw)
            item["answer_old"] = item.get("answer", "")
            item["reasoning_old"] = item.get("reasoning", "")
            item["answer"] = corrected_output.get("answer", item.get("answer", ""))
            item["reasoning"] = corrected_output.get("reasoning", item.get("reasoning", ""))

        return item

    except Exception as e:
        print(f"Error polishing item: {e}\n{traceback.format_exc()}")
        return None

# === Main script flow ===

def main():
    ori_path = f"./generation/GEN{'-' + args.suffix if args.suffix else ''}.json"
    tar_path = f"./generation/POL{'-' + args.suffix if args.suffix else ''}.json"

    with open(ori_path, "r", encoding="utf-8") as fr:
        data_list = json.load(fr)

    total = len(data_list)
    print(f"Loaded {total} records from {ori_path}")

    inputs = data_list[: min(args.nums, total)]

    start_time = time.time()
    with multiprocessing.Pool(processes=args.cores) as pool:
        results = pool.map(polish_item, inputs)

    polished = [r for r in results if r is not None]
    print(f"Polishing finished. Success: {len(polished)}/{len(inputs)} items processed.")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")

    with open(tar_path, "w", encoding="utf-8") as fw:
        json.dump(polished, fw, ensure_ascii=False, indent=4)

    print(f"Saved polished data to {tar_path}")

if __name__ == "__main__":
    main()

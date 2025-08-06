import json
import argparse
import multiprocessing
import time
import traceback
import google.generativeai as genai
from google.generativeai import GenerativeModel

# === Command line args ===
parser = argparse.ArgumentParser()
parser.add_argument('--cores', default=4, type=int, help="Number of parallel workers")
parser.add_argument('--nums', default=10, type=int, help="Number of data items to process")
parser.add_argument('--suffix', default=None, type=str, help="Suffix for input/output files")
parser.add_argument('--api_key', required=False, type=str, help="Google Gemini API key")
args = parser.parse_args()

# === Gemini API setup ===
API_KEY = "AIzaSyA5h1sYXwxvZrl4RUPNISLMx1WLt2B0wyE" 
genai.configure(api_key=API_KEY)
model = GenerativeModel("gemini-1.5-flash")  # Use your preferred Gemini model

# === Verification prompt for Indian legal domain in English ===
VERIFY_PROMPT = """
You are a legal expert specializing in Indian civil law.

Given a JSON-formatted legal question-answer pair containing:
- instruction: the requirement for the question
- question: the legal question asked
- answer: the provided answer to the question
- references: legal statutes or sections cited (as a dictionary)
- reasoning: the reasoning path/logic for the answer

Please evaluate the correctness of the reasoning and answer based on the question and references.

Return a JSON object with two fields:
- verify: either "correct" or "incorrect"
- message: a brief explanation justifying your judgment

Respond with JSON only, no extra commentary.

Input JSON:
{JSON}
"""

# === Helper functions ===

def gemini_generate(prompt: str) -> str:
    """Call Gemini LLM synchronously and return text response."""
    # Simple retry logic
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
    """Strip markdown code blocks and parse JSON."""
    reply = reply.strip()
    if reply.startswith("```json"):
        reply = reply[len("```json"):].strip()
    elif reply.startswith("```"):
        reply = reply[len("```"):].strip()
    if reply.endswith("```"):
        reply = reply[:-3].strip()
    return json.loads(reply)

def prepare_input(data_item):
    """Create a minimal JSON for verification prompt."""
    return {
        "instruction": data_item.get("instruction", ""),
        "question": data_item.get("question", ""),
        "answer": data_item.get("answer", ""),
        "references": data_item.get("references", {}),
        "reasoning": data_item.get("reasoning", ""),
    }

def verify_item(item):
    """Verify one data item, add verify and message fields."""
    try:
        input_json = prepare_input(item)
        prompt = VERIFY_PROMPT.format(JSON=json.dumps(input_json, ensure_ascii=False))

        raw_resp = gemini_generate(prompt)
        if raw_resp is None:
            return None

        resp_json = clean_json_response(raw_resp)

        # Add verify info to item
        item["verify"] = resp_json.get("verify", "incorrect").lower()
        item["message"] = resp_json.get("message", "")
        return item

    except Exception as e:
        print(f"Error verifying item: {e}\n{traceback.format_exc()}")
        return None

# === Main ===

def main():
    input_path = f"./generation/POL{'-' + args.suffix if args.suffix else ''}.json"
    output_path = f"./generation/VER{'-' + args.suffix if args.suffix else ''}.json"

    with open(input_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    total = len(data_list)
    print(f"Loaded {total} records from {input_path}")

    inputs = data_list[: min(args.nums, total)]

    start_time = time.time()
    with multiprocessing.Pool(processes=args.cores) as pool:
        results = pool.map(verify_item, inputs)

    verified = [r for r in results if r is not None]
    correct_count = sum(1 for r in verified if r.get("verify") == "correct")
    print(f"Verification finished. Correct: {correct_count}/{len(verified)} items.")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")

    with open(output_path, "w", encoding="utf-8") as fw:
        json.dump(verified, fw, ensure_ascii=False, indent=4)

    print(f"Saved verified data to {output_path}")

if __name__ == "__main__":
    main()

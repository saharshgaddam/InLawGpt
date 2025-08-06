import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed_file', default="seed.json", type=str, help="Original seed dataset (json)")
parser.add_argument('--verified_file', default="./generation/VER.json", type=str, help="Verified synthetic Q&A")
parser.add_argument('--output_file', default="./generation/MIX.json", type=str, help="Output mixed dataset")
parser.add_argument('--filter_correct', action='store_true', help="Only keep correct-verified synthetic samples")
args = parser.parse_args()

# Load seed/original data
with open(args.seed_file, "r", encoding="utf-8") as f:
    seed_data = json.load(f)
print(f"Loaded {len(seed_data)} seed items.")

for item in seed_data:
    if "reasoning" not in item:
        item["reasoning"] = ""
    if "references" not in item:
        item["references"] = {}
# Load synthetic (verified) data
with open(args.verified_file, "r", encoding="utf-8") as f:
    ver_data = json.load(f)
print(f"Loaded {len(ver_data)} verified synthetic items.")

# Optionally filter synthetic for only 'correct'
if args.filter_correct:
    ver_data = [d for d in ver_data if d.get("verify") == "correct"]
    print(f"Filtered synthetic data: {len(ver_data)} are verified as correct.")

# Combine, removing duplicates (by question field)
combined = {d["question"]: d for d in seed_data}  # seed data entries first
for d in ver_data:
    q = d["question"]
    if q not in combined:  # Only add if not already in seed
        combined[q] = d

mix_data = list(combined.values())
print(f"Mixture dataset size: {len(mix_data)}")

# Save
with open(args.output_file, "w", encoding="utf-8") as f:
    json.dump(mix_data, f, ensure_ascii=False, indent=4)

print(f"Saved mixture dataset to {args.output_file}")

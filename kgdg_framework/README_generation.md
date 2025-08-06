# Legal Question-Answer Data Generation Script (`generation.py`)

## Overview

`generation.py` is a Python script designed to generate synthetic legal question-answer (Q&A) pairs based on a seed dataset and a legal knowledge base. It leverages Google Gemini's large language model to produce new, realistic civil law Q&A samples specifically tailored for Indian civil legal documents.

## Key Features

- **Seed-based generation:** Starts from human-curated seed questions and answers to guide the generation.
- **Reference-based grounding:** Samples real legal cases from a knowledge base (JSONL format) and uses them as factual foundations.
- **Context-aware prompting:** Uses carefully crafted prompts to instruct the LLM on how to generate high-quality, legally sound Q&A pairs.
- **Additional output fields:** Generated data includes reasoning and legal statute references for transparency and traceability.
- **Robust error handling and retry logic:** Handles API failures, rate limits with retries and exponential backoff.
- **Parallel processing support:** Uses multiprocessing to speed up generation across multiple CPU cores.
- **Token limit awareness:** Truncates input data intelligently to stay within model context size limits.
- **Automatic resumption:** Appends results to existing output files, allowing safe continuation if interrupted.

## Directory Structure & Key Files

- `kgdg_framework/generation.py`: The main script to run legal Q&A data generation.
- `extraction/all_cases.jsonl`: The legal knowledge base containing reference cases in JSONL format.
- `seed.json`: The seed legal question-answer pairs used as generation starting points.
- `generation/`: Output directory where generated Q&A JSON files are saved.

## Usage

### Requirements

- Python 3.7+
- `google-generativeai` SDK installed:  
pip install google-generativeai

- Google Gemini API access with a valid API key.

### Running the Script


python3 kgdg_framework/generation.py --api_key YOUR_GEMINI_API_KEY --cores 4 --nums 20 --suffix batch1


- `--api_key`: Your Google Gemini API key (can be hardcoded in the script if preferred).
- `--cores`: Number of parallel worker processes.
- `--nums`: Number of seed Q&A to process for generation in this run.
- `--suffix`: Optional suffix for output filename to distinguish runs.

### Output

The generated questions and answers will be saved in JSON files under the `generation/` directory, e.g., `GEN-batch1.json`.

Each generated record includes:

- `instruction`
- `question`
- `answer`
- `reasoning`
- `references` (legal statutes cited)
- Metadata fields such as source seed questions and reference document index.

## Extending and Customizing

- Modify prompts (`CONTROL_PROMPT`, `GENERATE_PROMPT`) inside `generation.py` to adjust generation instructions or language style.
- Replace or expand the knowledge base with more Indian legal cases for richer generation.
- Adapt the script for other legal domains (criminal law, writ petitions) by updating sampling functions.
- Integrate with other LLMs by replacing the Gemini API calls.

## License

[Your License Here]

## Contact

For questions or contributions, please reach out at [your email/contact info].



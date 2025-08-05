# extraction_from_metadata.py

This script automates the extraction and structuring of Indian High Court judgment data based on metadata JSON files.

## Features

- Downloads judgment PDFs from AWS S3 using metadata information.
- Extracts text from PDFs using pdfplumber.
- Uses Google Gemini LLM API to parse and structure the text into useful sections:
  - Judgment Facts
  - Reasoning & Background
  - Verdict Results
  - Relevant Statutes  
- Validates output against a JSON schema.
- Avoids re-processing already-extracted cases.

## Usage

1. Ensure you have Python, AWS CLI, and dependencies installed (`pdfplumber`, `jsonschema`, `requests`).
2. Place your metadata JSONs and configure paths as needed in the script.
3. Run:
4. Outputs are saved as newline-delimited JSON (`.jsonl`).

## Configuration

- Set your AWS bucket and source folders in the script.
- Requires a Google Gemini API key for LLM extraction.

"""
Offline pipeline: reads terveys_markdown/, calls an LLM to extract a structured
JSON schema per article, saves results to terveys_structured/.

Resumes automatically — already-processed files are skipped.
Handles Groq rate limits by parsing the retry time from the 429 error and
sleeping the correct amount. Stops cleanly when the daily token limit is hit
so the next run picks up where this one left off.
"""

import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from vector_db import extract_article_content

load_dotenv()

MARKDOWN_DIR = "terveys_markdown"
STRUCTURED_DIR = "terveys_structured"
INPUT_CHAR_LIMIT = 3000       # truncate input to keep token usage low
SLEEP_BETWEEN_REQUESTS = 2.5  # baseline sleep between requests (seconds)
MAX_RETRIES = 4               # retries per article on rate-limit errors
DAILY_LIMIT_THRESHOLD = 5     # stop after this many consecutive TPD failures

SCHEMA_PROMPT = """Analysoi tämä Terveyskirjaston artikkeli ja palauta JSON-objekti. Palauta VAIN validi JSON, ei muuta tekstiä, ei selityksiä.

{{
  "title": "artikkelin otsikko",
  "article_type": "disease|symptom|treatment|drug|test|procedure|general",
  "conditions": ["sairaus tai tila"],
  "symptoms": ["oire"],
  "risk_factors": ["riskitekijä"],
  "treatments": ["hoitomuoto tai lääke"],
  "contraindications": ["vasta-aihe"],
  "related_conditions": ["liittyvä sairaus"],
  "when_to_see_doctor": ["milloin lääkäriin"],
  "summary": "1-2 lausetta"
}}

Käytä tyhjiä listoja [] kentille jotka eivät sovi artikkeliin. Älä keksi tietoja.

Artikkeli:
{article}"""


def get_llm():
    # Swap this one line to change LLM provider
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0)


def _parse_retry_after(error_message: str) -> float:
    """Extract the suggested retry delay (seconds) from a Groq 429 error."""
    m = re.search(r'try again in (\d+)m([\d.]+)s', error_message)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2)) + 2
    m = re.search(r'try again in ([\d.]+)s', error_message)
    if m:
        return float(m.group(1)) + 2
    m = re.search(r'try again in (\d+)ms', error_message)
    if m:
        return float(m.group(1)) / 1000 + 1
    return 65  # fallback


def _extract_json(text: str) -> dict:
    """Parse JSON from LLM output, tolerating leading/trailing prose."""
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in output: {text[:120]!r}")
    return json.loads(text[start:end + 1])


def extract_structure(content: str, llm) -> dict:
    prompt = ChatPromptTemplate.from_template(SCHEMA_PROMPT)
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"article": content[:INPUT_CHAR_LIMIT]})
    return _extract_json(raw)


def extract_with_retry(content: str, llm) -> dict:
    for attempt in range(MAX_RETRIES):
        try:
            return extract_structure(content, llm)
        except Exception as e:
            msg = str(e)
            if '429' not in msg:
                raise
            is_daily = 'per day' in msg or 'TPD' in msg
            if is_daily:
                raise  # bubble up so main() can decide to stop
            wait = _parse_retry_after(msg)
            print(f"  TPM limit, waiting {wait:.0f}s (attempt {attempt + 1}/{MAX_RETRIES})...")
            time.sleep(wait)
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries")


def main():
    os.makedirs(STRUCTURED_DIR, exist_ok=True)
    md_files = sorted(Path(MARKDOWN_DIR).glob("*.md"))
    total = len(md_files)

    already_done = sum(1 for f in md_files if (Path(STRUCTURED_DIR) / (f.stem + ".json")).exists())
    remaining = total - already_done
    print(f"Found {total} articles — {already_done} already done, {remaining} to process\n")

    llm = get_llm()
    success = skip = fail = 0
    consecutive_daily_fails = 0

    for i, md_path in enumerate(md_files):
        out_path = Path(STRUCTURED_DIR) / (md_path.stem + ".json")
        if out_path.exists():
            skip += 1
            continue

        raw = md_path.read_text(encoding="utf-8")
        url, title, content = extract_article_content(raw)
        if not content:
            skip += 1
            continue

        try:
            structured = extract_with_retry(content, llm)
            structured["source"] = md_path.name
            structured["url"] = url
            out_path.write_text(
                json.dumps(structured, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            success += 1
            consecutive_daily_fails = 0
            time.sleep(SLEEP_BETWEEN_REQUESTS)

        except Exception as e:
            msg = str(e)
            if '429' in msg and ('per day' in msg or 'TPD' in msg):
                consecutive_daily_fails += 1
                wait = _parse_retry_after(msg)
                print(f"  Daily limit hit, waiting {wait:.0f}s... ({consecutive_daily_fails}/{DAILY_LIMIT_THRESHOLD})")
                time.sleep(wait)
                if consecutive_daily_fails >= DAILY_LIMIT_THRESHOLD:
                    print(f"\nDaily token limit reached. Progress saved ({success} new articles).")
                    print("Re-run tomorrow — the script will resume from where it stopped.")
                    break
            else:
                print(f"  FAIL {md_path.name}: {e}")
                fail += 1
                consecutive_daily_fails = 0

        if (i + 1) % 100 == 0:
            print(f"[{i + 1}/{total}] ok={success}  skip={skip}  fail={fail}")

    print(f"\nDone. ok={success}  skip={skip}  fail={fail}")
    print(f"Results in {STRUCTURED_DIR}/  ({sum(1 for _ in Path(STRUCTURED_DIR).glob('*.json'))} total files)")


if __name__ == "__main__":
    main()

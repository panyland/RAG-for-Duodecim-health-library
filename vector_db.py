import json
import os
import re
import shutil
import sys

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DB_LOCATION = "./terveys_chroma_db"
COLLECTION_NAME = "terveys_articles"
STRUCTURED_DIR = "./terveys_structured"
BATCH_SIZE = 5000

_URL_RE = re.compile(r'\[Suoraan sisältöön\]\((https://www\.terveyskirjasto\.fi/[^#)]+)')
_H1_RE = re.compile(r'^# .+', re.MULTILINE)
_CONTENT_END_RE = re.compile(r'^# Katso myös|^## Ajankohtaista', re.MULTILINE)


def extract_article_content(text: str) -> tuple[str, str, str]:
    """Strip nav/footer boilerplate. Returns (url, title, clean_content)."""
    url = ""
    m = _URL_RE.search(text)
    if m:
        url = m.group(1).strip()

    h1 = _H1_RE.search(text)
    if not h1:
        return url, "", ""

    title = h1.group(0).lstrip("#").strip()
    start = h1.start()
    end = _CONTENT_END_RE.search(text, start)
    content = text[start: end.start() if end else len(text)]
    return url, title, content.strip()


def build_context_prefix(metadata: dict) -> str:
    parts = [metadata[k] for k in ("h1", "h2", "h3") if k in metadata]
    return " > ".join(parts)


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


def get_vector_store(embeddings=None):
    if embeddings is None:
        embeddings = get_embeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_LOCATION,
        embedding_function=embeddings,
    )


def chunk_markdown_files(markdown_dir="terveys_markdown", structured_dir=None):
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        return_each_line=False,
        strip_headers=True,
    )

    documents = []
    skipped = 0
    for filename in os.listdir(markdown_dir):
        if not filename.endswith(".md"):
            continue
        with open(os.path.join(markdown_dir, filename), encoding="utf-8") as f:
            raw = f.read()

        url, title, content = extract_article_content(raw)
        if not content:
            skipped += 1
            continue

        chunks = splitter.split_text(content)

        structured_meta = {}
        if structured_dir:
            json_path = os.path.join(structured_dir, filename.replace(".md", ".json"))
            if os.path.exists(json_path):
                with open(json_path, encoding="utf-8") as jf:
                    data = json.load(jf)
                structured_meta = {
                    "article_type": data.get("article_type", ""),
                    "conditions": ", ".join(data.get("conditions", [])),
                }

        for chunk in chunks:
            chunk.metadata["source"] = filename
            chunk.metadata["url"] = url
            chunk.metadata["article_title"] = title
            chunk.metadata.update(structured_meta)

            prefix = build_context_prefix(chunk.metadata)
            if prefix:
                chunk.page_content = f"[{prefix}]\n\n{chunk.page_content}"

        documents.extend(chunks)

    if skipped:
        print(f"Skipped {skipped} files with no extractable content")
    return documents


def batch(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


if __name__ == "__main__":
    reset = "--reset" in sys.argv

    if reset and os.path.exists(DB_LOCATION):
        shutil.rmtree(DB_LOCATION)
        print(f"Deleted {DB_LOCATION}")

    use_structured = os.path.isdir(STRUCTURED_DIR)
    print(f"Structured metadata: {'enabled (' + STRUCTURED_DIR + ')' if use_structured else 'not found, skipping'}")

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)

    print("Chunking markdown files...")
    docs = chunk_markdown_files(
        "terveys_markdown",
        structured_dir=STRUCTURED_DIR if use_structured else None,
    )
    print(f"Total chunks: {len(docs)}")

    for i, chunk in enumerate(batch(docs, BATCH_SIZE)):
        print(f"Adding batch {i + 1} ({len(chunk)} documents)...")
        vector_store.add_documents(chunk)

    print("Done.")

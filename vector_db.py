import json
import os
import re
import shutil
import sys

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# E5 accepts 512 tokens. Finnish medical text in this corpus runs ~2.5 chars/token
# at the dense end, so 1100 chars + context prefix stays comfortably under the limit.
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
DB_LOCATION = "./terveys_chroma_db"
COLLECTION_NAME = "terveys_articles"
STRUCTURED_DIR = "./terveys_structured"
BATCH_SIZE = 5000

CHUNK_SIZE = 1100
CHUNK_OVERLAP = 150
MIN_CHUNK_CHARS = 15

# Drug-leaflet sections that carry no health information. Verified to appear only in
# /far articles, so dropping them by name touches nothing else. Marketer names in
# particular ("ASTELLAS PHARMA") are Latin-script and read as English to a multilingual
# embedder, so English queries matched them ahead of actual Finnish clinical prose.
# Clinical leaflet sections (Käyttötarkoitus, Annostus, Haittavaikutukset,
# Lääkkeen ottaminen, Vaikutustapa, Raskaus ja imetys, Vaikuttava aine) are kept.
LEAFLET_BOILERPLATE_SECTIONS = {
    "Markkinoija",
    "Hinta ja pakkaustiedot",
    "Säilytys",
    "Ulkonäkö",
}

_LINK_RE = re.compile(r'\[[^\]]*\]\([^)]*\)')
PROSE_KEEP_CHARS = 80  # non-link text above this means the chunk says something itself
HEADER_BLOCK_MAX_CHARS = 400  # share widget sits <=170 chars below the H1 in every article

_URL_RE = re.compile(r'\[Suoraan sisältöön\]\((https://www\.terveyskirjasto\.fi/[^#)]+)')
_H1_RE = re.compile(r'^# .+', re.MULTILINE)
_CONTENT_END_RE = re.compile(r'^# Katso myös|^## Ajankohtaista', re.MULTILINE)
_SHARE_RE = re.compile(r'^.*(?:javascript:void|Jaa Facebookissa).*$', re.MULTILINE)


def strip_header_block(content: str) -> str:
    """Remove the byline/date lines and social-share widget wedged between the H1
    and the article body. Returns content unchanged if the widget isn't found,
    so the function degrades safely on unexpected markup."""
    nl = content.find("\n")
    if nl == -1:
        return content

    head, rest = content[: nl + 1], content[nl + 1:]
    m = _SHARE_RE.search(rest)
    if not m or m.start() > HEADER_BLOCK_MAX_CHARS:
        return content
    return head + rest[m.end():].lstrip("\n")


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
    return url, title, strip_header_block(content.strip())


def is_nav_list(body: str) -> bool:
    """True for a chunk that is mostly a bullet list of links — an article's own table
    of contents, which every article repeats and which answers nothing."""
    lines = [l.strip() for l in body.strip().split("\n") if l.strip()]
    if len(lines) < 3:
        return False
    bullets = [l for l in lines if l.startswith(("*", "-", "1.", "2."))]
    if len(bullets) < 3 or len(bullets) / len(lines) < 0.7:
        return False
    # A line-ratio test alone misfires on short articles that introduce a link list with
    # real prose. Keep anything with meaningful text left once the links are removed.
    prose = _LINK_RE.sub("", body).replace("*", " ").replace("#", " ").strip()
    if len(prose) >= PROSE_KEEP_CHARS:
        return False
    linky = sum(
        1 for b in bullets
        if _LINK_RE.search(b) and len(_LINK_RE.sub("", b).strip(" *-").strip()) < 15
    )
    return linky / len(bullets) >= 0.7


def is_useful_chunk(body: str, metadata: dict) -> bool:
    """Short bodies are almost always drug package-leaflet field values sitting under
    a subheading ("[HYDRAN > Ulkonäkö]" -> "-"). A short body directly under the
    article title is a glossary definition ("[aqua]" -> "vesi"), which is real content
    and exactly the kind of term lookup this corpus is useful for."""
    if metadata.get("h2") in LEAFLET_BOILERPLATE_SECTIONS:
        return False
    if is_nav_list(body):
        return False
    if len(body) >= MIN_CHUNK_CHARS:
        return True
    is_glossary = "h2" not in metadata and "h3" not in metadata
    return is_glossary and len(body) >= 3


def build_context_prefix(metadata: dict) -> str:
    parts = [metadata[k] for k in ("h1", "h2", "h3") if k in metadata]
    return " > ".join(parts)


def _resolve_device() -> str:
    """Prefer the GPU when one is present; MDAI_DEVICE overrides for CPU-only machines."""
    override = os.environ.get("MDAI_DEVICE")
    if override:
        return override
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def get_embeddings():
    # E5 requires asymmetric prefixes: documents are "passage: ", queries are "query: ".
    # Using the same prefix on both sides, or none, measurably degrades retrieval.
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": _resolve_device()},
        encode_kwargs={
            "prompt": "passage: ",
            "normalize_embeddings": True,
            "batch_size": 64,
        },
        query_encode_kwargs={
            "prompt": "query: ",
            "normalize_embeddings": True,
        },
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
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        return_each_line=False,
        strip_headers=True,
    )
    # The header splitter has no size cap, so a long section becomes one oversized
    # chunk whose tail is silently truncated at embedding time. This second pass caps it.
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents = []
    skipped_files = 0
    dropped_chunks = 0

    for filename in os.listdir(markdown_dir):
        if not filename.endswith(".md"):
            continue
        with open(os.path.join(markdown_dir, filename), encoding="utf-8") as f:
            raw = f.read()

        url, title, content = extract_article_content(raw)
        if not content:
            skipped_files += 1
            continue

        sections = header_splitter.split_text(content)

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

        for section in sections:
            section.metadata["source"] = filename
            section.metadata["url"] = url
            section.metadata["article_title"] = title
            section.metadata.update(structured_meta)

        # Prefix after the size split so every sub-chunk keeps its heading trail.
        for chunk in size_splitter.split_documents(sections):
            body = chunk.page_content.strip()
            if not is_useful_chunk(body, chunk.metadata):
                dropped_chunks += 1
                continue

            prefix = build_context_prefix(chunk.metadata)
            chunk.page_content = f"[{prefix}]\n\n{body}" if prefix else body
            documents.append(chunk)

    if skipped_files:
        print(f"Skipped {skipped_files} files with no extractable content")
    if dropped_chunks:
        print(f"Dropped {dropped_chunks} boilerplate/empty chunks")
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

    print(f"Loading embedding model: {EMBEDDING_MODEL} on {_resolve_device()}")
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
